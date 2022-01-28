% test_LDSBernoulli_LaplaceEvidenceOptim.m
%
% 1. Sample from a latent Gaussian linear dynamical system (LDS) model with Bernoulli observations.
% 2. Compute ML fit via brute force evidence optimization, using Laplace approximation

% Basic equations:
% -----------------
% X_t = A*X_{t-1} + eps_x,  eps_x ~ N(0,Q)   % latents
% Y_t ~ Bernoulli(f(C*X_t)  % observations

addpath inference
addpath utils

% Set dimensions
nz = 2;  % dimensionality of latent z
ny = 5; % dimensionality of observation y
ntrials = 1000;  % number of time bins

% Set model parameters
% --------------------

% Set dynamics matrix A

% % Generate random A
% sA = sort((rand(nz,1)*0.02+.98), 'descend');  % eigenvalues of A
% uA = orth(randn(nz)); % singular vectors
% A = uA*diag(sA)*uA';

% Use rotation matrix if nz = 2
thet = pi/100;
A = [cos(thet), sin(thet); -sin(thet), cos(thet)]*.995;

% Dynamics noise covariance
Q = randn(nz); Q = .01*(Q'*Q+eye(nz)); % dynamics noise covariance
sampznoise = @(n)(mvnrnd(zeros(n,nz),Q)'); % x noise

% Observation matrix C
C = randn(ny,nz); % observation matrix

% Use discrete Lyapunov equation solver to compute asymptotic covariance
P = dlyap(A,Q);

% function handles for logistic
flogistic = @(x)(1./(1+exp(-x)));


%% Simulate data from the LDS-Bernoulli model

zz = zeros(nz,ntrials);
xx = zeros(ny,ntrials);
yy = zeros(ny,ntrials);

zz(:,1) = mvnrnd(zeros(1,nz),Q)';
xx(:,1) = C*zz(:,1);
yy(:,1) = rand(ny,1)<flogistic(xx(:,1));

for jj = 2:ntrials
    zz(:,jj) = A*zz(:,jj-1)+sampznoise(1);
    xx(:,jj) = C*zz(:,jj);
    yy(:,jj) = rand(ny,1)<flogistic(xx(:,jj));
end

%% Compute MAP estimate of latents using true params

% % Compute posterior mean given true parameters
[zzmap,H,logevtrue] = computeMAP_LDSBernoulli(zeros(nz,ntrials),yy,A,C,Q);
fprintf('log-evidence given true params: %.2f\n', logevtrue);

% Plot true and inferred latents
colr = get(gca,'colororder');
ii = 1:min(ntrials,1000);
h = plot(ii, zz(:,ii)', ii, zzmap(:,ii), '--', 'linewidth', 2);
for jj = 1:nz; set(h(jj+nz),'color', colr(jj,:)); end
xlabel('time (bin)');
ylabel('latent'); drawnow;

%% Compute max-evidence estimate of A and C using laplace approx

% 1. Initialize from true params
mmtrue = struct('A',A, 'C',C, 'Q', Q);  % make initial struct
mmtrueFit = runMaxLapEvidence_LDSBernoulli(yy,mmtrue);

% 2. Initialize from somewhere else
mm0 = struct('A',A*.1, 'C',C*.1, 'Q', Q);  % make initial struct
mm1 = runMaxLapEvidence_LDSBernoulli(yy,mm0);

fprintf('Final log-evidence, initialized from true params:  %.2f\n', mmtrueFit.logEvidence);
fprintf('Final log-evidence, initialized from farther away: %.2f\n', mm1.logEvidence);


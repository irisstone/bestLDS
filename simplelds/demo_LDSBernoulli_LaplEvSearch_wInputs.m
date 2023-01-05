% test_LDSBernoulli_LaplEvSearch_wInputs.m
%
% 1. Sample from a latent Gaussian linear dynamical system (LDS) model 
%    WITH INPUTS, with Bernoulli observations. 
% 2. Compute max-evidence fit of LDS parameters via brute-force
%    optimization of evidence, evaluated using Laplace approximation 

% Basic equations:
% -----------------
% X_t = A*X_{t-1} + B*S_t + eps_x,  eps_x ~ N(0,Q)   % latents
% Y_t ~ Bernoulli(f(C*X_t + D*S_t)  % observations

addpath inference
addpath utils

% Set dimensions
nz = 2;  % dimensionality of latent z
ny = 5;  % dimensionality of observation y
ns = 1; % dimensionality of inputs
ntrials = 1000;  % number of time bins

% Set model parameters
% --------------------

% Set dynamics matrix A
if nz == 2 
    % Use rotation matrix if nz = 2
    thet = pi/25;
    A = [cos(thet), sin(thet); -sin(thet), cos(thet)]*.99;
else
    % Generate random stable A
    A = randn(nz);
    [u,s] = eig(A,'vector'); % get eigenvectors and eigenvals
    s = s/max(abs(s))*.98; % set largest eigenvalue to lie inside unit circle (enforcing stability)
    s(real(s)<0) = -s(real(s)<0); % set real parts to be positive (encouraging smoothness)
    A = real(u*(diag(s)/u));  % reconstruct A
end

% Dynamics noise covariance
Q = randn(nz); Q = .01*(Q'*Q+eye(nz)); % dynamics noise covariance
sampznoise = @(n)(mvnrnd(zeros(n,nz),Q)'); % x noise

% Observation matrix C
C = randn(ny,nz); % observation matrix

% Input weights B & D
B = randn(nz,ns)*.4;
D = randn(ny,ns)*.55;

% % Use discrete Lyapunov equation solver to compute asymptotic covariance, if desired
% P = dlyap(A,Q);

% function handles for logistic
flogistic = @(x)(1./(1+exp(-x)));


%% Simulate data from the LDS-Bernoulli-with-Inputs model

% Initialize latents and outputs
zz = zeros(nz,ntrials); xx = zeros(ny,ntrials); yy = zeros(ny,ntrials);

% Set inputs
ss = .8*randn(ns,ntrials); % external inputs
muz = B*ss;  % additive intput to latents
muy = D*ss;  % additive intput to observations

zz(:,1) = mvnrnd(zeros(1,nz),Q)' + muz(:,1);  % latents
xx(:,1) = C*zz(:,1) + muy(:,1); % projected latents
yy(:,1) = rand(ny,1)<flogistic(xx(:,1)); % observations

for jj = 2:ntrials
    zz(:,jj) = A*zz(:,jj-1)+sampznoise(1) + muz(:,jj);
    xx(:,jj) = C*zz(:,jj) + muy(:,jj);
    yy(:,jj) = rand(ny,1)<flogistic(xx(:,jj));
end

%% Compute MAP estimate of latents using true params

% % Compute posterior mean given true parameters
[zzmap,H,logevtrue] = computeMAP_LDSBernoulli_wInputs(zeros(nz,ntrials),yy,ss,A,B,C,D,Q);
fprintf('log-evidence given true params: %.2f\n', logevtrue);

% Plot true and inferred latents
colr = get(gca,'colororder');
ii = 1:min(ntrials,1000);
h = plot(ii, zz(:,ii)', ii, zzmap(:,ii), '--', 'linewidth', 2);
for jj = 1:nz; set(h(jj+nz),'color', colr(jj,:)); end
xlabel('time (bin)');
ylabel('latent'); drawnow;

% % Compute MSE in recovered latent
% mse = [mean((zz(1,:)-zzmap(1,:)).^2), mean((zz(2,:)-zzmap(2,:)).^2)]


%% Compute max-evidence estimate of A, B, C, D using laplace approx

% 1. Initialize from true params
mmtrue = struct('A',A, 'B', B, 'C',C, 'D', D, 'Q', Q);  % make initial struct
mmtrueFit = runMaxLapEvidence_LDSBernoulli_wInputs(yy,ss,mmtrue); % optimize evidence, starting from true params

% 2. Initialize from somewhere else
mm0 = struct('A',A*.1, 'B', B*0.1, 'C',C*.1, 'D', D*0.1, 'Q', Q);  % make initial struct
mm1 = runMaxLapEvidence_LDSBernoulli_wInputs(yy,ss,mm0); % optimize evidence, starting far from true params

fprintf('Final log-evidence, initialized from true params:  %.2f\n', mmtrueFit.logEvidence);
fprintf('Final log-evidence, initialized from farther away: %.2f\n', mm1.logEvidence);


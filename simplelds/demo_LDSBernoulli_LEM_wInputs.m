% demo_LDSBernoulli_LaplEvSearch_wInputs.m
%
% 1. Sample from a latent Gaussian linear dynamical system (LDS) model 
%    WITH INPUTS, with Bernoulli observations. 
% 2. Compute max-evidence fit of LDS parameters via brute-force
%    optimization of evidence, evaluated using Laplace approximation 

% Basic equations:
% -----------------
% X_t = A*X_{t-1} + B*S_t + eps_x,  eps_x ~ N(0,Q)   % latents
% Y_t ~ Bernoulli(f(C*X_t + D*S_t)  % observations

addpath inference_Bernoulli/
addpath utils

% Set dimensions
nz = 3;  % dimensionality of latent z
ny = 1;  % dimensionality of observation y
nu = 4; % dimensionality of inputs

nT = 1000;  % number of time bins

% Set model parameters
% --------------------

% Set dynamics matrix A
if nz == 2 
    % Use rotation matrix if nz = 2
    thet = pi/100;
    A = [cos(thet), sin(thet); -sin(thet), cos(thet)]*.99;
else
    % Generate random stable A
    A = randn(nz);
    [u,s] = eig(A,'vector'); % get eigenvectors and eigenvals
    s = s/max(abs(s))*.99; % set largest eigenvalue to lie inside unit circle (enforcing stability)
    s(real(s)<0) = -s(real(s)<0); % set real parts to be positive (encouraging smoothness)
    A = real(u*(diag(s)/u));  % reconstruct A
end

% Dynamics noise covariance
Q = randn(nz); Q = .01*(Q'*Q+eye(nz)); % dynamics noise covariance
Q0 = eye(nz);  % covariance of initial latents

% Make function for sampling z noise
sampznoise = @(n)(mvnrnd(zeros(n,nz),Q)'); % x noise

% Observation matrix C
C = 0.25*randn(ny,nz); % observation matrix
% --------------------
% NOTE (critical observation): LEM tends to increase then DECREASE the ELBO
% if norm of C is too large (eg stdev >= 1). 

disp(norm(C));
disp(std(C));
% --------------------

% Set input weights B & D
B = randn(nz,nu)*.25;
D = randn(ny,nu)*.25;


%% Sample data from the LDS-Bernoulli-with-Inputs model

% Set inputs
uu = 0.5*randn(nu,nT); % external inputs

mmtrue = struct('A',A,'B',B,'C',C,'D',D,'Q',Q,'Q0',Q0); % parameter struct for model
[yy,zz,yyprob] = sampleLDSBernoulli(mmtrue,nT,uu); % sample data from model

%% Compute MAP latents given true parameters

[zzmaptrue,H,logevtrue] = computeZmap_LDSBernoulli(yy,mmtrue,uu); % inference
fprintf('log-evidence at true params: %.2f\n\n', logevtrue);

%% Make plots showing true and inferred latents

ii = 1:min(nT,500); % indices to plot
clf; subplot(421);  % Plot 1st latent
plot(ii,zz(1,ii)', ii,zzmaptrue(1,ii)'); % plot true and MAP latents
title('latent 1'); xlabel('time bin'); box off;
legend('z true','z | \theta_{true}', 'location', 'northwest');

subplot(423); % plot 2nd latent (if present)
if nz>1, plot(ii,zz(2,ii)', ii,zzmaptrue(2,ii)');
    title('latent 2'); xlabel('time bin'); box off;
else, delete(gca)
end

subplot(425); % Plot observation probabilities
plot(ii,yyprob(:,ii)');
xlabel('time (bin)');ylabel('P(spike)');title('true P(spike)'); drawnow;


%% Compute max-evidence estimate of model parameters using Laplace-EM

% Set options for EM     
optsEM.maxiter = 20;    % maximum # of iterations
optsEM.display = 10;  % display frequency
% optsEM.dlogptol = 0.01;  % stopping tolerance (NOT USED)
optsEM.nMCsamps = 10;  % number of monte carlo samples for evaluating total-data log-likelihood in M step

% Specify which parameters to learn.  (Set to 0 or 'false' to NOT update).
optsEM.update.A = 1;
optsEM.update.Q = 1;
optsEM.update.C = 1; % Note: if 1, update.D must be 1 too
optsEM.update.B = 1; % Note: if 1, uu must not be all-0.
optsEM.update.D = 1; % Note: if 1, update.C must be 1 too, and uu must not be all-0.

% Initialize fitting struct
mm0 = struct('A',A,'B',B,'C',C,'D',D,'Q',Q,'Q0',Q0);  % make struct with initial params
if optsEM.update.A, mm0.A = A*.9+randn(nz)*.1; end % initial A param
if optsEM.update.C, mm0.C = C*.9+randn(ny,nz)*.1; end % initial C param
if optsEM.update.Q, mm0.Q = Q*2; end % initial Q param
if optsEM.update.B, mm0.B = B*.5; end % initial B param
if optsEM.update.D, mm0.D = D*.5; end % initial D param

% Run Laplace-EM inference for model parameters
[mm1,mmall,logEvTrace1,logev1,zzm1,H1,Trun] = runLEM_LDSBernoulli(yy,mm0,uu,optsEM);
%[mm1,logEvTrace1,logev1,zzm1,H1] = runLEM_LDSBernoulli(yy,mm1,uu,optsEM);  % Uncomment to run more EM iterations

% Report whether optimization succeeded in finding a posible global optimum
fprintf('\nLog-evidence at true params:      %.2f\n', logevtrue);
fprintf('Log-evidence at inferred params:  %.2f\n', logev1);
% Report if we found the global optimum
if logev1>=logevtrue, fprintf('(found optimum -- SUCCESS!)\n');
else,   fprintf('(FAILED to find optimum!)\n');
end

%% Find best alignment between inferred and MAP-true latents

Wa = ((zzm1*zzm1')\(zzm1*zzmaptrue'))'; % alignment weights

% Transform inferred params to align with true params
mm1a = []; % initialize
mm1a.A = Wa*mm1.A/Wa;  % transform dynamics
mm1a.C = mm1.C/Wa;     % transform observations
mm1a.B = Wa*mm1.B;     % transform observations
mm1a.D = mm1.D;     % transform observations
mm1a.Q = Wa*mm1.Q*Wa'; % transform cov
mm1a.Q0 = Q0;          % initial covariance
[zzm1a,H1a,logev1a] =computeZmap_LDSBernoulli(yy,mm1a,uu); % recompute
% Note logev1a and logev1 should be (nearly) identical

%% Make plots

subplot(422);  % Plot evidence over EM iterations
nEM = length(logEvTrace1);
plot([1 nEM], logevtrue*[1 1], 'k--', 1:nEM,logEvTrace1); box off;
xlabel('EM iteration'); ylabel('Laplace log-evidence');
title('EM performance');

subplot(421);  % Plot 1st latent
ii = 1:min(nT,500); % indices to plot
plot(ii,zz(1,ii)', ii,zzmaptrue(1,ii)',ii,zzm1a(1,ii)', '--');
title('latent 1'); xlabel('time bin'); box off;
legend('z true','z | \theta_{true}', 'z | \theta_{hat}', 'location', 'northwest');
subplot(423); % plot 2nd latent (if present
if nz>1,  plot(ii,zz(2,ii)', ii,zzmaptrue(2,ii)', ii,zzm1a(2,ii)', '--');
    title('latent 2'); xlabel('time bin'); box off;
else, cla;
end

% Plot observation probabilities
subplot(427); 
rr1a = 1./(1+exp(-(mm1a.C*zzm1a + mm1a.D*uu))); % output probabilities
plot(ii,rr1a(:,ii)');
xlabel('time (bin)');ylabel('P(spike)');
title('inferred P(spike)'); drawnow;

% Plot recovered params
subplot(447);
if optsEM.update.A
    mx = max(abs(A(:)))*1.2;
    plot(mx*[-1 1], mx*[-1 1], 'k--', A(:), mm1a.A(:), 'o');
    xlabel('true A'); ylabel('inferred A');
    title('true vs. recovered A'); axis square; box off;
else, delete(gca);
end
subplot(448);
if optsEM.update.B
    mx = max(abs(B(:)))*1.2;
    plot(mx*[-1 1], mx*[-1 1], 'k--', B(:), mm1a.B(:), 'o');
    xlabel('true B'); ylabel('inferred B');
    title('true vs. recovered B'); axis square; box off;
else, delete(gca);
end
subplot(4,4,11);
if optsEM.update.C 
    mx = max(abs(C(:)))*1.2;
    plot(mx*[-1 1], mx*[-1 1], 'k--', C(:), mm1a.C(:), 'o');
    xlabel('true C'); ylabel('inferred C');
    title('true vs. recovered C'); axis square; box off;
else, delete(gca);
end
subplot(4,4,12);
if optsEM.update.D
    mx = max(abs(D(:)))*1.2;
    plot(mx*[-1 1], mx*[-1 1], 'k--', D(:), mm1a.D(:), 'o');
    xlabel('true D'); ylabel('inferred D');
    title('true vs. recovered D'); axis square; box off;
else, delete(gca);
end
subplot(4,4,16);
if optsEM.update.Q
    mx = max(abs(Q(:)))*1.2;
    plot(mx*[-1 1], mx*[-1 1], 'k--', Q(:), mm1a.Q(:), 'o');
    xlabel('true Q'); ylabel('inferred Q');
    title('true vs. recovered Q'); axis square; box off;
else, delete(gca);
end
% demo_LDSBernoulli_MaxLaplEv_vs_LEM.m
%
% Compare inference methods for latent LDS-Bernoulli model:
% 1. Laplace-EM  (runLEM)
% 2. Direct optimization of Laplace evidence (runMaxLapEvidence_LDSBernoulli)

addpath inference_Bernoulli/
addpath utils

% Set dimensions
nz = 2;  % dimensionality of latent z
ny = 10; % dimensionality of observation y
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
    s = s/max(abs(s))*.98; % set largest eigenvalue to lie inside unit circle (enforcing stability)
    s(real(s)<0) = -s(real(s)<0); % set real parts to be positive (encouraging smoothness)
    A = real(u*(diag(s)/u));  % reconstruct A from its eigs and eigenvects
end

% Dynamics noise covariance
Q = randn(nz); Q = .01*(Q'*Q+eye(nz)); % dynamics noise covariance
Q0 = eye(nz); % initial covariance

% Set observation matrix C
C = 0.8*randn(ny,nz); % loading weights

% Use discrete Lyapunov equation solver to compute asymptotic covariance
Pinf = dlyap(A,Q);

% Simulate data from the LDS-Bernoulli model
mmtrue = struct('A',A,'C',C,'Q',Q,'Q0',Q0);  % make param struct
[yy,zz,yyprob] = sampleLDSBernoulli(mmtrue,nT); % sample from model

% Compute MAP estimate of latents using true params
[zzmaptrue,H,logevtrue] = computeZmap_LDSBernoulli(yy,mmtrue);
fprintf('log-evidence at true params: %.2f\n\n', logevtrue);

% -------  Make plots showing true and inferred latents ----------------
ii = 1:min(nT,500); % time indices to plot
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
xlabel('time bin');ylabel('P(spike)');title('true P(spike)'); drawnow;

%% Method 1:  compute max-evidence estimate of A and C using Laplace-EM

% Set options for EM     
optsEM.maxiter = 100;    % maximum # of iterations
optsEM.display = 10;  % display frequency
optsEM.nMCsamps = 10;  % number of monte carlo samples for evaluating total-data log-likelihood in M step

% Specify which parameters to learn.  (Set to 'false' to NOT update).
optsEM.update.A = 1;
optsEM.update.C = 1;
optsEM.update.Q = 0;  % DON'T update Q for this comparison

% Initialize fitting struct
mm0 = struct('A',A,'C',C,'Q',Q,'Q0',Q0);  % make struct with initial params
if optsEM.update.A, mm0.A = A*.9+randn(nz)*.1; end % initial A param
if optsEM.update.C, mm0.C = C*.9+randn(ny,nz)*.1; end % initial C param
if optsEM.update.Q, mm0.Q = Q*2; end % initial Q param

% Run Laplace-EM inference for model parameters
[mm1,logEvTrace1,logev1,zzm1,H1] = runLEM_LDSBernoulli(yy,mm0,[],optsEM);


%% Method 2: Directly maximize Laplace-evidence for A and C

% 1. Initialize from true params
opts_MaxLapl = optimset('display', 'iter','maxiter',100);
[mm2,logev2] = runMaxLapEvidence_LDSBernoulli(yy,mm0,opts_MaxLapl); % Run optimization starting from true params

%% Report results:
fprintf('------------------------------------------\n');
fprintf('Relative log-ev at true params:             %.2f\n', 0);
fprintf('Relative Log-ev at LEM-inferred params:     %.2f\n', logev1-logevtrue);
fprintf('Relative Log-ev at MaxLap-inferred params:  %.2f\n', logev2-logevtrue);

% Plot recovered params
subplot(424);
mx = max(abs(A(:)))*1.2;
plot(mx*[-1 1], mx*[-1 1], 'k--', A(:), mm1.A(:), 'o',A(:), mm2.A(:), 'o');
xlabel('true A'); ylabel('inferred A');
title('true vs. recovered A'); axis square; box off;

subplot(426);
mx = max(abs(C(:)))*1.2;
plot(mx*[-1 1], mx*[-1 1], 'k--', C(:), mm1.C(:), 'o',C(:),mm2.C(:),'x');
xlabel('true C'); ylabel('inferred C');
title('true vs. recovered C'); axis square; box off;


% demo_LDSBernoulli_LEM.m
%
% 1. Sample from a latent Gaussian linear dynamical system (LDS) model with Bernoulli observations.
% 2. Compute ML fit via Laplace-EM

% Basic equations:
% -----------------
% X_t = A*X_{t-1} + eps_x,  eps_x ~ N(0,Q)   % latents
% Y_t ~ Bernoulli(f(C*X_t)  % observations

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
    A = real(u*(diag(s)/u));  % reconstruct A from its eigs and eigenvectors
end

% Dynamics noise covariance
Q = randn(nz); Q = .01*(Q'*Q+eye(nz)); % dynamics noise covariance
Q0 = eye(nz); % initial covariance

% Set observation matrix C
C = 0.25*randn(ny,nz); % loading weights
% --------------------
% NOTE (critical observation): If norm of C is too large (eg sigma>1), then
% ELBO tends to decrease under LEM when inferring C with A held fixed.
% --------------------

% Use discrete Lyapunov equation solver to compute asymptotic covariance
Pinf = dlyap(A,Q);

%% Simulate data from the LDS-Bernoulli model

mmtrue = struct('A',A,'C',C,'Q',Q,'Q0',Q0); % parameter struct for model
[yy,zz,yyprob] = sampleLDSBernoulli(mmtrue,nT); % sample data from model

%% Compute MAP estimate of latents using true params

[zzmaptrue,H,logevtrue] = computeZmap_LDSBernoulli(yy,mmtrue);
fprintf('log-evidence at true params: %.2f\n\n', logevtrue);

%% Make plots showing true and inferred latents

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

%% Compute max-evidence estimate of A and C using Laplace-EM

% Set options for EM     
optsEM.maxiter = 100;    % maximum # of iterations
% optsEM.dlogptol = 0.01;  % stopping tolerance (NOT USED)
optsEM.display = 10;  % display frequency
optsEM.nMCsamps = 10;  % number of monte carlo samples for evaluating total-data log-likelihood in M step

% Specify which parameters to learn.  (Set to 'false' to NOT update).
optsEM.update.A = 1;
optsEM.update.C = 1;
optsEM.update.Q = 1;

% Initialize fitting struct
mm0 = struct('A',A,'C',C,'Q',Q,'Q0',Q0);  % make struct with initial params
if optsEM.update.A, mm0.A = A*.9+randn(nz)*.1; end % initial A param
if optsEM.update.C, mm0.C = C*.9+randn(ny,nz)*.1; end % initial C param
if optsEM.update.Q, mm0.Q = Q*2; end % initial Q param

% Run Laplace-EM inference for model parameters
[mm1,logEvTrace1,logev1,zzm1,H1] = runLEM_LDSBernoulli(yy,mm0,[],optsEM);
%[mm1,logEvTrace1,logev1,zzm1,H1] = runLEM_LDSBernoulli(yy,mm1,optsEM);  % Uncomment to run more EM iterations

% Report whether optimization succeeded in finding a posible global optimum
fprintf('\nLog-evidence at true params:      %.2f\n', logevtrue);
fprintf('Log-evidence at inferred params:  %.2f\n', logev1);
% Report if we found the global optimum
if logev1>=logevtrue, fprintf('(found better optimum -- SUCCESS!)\n');
else,   fprintf('(FAILED to find optimum!)\n');
end

%% Find best alignment between inferred and MAP-true latents

Wa = ((zzm1*zzm1')\(zzm1*zzmaptrue'))'; % alignment weights via regression of new latents onto old

% Transform inferred params to align with true params
mm1a = struct('A', Wa*mm1.A/Wa,'C',mm1.C/Wa,'Q',Wa*mm1.Q*Wa','Q0',Q0); % transformed params
[zzm1a,H1a,logev1a] =computeZmap_LDSBernoulli(yy,mm1a); % recompute log-ev
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
rr1a = 1./(1+exp(-mm1a.C*zzm1a));  % output probabilities
plot(ii,rr1a(:,ii)');
xlabel('time bin');ylabel('P(spike)');
title('inferred P(spike)'); drawnow;

% Plot recovered params
subplot(424);
if optsEM.update.A
    mx = max(abs(A(:)))*1.2;
    plot(mx*[-1 1], mx*[-1 1], 'k--', A(:), mm1a.A(:), 'o');
    xlabel('true A'); ylabel('inferred A');
    title('true vs. recovered A'); axis square; box off;
else, delete(gca);
end
subplot(426);
if optsEM.update.C 
    mx = max(abs(C(:)))*1.2;
    plot(mx*[-1 1], mx*[-1 1], 'k--', C(:), mm1a.C(:), 'o');
    xlabel('true C'); ylabel('inferred C');
    title('true vs. recovered C'); axis square; box off;
else, delete(gca);
end
subplot(428);
if optsEM.update.Q
    mx = max(abs(Q(:)))*1.2;
    plot(mx*[-1 1], mx*[-1 1], 'k--', Q(:), mm1a.Q(:), 'o');
    xlabel('true Q'); ylabel('inferred Q');
    title('true vs. recovered Q'); axis square; box off;
else, delete(gca);
end
% demo_LDSgaussian_EMfitting.m
%
% Sample from a latent Gaussian linear dynamical system (LDS) model, then
% run EM to estimate the model parameters

% Basic equations:
% -----------------
% X_t = A*X_{t-1} + eps_x,  eps_x ~ N(0,Q)  % latents
% Y_t = C*X_t + eps_y,      eps_y ~ N(0,R)  % observations
%
% With X_1 ~ N(0,Q0)    initial condition:  

addpath inference_Gaussian/
addpath utils

% Set dimensions
nz = 2;  % dimensionality of latent z
ny = 10; % dimensionality of observation y
nT = 500; % number of time steps

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
    A = real(u*(diag(s)/u));  % reconstruct A from eigs and eigenvectors
end
% Set observation matrix C
C = 0.5*randn(ny,nz); % loading weights

% Dynamics noise covariance
Q = randn(nz); Q = .1*(Q'*Q+eye(nz)); % dynamics noise covariance
Q0 = 1.5*eye(nz); % initial covariance
R = .25*diag(1*rand(ny,1)+.1); %  Y noise covariance

% Use discrete Lyapunov equation solver to compute asymptotic covariance
Pinf = dlyap(A,Q);

%% Sample data from LDS model

mmtrue = struct('A',A,'C',C,'Q',Q,'R',R,'Q0',Q0);  % make param struct
[yy,zz] = sampleLDSgauss(mmtrue,nT); % sample from model

%% Compute latents and log-marginal likelihood given true params

% Run Kalman Filter-Smoother to get posterior over latents given true data
[zzmutrue,loglitrue,zzcovtrue] = runKalmanSmooth(yy,[],mmtrue);

%% Compute ML estimate for model params using EM

% Set options for EM     
optsEM.maxiter = 500;    % maximum # of iterations
optsEM.dlogptol = 1e-4;  % stopping tolerance
optsEM.display = 10;  % display frequency

% Specify which parameters to learn.  (Set to '0' or 'false' to NOT update).
optsEM.update.A = 1;
optsEM.update.C = 1;
optsEM.update.Q = 1;
optsEM.update.R = 1;

% Initialize fitting struct
mm0 = struct('A',A,'C',C,'Q',Q,'R',R,'Q0',Q0);  % make struct with initial params
if optsEM.update.A, mm0.A = A*.9+randn(nz)*.1; end % initial A param
if optsEM.update.C, mm0.C = C*.9+randn(ny,nz)*.1; end % initial C param
if optsEM.update.Q, mm0.Q = Q*1.33; end % initial Q param
if optsEM.update.R, mm0.R = R*1.5; end % initial R param

%%

% Run EM inference for model parameters
[mm1,logEvTrace] = runEM_LDSgaussian(yy,mm0,[],optsEM);

% Compute MAP latents and log-evidence at optimum
[zzm1,logli1,zzcov1] =runKalmanSmooth(yy,[],mm1);

% Report whether optimization succeeded in finding a posible global optimum
fprintf('\nLog-evidence at true params:      %.2f\n', loglitrue);
fprintf('Log-evidence at inferred params:  %.2f\n', logli1);
% Report if we found the global optimum
if logli1>=loglitrue, fprintf('(found better optimum -- SUCCESS!)\n');
else,   fprintf('(FAILED to find optimum!)\n');
end


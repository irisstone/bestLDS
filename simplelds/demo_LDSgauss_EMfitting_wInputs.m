% test_LDSgaussian_EMfitting.m
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
nu = 3;  % dimensionality of external inputs
nT = 1000; % number of time steps

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

% Set observation matrix C
C = 0.5*randn(ny,nz); % loading weights

% Set input matrices B and D
B = 0.5*randn(nz,nu);  % weights from inputs to latents
D = 0.5*randn(ny,nu);  % weights from inputs to observed

% Dynamics noise covariance
Q = randn(nz); Q = .1*(Q'*Q+eye(nz)); % dynamics noise covariance
R = diag(1*rand(ny,1)+.1); %  Y noise covariance
Q0 = 2*eye(nz); % Covariance for latent in first time step

% Use discrete Lyapunov equation solver to compute asymptotic covariance
P = dlyap(A,Q);

%% Sample data from LDS model

uu = randn(nu,nT); % external inputs

mmtrue = struct('A',A,'B',B,'C',C,'D',D,'Q',Q,'R',R,'Q0',Q0);  % make param struct
[yy,zz] = sampleLDSgauss(mmtrue,nT,uu); % sample from model

%% Compute latents and log-marginal likelihood given true params

% Run Kalman Filter-Smoother to get posterior over latents given true data
[zzmutrue,loglitrue,zzcovtrue] = runKalmanSmooth(yy,uu,mmtrue);

%% Compute ML estimate for model params using EM

% Set options for EM     
optsEM.maxiter = 250;    % maximum # of iterations
optsEM.dlogptol = 1e-4;  % stopping tolerance
optsEM.display = 10;  % display frequency

% Specify which parameters to learn.  (Set to '0' or 'false' to NOT update).
optsEM.update.A = 1;
optsEM.update.B = 1;
optsEM.update.C = 1;
optsEM.update.D = 1;
optsEM.update.Q = 1;
optsEM.update.R = 1;

% Initialize fitting struct
mm0 = struct('A',A,'C',C,'Q',Q,'R',R,'Q0',Q0,'B',B,'D',D);  % make struct with initial params
if optsEM.update.A, mm0.A = A*.5+randn(nz)*.1; end % initial A param
if optsEM.update.C, mm0.C = C*.9+randn(ny,nz)*.1; end % initial C param
if optsEM.update.Q, mm0.Q = Q*1.33; end % initial Q param
if optsEM.update.R, mm0.R = R*1.5; end % initial R param
if optsEM.update.B, mm0.B = B*.5; end % initial B param
if optsEM.update.D, mm0.D = D*.5; end % initial D param

%% Run EM and examine inferred params & final logli 

% Run EM inference for model parameters
[mm1,logEvTrace] = runEM_LDSgaussian(yy,mm0,uu,optsEM);

% Compute MAP latents and log-evidence at optimum
[zzm1,logli1,zzcov1] =runKalmanSmooth(yy,uu,mm1);

% Report whether optimization succeeded in finding a posible global optimum
fprintf('\nLog-evidence at true params:      %.2f\n', loglitrue);
fprintf('Log-evidence at inferred params:  %.2f\n', logli1);
% Report if we found the global optimum
if logli1>=loglitrue, fprintf('(found better optimum -- SUCCESS!)\n');
else,   fprintf('(FAILED to find optimum!)\n');
end


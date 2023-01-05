% test_LDSgaussian_EMfitting.m
%
% Unit tests for EM inference for LDS-Gaussian model.
% Checks that log-li doesn't decrease for EM updates from true params

addpath ../inference_Gaussian/
addpath ../utils
addpath ..

% Set dimensions
nz = 2;  % dimensionality of latent z
ny = 10; % dimensionality of observation y
nu = 3;  % dimensionality of external inputs

nT = 25; % number of time steps

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
R = 0.5*diag(1*rand(ny,1)+.1); %  Y noise covariance

% Set input matrices B and D
B = 0.5*randn(nz,nu);  % weights from inputs to latents
D = 0.5*randn(ny,nu);  % weights from inputs to observed

%% Sample data from LDS model

uu = randn(nu,nT); % external inputs

mmtrue = struct('A',A,'B',B,'C',C,'D',D,'Q',Q,'R',R,'Q0',Q0);  % make param struct
[yy,zz] = sampleLDSgauss(mmtrue,nT,uu); % sample from model

%% Compute latents and log-marginal likelihood given true params

% Run Kalman Filter-Smoother to get posterior over latents given true data
[~,loglitrue] = runKalmanSmooth(yy,uu,mmtrue);

%% Compute ML estimate for model params using EM

% Set options for EM     
optsEM.maxiter = 10;    % maximum # of iterations
optsEM.dlogptol = 1e-3;  % stopping tolerance
optsEM.display = inf;  % display frequency

% Set set of parameters to params to learn
updateSettings = [eye(6),[1;1;0;0;0;0],[1;1;0;0;1;0],...
    [0;0;1;1;0;0],[0;0;1;1;0;1], ones(6,1)];
nSettings = size(updateSettings,2);

loglis_new = zeros(1,nSettings);

for jj = 1:nSettings
    % Specify which parameters to learn. (Set to '0' or 'false' to NOT update).
    optsEM.update.A = updateSettings(1,jj);
    optsEM.update.B = updateSettings(2,jj);
    optsEM.update.C = updateSettings(3,jj);
    optsEM.update.D = updateSettings(4,jj);
    optsEM.update.Q = updateSettings(5,jj);
    optsEM.update.R = updateSettings(6,jj);

    % Run EM inference for model parameters
    mm1 = runEM_LDSgaussian(yy,mmtrue,uu,optsEM);

    % Compute MAP latents and log-evidence at optimum
    [~,logli1] =runKalmanSmooth(yy,uu,mm1);
    
    loglis_new(jj) = logli1;
end

logli_improvements = loglis_new - loglitrue;

if any(logli_improvements < 0)
    warning('test_LDSgauss_EMfitting_wInputs.m test FAILED: log-li didn''t improve for some params');
else
    fprintf('test_LDSgauss_EMfitting_wInputs.m test PASSED: log-li improved for all params\n'); 
end

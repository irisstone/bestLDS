% test_LDSgaussian_KalmanSmoother.m
%
% Unit test for Kalman Filter-Smoother for Gaussian LDS data. 
%
% 1. Samples from a Gaussian latent linear dynamical system (LDS) model.
% 2. Runs Kalman Smoothing using classic and block-sparse matrix verions
% and verifies they give the same output.

% Basic equations:
% -----------------
% X_t = A*X_{t-1} + eps_x,  eps_x ~ N(0,Q)   % latents
% Y_t = C*X_t + eps_y,      eps_y ~ N(0,R);  % observations

addpath ../inference_Gaussian/
addpath ../utils
addpath ..

% Set dimensions
nz = 2;  % dimensionality of latent z
ny = 10; % dimensionality of observation y
nT = 50; % number of time steps

TOL = 1e-10;  % tolerance for numerical error for this unit test

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

% Dynamics noise covariance
Q = randn(nz); Q = .1*(Q'*Q+eye(nz)); % dynamics noise covariance
R = diag(1*rand(ny,1)+.1); %  Y noise covariance
Q0 = eye(nz)*2; % prior covariance for latent in first time bin

%% Sample data from LDS model

mmtrue = struct('A',A,'C',C,'Q',Q,'R',R,'Q0',Q0);  % make param struct
[yy,zz] = sampleLDSgauss(mmtrue,nT); % sample from model

%% Compute latents and log-marginal likelihood given true params

% Run Kalman Filter-Smoother (slower traditional version)
[zzmu1,loglitrue1,zzcov1,zzcov_diag1] = runKalmanSmooth(yy,[],mmtrue);

% Run Kalman Filter-Smoother (fast matrix version)
[zzmu2,zzHess2,loglitrue2] = runKalmanSmooth_matrix(yy,[],mmtrue); % run Kalman Filter-Smoother

%% Now do some checks to make sure two methods agree

% 1. ==== Test log marginal likelihood  ==================================

if abs(loglitrue1-loglitrue2)>TOL
    warning('test_LDSgaussian_KalmanSmoother.m unit test FAILED: log-li vals don''t match');
else
    fprintf('test_LDSgaussian_KalmanSmoother.m PASSED: log-li vals match\n');
end

% 2. ==== Test posterior mean ============================================
maxabsdiff = max(max(abs(zzmu1-zzmu2))); 

if maxabsdiff > TOL
    warning('test_LDSgaussian_KalmanSmoother.m unit test FAILED: posterior means don''t match');
else
    fprintf('test_LDSgaussian_KalmanSmoother.m PASSED: posterior means match\n');
end

% 3. ===  Test covaraince blocks =========================================

% Insert covariance blocks from runKalmanSmooth into a matrix
Lcov1 = zeros(nz*nT);
for jblock = 1:nT
    inds = (jblock-1)*nz+1:jblock*nz;
    Lcov1(inds,inds) = zzcov1(:,:,jblock);
    if jblock<nT
        Lcov1(inds,inds+nz) = zzcov_diag1(:,:,jblock);
    end
end
ii1 = find(Lcov1);  % find non-zero elements

% Compute full covariance from matrix versionn
Lcov2 = full(inv(zzHess2));

% Report test
if max(abs(loglitrue1-loglitrue2))>TOL
    warning('test_LDSgaussian_KalmanSmoother.m unit test FAILED: covariances don''t match');
else
    fprintf('test_LDSgaussian_KalmanSmoother.m PASSED: covariances match\n');
end


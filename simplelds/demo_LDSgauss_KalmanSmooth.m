% demo_LDSgaussian_KalmanSmoother.m
%
% Sample from a latent Gaussian linear dynamical system (LDS) model, then
% compute posterior mean using 2 different implementations of Kalman filter-smooother

% Basic equations:
% -----------------
% X_t = A*X_{t-1} + eps_x,  eps_x ~ N(0,Q)   % latents
% Y_t = C*X_t + eps_y,      eps_y ~ N(0,R);  % observations

addpath inference_Gaussian/
addpath utils

% Set dimensions
nz = 4;  % dimensionality of latent z
ny = 20; % dimensionality of observation y
nT = 100; % number of time steps

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
R = diag(1*rand(ny,1)+.1); %  Y noise covariance
Q0 = 4*eye(nz);

% Use discrete Lyapunov equation solver to compute asymptotic covariance
Pinf = dlyap(A,Q);

% % Equiv solution using using Ricatti eqn solver:
% Pinf = dare(A',zeros(nz),Q,eye(nz));


%% Sample data from LDS model

mmtrue = struct('A',A,'C',C,'Q',Q,'R',R,'Q0',Q0);  % make param struct
[yy,zz] = sampleLDSgauss(mmtrue,nT); % sample from model


%% Compute latents and log-marginal likelihood given true params

% Method 1: Run Kalman Filter-Smoother (slower traditional version)
tic;
[zzmu1,loglitrue1,zzcov1,zzcov_diag1] = runKalmanSmooth(yy,[],mmtrue);
toc;

% Method 2: Run Kalman Filter-Smoother (fast matrix version)
tic;
[zzmu2,zzHess2,loglitrue2] = runKalmanSmooth_matrix(yy,[],mmtrue); % run Kalman Filter-Smoother
toc;

%% Now do some checks to make sure two methods agree (only for small problems)

%fprintf('log-evidence at true params: %.2f\n\n', loglitrue);
fprintf('Diff in log-li values (Method1-Method2): %g\n', loglitrue1-loglitrue2);

if ny*nT < 5000
    Lcov = full(inv(zzHess2));
    
    % Pick 2 blocks to examine
    blk = 1;  % time step to examine
    ii = nz*(blk-1)+1:nz*(blk+1); % indices for cov of this block
    
    % Compute blocks
    CovBlock_Method1 = Lcov(ii,ii);
    CovBlock_Method2 = [zzcov1(:,:,blk), zzcov_diag1(:,:,blk);
        zzcov_diag1(:,:,blk)', zzcov1(:,:,blk+1)];
    Errs = CovBlock_Method2-CovBlock_Method1;
    
    subplot(211);
    plot(1:nT, zzmu1', 1:nT, zzmu2', '--');
    title('Kalman smoother ouput comparison');
    legend('dim1, method 1', 'dim2, method 1', 'dim1, method 2', 'dim2, method 2');
    xlabel('time');
    
    subplot(234);
    imagesc(CovBlock_Method1);title('cov block, method 1');
    
    subplot(235);
    imagesc(CovBlock_Method2);title('cov block, method 2');
    
    subplot(236);
    plot(Errs); title('errors (cov2-cov1)');
else
    fprintf('Skipping direct comparison of large covariance matrices\n');
end

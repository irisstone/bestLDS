function [zzmu,logli,zzcov,zzcov_offdiag1] = runKalmanSmooth(yy,uu,mm)
% [zzmu,logli,zzcov,zzcov_abovediag] = runKalmanSmooth(yy,uu,mm)
%
% Run Kalman Filter-Smoother for latent LDS Gaussian model
%
% INPUTS:
% -------
%  yy [ny x nT] - matrix of observations
%  ss [ns x nT] - inputs
%  mm  [struct] - model struct with fields
%           .A [nz x nz] - dynamics matrix
%           .B [nz x ns] - input matrix (optional)
%           .C [ny x nz] - latents-to-observations matrix
%           .D [ny x ns] - input-to-observations matrix (optional)
%           .Q [nz x nz] - latent noise covariance
%
% OUTPUTS:
% --------
%    zzmu [nz x nT]      - posterior mean latents zz | yy
%   logli [1 x 1]        - log-likelihood P( yy | ss, theta )
%   zzcov [nz x nz x nT] - diagonal blocks of cov
% zzcov_off [nz x nz x nT] - diagonal blocks of cov
%
%
% Basic equations:
% -----------------
% X_t = A*X_{t-1} + w_t,    w_t ~ N(0,Q)   % latent dynamics
% Y_t = C*X_t     + v_t,    v_t ~ N(0,R);  % observations


% Extract sizes
nz = size(mm.A,1);  % number of obs and latent dimensions
nT = size(yy,2); % number of time bins

% pre-compute C'*inv(R) and C'*inv(R)*C;
CtRinv = mm.C'/mm.R;
CtRinvC = CtRinv*mm.C;

% check if input-latent matrix is provided
if isfield(mm,'B') && ~isempty(mm.B) 
    zin = [zeros(nz,1), mm.B*uu(:,2:end)];  % additive intput to latents (as column vectors)
else
    zin = zeros(nz,nT);
end

% check if intput-obs matrix is provided
if isfield(mm,'D') && ~isempty(mm.D) 
    yyctr = yy-(mm.D*uu);  % subtract additive intput to observations
else
    yyctr = yy;
end

% Allocate storage
zzmu = zeros(nz,nT);   % posterior mean E[ zz(t) | Y]
zzcov = zeros(nz,nz,nT); % marginal cov of zz(t)
munext = zeros(nz,nT);   % prior mean for next step:  E[ z(t) | y(1:t-1)]
Pnext = zeros(nz,nz,nT); % prior cov for next step: cov[ z(t) | y(1:t-1)]
logcy = zeros(1,nT); % store conditionals P(y(t) | y(1:t))

% ============================================
% Kalman Filter
% ============================================

% process 1st time bin
zzcov(:,:,1) = inv(inv(mm.Q0)+CtRinvC);  
zzmu(:,1) = zzcov(:,:,1)*(CtRinv*yyctr(:,1));  % NOTE: no inputs in first time bin
logcy(1) = logmvnpdf(yyctr(:,1)',(mm.C*zin(:,1))',mm.C*mm.Q0*mm.C' + mm.R);

for tt = 2:nT
    Pnext(:,:,tt) = mm.A*zzcov(:,:,tt-1)*mm.A' + mm.Q; % prior cov for time bin t
    munext(:,tt) = mm.A*zzmu(:,tt-1)+zin(:,tt); % mean for next time bin t
    
    zzcov(:,:,tt) = inv(inv(Pnext(:,:,tt))+CtRinvC);   % KF cov for time bin t
    zzmu(:,tt) = zzcov(:,:,tt)*(CtRinv*yyctr(:,tt) + (Pnext(:,:,tt)\munext(:,tt))); % KF mean
    
    % compute log P(y_t | y_{1:t-1})
    logcy(tt) = logmvnpdf(yyctr(:,tt)',(mm.C*munext(:,tt))',mm.C*Pnext(:,:,tt)*mm.C'+mm.R);
end

% compute marginal log-likelihood P(y | theta)
logli = sum(logcy);

if nargout > 2
    
    if nargout > 3
        zzcov_offdiag1 = zeros(nz,nz,nT-1); % above-diagonal covariance block
    end

    % ============================================
    % Kalman Smoother (if covariance needed)
    % ============================================

    % Pass backwards, updating mean and covariance with info from the future
    for tt = (nT-1):-1:1
        Jt = (zzcov(:,:,tt)*mm.A')/Pnext(:,:,tt+1); % matrix we need
        zzcov(:,:,tt) = zzcov(:,:,tt) + Jt*(zzcov(:,:,tt+1)-Pnext(:,:,tt+1))*Jt'; % update cov
        zzmu(:,tt) = zzmu(:,tt) + Jt*(zzmu(:,tt+1)-munext(:,tt+1)); % update mean

        if nargout > 3
            zzcov_offdiag1(:,:,tt) = Jt*zzcov(:,:,tt+1);
        end
    end
    
end
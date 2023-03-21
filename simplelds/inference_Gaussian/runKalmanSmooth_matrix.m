function [zzmu,zzHess,logli] = runKalmanSmooth_matrix(yy,uu,mm)
% [zzmu,zzHess,logli] = runKalmanSmooth_matrix(yy,uu,mm)
%
% Run Kalman Filter-Smoother for latent LDS model using efficient block
% matrix formulation
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
%           .Q0 [nz x nz] - latent noise covariance for 1st time step
%           .R [ny x ny] - observed noise covariance
%
% OUTPUTS:
% --------
%    zzmu [nz x nT]       - posterior mean latents zz | yy
%  zzHess [nz*nT x nz*nT] - sparse inverse Hessian
%  logli - log-likelihood log P( yy | ss, theta)
%
%
% Basic equations:
% -----------------
% X_t = A*X_{t-1} + w_t,    w_t ~ N(0,Q)   % latent dynamics
% Y_t = C*X_t     + v_t,    v_t ~ N(0,R);  % observations
%
% Matrix version:
% ---------------
% Model can be rewritten as a pair of matrix equations with vectorized x
% and y as follows:
% 
%    Hm*Xvec + B*Uvec = Wt
%    Yvec = Cm*Xvec + D*uu + Vt
%
% where Hm is a block matrix of 1st order differences from the dynamics
% equation:
% 
% Hm = [I      
%        -A  I 
%           -A I 
%               ...
%                 -A I],
%
% Cm is a block-diagonal matrix with C on the diagonals:
%
% Cm = [C 
%         C
%          ...
%             C],
%
% and Wt and Vt are noise vectors whose covariances are both block-diagonal
% with Q and R along the diagonals, respectively.
%
%  The posterior is thus:
%     X | Y ~ N( LC'R^{-1} yy, L),  
%  where posterior covariance L is 
%     L = (C'R^{-1}C + (D^{-1})*Q*(D')^{-1})^{-1}

% Extract sizes
[ny,nz] = size(mm.C);  % number of obs and latent dimensions
nT = size(yy,2); % number of time bins

% % Build relevant block matrices
Hmat = kron(spdiags(ones(nT-1,1),-1,nT,nT),-mm.A) + speye(nz*nT);  % latent dynamics
Cmat = kron(speye(nT),mm.C); % C projection from latents to observed
Qinvmat = kron(speye(nT),inv(mm.Q)); % Q noise in latents
Qinvmat(1:nz,1:nz) = inv(mm.Q0); % covariance over latents on 1st time step
Rmat = kron(speye(nT),mm.R); % R noise in observed 

% Process input-latents if provided
if isfield(mm,'B') && ~isempty(mm.B) 
    zin = [zeros(nz,1), mm.B*uu(:,2:end)];  % additive intput to latents (with none in 1st bin)
    HinvZin = Hmat\zin(:);  % prior mean of latents 
else
    HinvZin = zeros(nz*nT,1);  % prior mean of latents 
end

% check if intput-obs matrix is provided
if isfield(mm,'D') && ~isempty(mm.D) 
    yctr = yy(:)-reshape(mm.D*uu,ny*nT,1);  % additive intput to observations (as column vectors)
else
    yctr = yy(:);
end

% % Compute posterior inverse covariance
QtildeInv = Hmat'*(Qinvmat*Hmat);  % H*inv(Q)*H'
zzHess = (Cmat'*(Rmat\Cmat) + QtildeInv); % inverse of posterior covariance 

% % Compute posterior mean (Kalman smoother)
zzmu = reshape(zzHess\((Cmat'*(Rmat\yctr)) + QtildeInv*HinvZin),nz,nT); % posterior mean 

if nargout >= 3
    yvec = yctr - Cmat*HinvZin;
    RinvY = Rmat\yvec; %  R^-1 y
    CRinvY = Cmat'*RinvY; % C^T R^-1  y
    quadtrm = yvec'*RinvY - CRinvY'*(zzHess\CRinvY); % quadratic term
    logdettrm = logdet(Rmat) - logdet(QtildeInv) + logdet(zzHess) + ny*nT*log(2*pi); % log-det term
    
    logli = -.5*quadtrm - .5*logdettrm;
end

% % -------------------------------------------------------------------------
% % For debugging purposes, check marginal likelihood using expensive formula
% % -------------------------------------------------------------------------
% yCov = Cmat*inv(QtildeInv)*Cmat'+Rmat; % marginal covariance of y
% ymu = zeros(ny*nT,1); % marginal mean of y
% logli2 = logmvnpdf(yy(:)',ymu, yCov);
% [logli-logli2] 
% % -------------------------------------------------------------------------




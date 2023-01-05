function [zzmap,zzHess,logEv] = computeZmap_LDSBernoulli(yy,mm,uu,optsFminunc,zz0)
% [zzmap,zzHess,logEv] = computeZmap_LDSBernoulli(yy,mm,uu,optsFminunc,zz0)
%
% Computes MAP estimate for latents given binary observations under latent
% LDS-Bernoulli model
%
% INPUTS
% -------
%  yy [ny x nT] - matrix of observations
%  mm  [struct] - model struct with fields
%           .A [nz x nz] - dynamics matrix
%           .B [nz x ns] - input matrix (optional)
%           .C [ny x nz] - latents-to-observations matrix
%           .D [ny x ns] - input-to-observations matrix (optional)
%           .Q [nz x nz] - latent noise covariance
%           .Q0 [nz x nz] - latent noise covariance for 1st time bin
%  uu [nu x nT] - external inputs (OPTIONAL)
% opts [struct] - optimization structure for fminunc (OPTIONAL)
%  zz0 [nz x T] - initial guess at latents (OPTIONAL)
%
% OUTPUTS
% -------
%     zzmap - MAP estimate of latents
%    zzHess - Hessian of negative log-posterior at zzmap
%     logEv - log-evidence at the mode (computed using Laplace approx)


% Basic equations:
% -----------------
% X_t = A*X_{t-1} + B*S_t + w_t,  w_t ~ N(0,Q)   % latent dynamics
% Y_t ~ Bernoulli(C*X_t + D*S_t)                 % observations
%
% Matrix version:
% ---------------
% Model can be rewritten as a pair of matrix equations:
% 
%    Hm*Zvec + B*Uvec = Wvec
%    Yvec ~ Ber(logistic(Cm*Zvec + D*Uvec))
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
% Cm is block-diagonal matrix with C along the diagonals, 
% and Wt is a noise vector whose covariance is block-diagonal
% with Q the diagonals.

% ------------------------------------------------------------------
% Parse inputs 
% ------------------------------------------------------------------

% Extract sizes
nz = size(mm.C,2);  % number of latent dimensions
nT = size(yy,2); % number of time bins

% set zz0 if initial value of latents not provided
if nargin < 5 || isempty(zz0)  
    zz0 = zeros(nz,nT);
end

% set optimization params (if necessary)
if nargin < 4 || isempty(optsFminunc)
    optsFminunc = optimoptions('fminunc','algorithm','trust-region',...
        'SpecifyObjectiveGradient',true,'HessianFcn','objective','display','off');
end

% check if intput-obs matrix is provided
if isfield(mm,'D') && ~isempty(mm.D) 
    muy = (mm.D*uu);  % additive intput to observations (as column vectors)
else
    muy = 0;
end

% ------------------------------------------------------------------
% Build relevant block matrices 
% ------------------------------------------------------------------

% % Build relevant block matrix Dm
Hmat = kron(spdiags(ones(nT-1,1),-1,nT,nT),-mm.A) + speye(nz*nT);  % latent dynamics matrix
Cmat = kron(speye(nT),mm.C); % C projection from latents to observed
Qinvmat = kron(speye(nT),inv(mm.Q)); % inverse covariance for latent noise
Qinvmat(1:nz,1:nz) = inv(mm.Q0); % covariance over latents on 1st time step
QtildeInv = Hmat'*(Qinvmat*Hmat);  % H'*inv(Q)*H

% Compute indices for block-diagonal sparse Hessian of log-likelihood
nn = repmat(1:nz,nz,1); % indices over latents
ii = reshape(nn',[],1)+(0:nz:nz*(nT-1));  % row indices (for Hessian)
jj = nn(:)+(0:nz:nz*(nT-1)); % column indices (for Hessian)

% ------------------------------------------------------------------
% Process input-latents if provided
% ------------------------------------------------------------------
if isfield(mm,'B') && ~isempty(mm.B) 
    HinvZin = Hmat\reshape(mm.B*uu,nz*nT,1);  % prior mean of latents: H^-1*B*uu
else
    HinvZin = zeros(nz*nT,1); 
end

% ------------------------------------------------------------------
% Compute MAP estimate of latents z
% ------------------------------------------------------------------

% Set posterior arguments
postargs = {Cmat,yy(:),QtildeInv,mm.C,ii(:),jj(:),HinvZin(:),muy(:)}; % arguments to neg-log posterior

% Make neg-log-posterior function
fneglogpost = @(zz)(neglogpost_LDSBernoulli(zz,postargs{:}));
% HessCheck(fneglogpost,zz0(:)); % OPTIONAL: check Hessian numerically

% Compute MAP estimate 
zzmap = fminunc(fneglogpost,zz0(:),optsFminunc); 
zzmap = reshape(zzmap,size(zz0));

% =======  Optional Outputs ================================ %
if nargout > 1 % Compute covariance
    [neglogpost,~,zzHess] = fneglogpost(zzmap(:)); % compute Hessian
end

if nargout > 2 % Compute log-evidence
    % Compute negative log evidence (using Laplace approximation)
    logEv = -neglogpost + 0.5*logdet(Qinvmat) - 0.5*logdet(zzHess);
end
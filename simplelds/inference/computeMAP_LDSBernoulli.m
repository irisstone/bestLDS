function [zzmap,zzHess,logEv] = computeMAP_LDSBernoulli(zz0,yy,A,C,Q,opts)
% [wmap,Claplace,postargs] = computeMAP_bernoulliGLM(w0,xdat,ydat,lambdaRidge,opts)
%
% Computes MAP estimate for weights under Bernoulli GLM
%
% INPUTS
% -------
%    zz0 [m x T] - initial guess at latents
%     yy [n x T] - Bernoulli observationsk- design matrix
%      A [m x m] - dynamics matrix
%      C [n x m] - observation matrix
%      Q [m x m] - latent noise covariance
%           opts - optimization structure for fminunc (optional)
%
% OUTPUTS
% -------
%     zzmap - MAP estimate of latents
%    zzHess - Hessian of negative log-posterior at zzmap
%     logEv - log-evidence at the mode (computed using Laplace approx)


% Basic equations:
% -----------------
% X_t = A*X_{t-1} + w_t,  w_t ~ N(0,Q)   % latent dynamics
% Y_t ~ Bernoulli(C*X_t)                 % observations
%
% Matrix version:
% ---------------
% Model can be rewritten as a pair of matrix equations:
% 
%    Am*Xvec = Wvec
%    Yvec ~ Ber(Cm*Xvec)
%
% where Dm is a block matrix of 1st order differences from the dynamics
% equation:
% 
% Am = [I      
%        -A  I 
%           -A I 
%               ...
%                 -A I],   
%
% Cm is block-diagonal matrix with C along the diagonals, 
% and Wt is a noise vectors whose covariances is block-diagonal
% with Q the diagonals.


if nargin < 6
    opts = optimoptions('fminunc','algorithm','trust-region',...
        'SpecifyObjectiveGradient',true,'HessianFcn','objective','display','off');
end

% Extract sizes
nz = size(C,2);  % number of latent dimensions
nT = size(yy,2); % number of time bins

if isempty(zz0)
    zz0 = zeros(nz,nT);
end

% % Build relevant block matrix Dm
Amat = kron(spdiags(ones(nT-1,1),-1,nT,nT),-A) + speye(nz*nT);  % latent dynamics matrix
Cmat = kron(speye(nT),C); % C projection from latents to observed
Qinvmat = kron(speye(nT),inv(Q)); % inverse covariance for latent noise
QtildeInv = Amat'*(Qinvmat*Amat);  % D'*inv(Q)*D

% Compute indices for block-diagonal sparse Hessian of log-likelihood
nn = repmat(1:nz,nz,1); % indices over latents
ii = reshape(nn',[],1)+(0:nz:nz*(nT-1));  % row indices
jj = nn(:)+(0:nz:nz*(nT-1)); % column indices

% Set posterior arguments
postargs = {Cmat,yy(:),QtildeInv,C,ii(:),jj(:)}; % arguments to neg-log posterior

% Make neg-log-posterior function
fneglogpost = @(zz)(neglogpost_LDSBernoulli(zz,postargs{:}));
% HessCheck(fneglogpost,zz0(:)); % OPTIONAL: check Hessian numerically

% Compute MAP estimate 
zzmap = fminunc(fneglogpost,zz0(:),opts); 
zzmap = reshape(zzmap,size(zz0));

% =======  Optional Outputs ================================ %
if nargout > 1 % Compute covariance
    [neglogpost,~,zzHess] = fneglogpost(zzmap(:)); % compute Hessian
end
if nargout > 2 % Compute log-evidence 
    % Compute negative log evidence (using Laplace approximation)
    logEv = -neglogpost + 0.5*logdet(Qinvmat) - 0.5*logdet(zzHess);
end
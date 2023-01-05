function [neglogP,grad,Hess] = neglogpost_LDSBernoulli(zz,Cmat,Y,Qinv,C,ii,jj,muz,muy)
% [neglogP,grad,Hess] = neglogpost_LDSBernoulli(wts,X,Y,Qinv,muz,muy)
%
% Compute negative log-posterior of data under logistic regression model,
% plus gradient and Hessian
%
% Inputs:
% -------
%    zz [d x 1]   - latent variables
%  Cmat [n*t x d] - design matrix 
%     Y [n*t x 1] - output (binary vector of 1s and 0s)
%  Qinv [d x d]   - inverse of prior covariance
%     C [n x d]   - output matrix (single block from Cmat)
% ii,jj [d*d*t,1] - vector of indices required for forming sparse block-diagonal
%   muz [t x d]   - additive mean of latents
%   muy [t x n]   - additive mean of observations
%
%
% Outputs:
% --------
%    neglogP [1 x 1] - negative logposterior (up to a constant)
%       grad [d x 1] - gradient
%       Hess [d x d] - Hessian (2nd deriv matrix)
%
% Note: includes only the "penalty" term from log-prior:  0.5 zz'*Qinv*zz

% Parse inputs
if nargin < 8
    muz = 0;
end
if nargin < 9
    muy = 0;
end

% Compute projection of inputs onto GLM weights for each class
xproj = Cmat*zz + muy;  % logit of input to each observation
zzctr = zz - muz;  % zero-mean latent

if nargout <= 1     % Evaluate neglogli only
    
    neglogP = -Y'*xproj + sum(softplus(xproj)); % neg log-likelihood
    neglogP = neglogP + 0.5*sum(zzctr.*(Qinv*zzctr),1); % neg log-posterior
    
elseif nargout == 2 % Evaluate gradient
    [f,df] = softplus(xproj);  % evaluate log-normalizer & deriv

    QinvZ = Qinv*zzctr;  % compute C^{-1} zz
    neglogP = -Y'*xproj + sum(f) + 0.5*sum(zzctr.*(QinvZ),1); % neg log posterior
    grad = Cmat'*(df-Y) + QinvZ;         % gradient

elseif nargout == 3 % Evaluate gradient and Hessian
    [f,df,ddf] = softplus(xproj); % evaluate log-normalizer & derivs
    
    QinvZ = Qinv*zzctr;  % compute C^{-1} w
    neglogP = -Y'*xproj + sum(f) + 0.5*sum(zzctr.*(QinvZ),1); % neg log posterior
    grad = Cmat'*(df-Y) + QinvZ;   % gradient
    
    % Compute hessian
    % Hess2 = C'*(C.*ddf);  % slow way
    
    % fast way
    ny = size(C,1);
    Cddf = C'.*reshape(ddf,1,ny,[]);
    CddfC = pagemtimes(Cddf,C);
    Hess = sparse(ii,jj,CddfC(:)); % Hessian of log-likelihood
        
    Hess = Hess + Qinv;  % Hessian of log-posterior

end

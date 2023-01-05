function [negL,dnegL,H] = neglogli_bernoulliGLM(wts,X,Y)
% [negL,dnegL,H] = neglogli_bernoulliGLM(wts,X,Y)
%
% Compute negative log-likelihood of data under logistic regression model,
% plus gradient and Hessian
%
% Inputs:
% -------
% wts [d x 1] - regression weights
%   X [N x d] - regressors
%   Y [N x 1] - output (binary vector of 1s and 0s).
%
% Outputs:
% --------
%    negL [1 x 1] - negative loglikelihood
%   dnegL [d x 1] - gradient
%       H [d x d] - Hessian (2nd deriv matrix)

% Compute projection of inputs onto GLM weights for each class
xproj = X*wts;

if nargout <= 1     % Evaluate neglogli only
    
    negL = -Y'*xproj + sum(softplus(xproj)); % neg log-likelihood

elseif nargout == 2 % Evaluate gradient

    [f,df] = softplus(xproj);  % evaluate log-normalizer & deriv
    negL = -Y'*xproj + sum(f); % neg log-likelihood
    dnegL = X'*(df-Y);         % gradient

elseif nargout == 3 % Evaluate Hessian
    
    [f,df,ddf] = softplus(xproj); % evaluate log-normalizer & derivs
    negL = -Y'*xproj + sum(f);    % neg log-likelihood
    dnegL = X'*(df-Y);            % gradient
    H = X'*bsxfun(@times,X,ddf);  % Hessian

end

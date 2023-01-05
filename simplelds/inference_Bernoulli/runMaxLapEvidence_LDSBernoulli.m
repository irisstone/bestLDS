function [mm,logEv_final] = runMaxLapEvidence_LDSBernoulli(yy,mm,opts)
% mm = runMaxLapEvidence_LDSBernoulli(yy,mm,opts)
%
% Maximum Laplace-Approximation based evidence estimate for LDS-Bernoulli
% model parameters  
%
% INPUTS
% -------
%     yy [n x T] - Bernoulli observations- design matrix
%    mm [struct] - model struct with fields
%               .A [m x m] - initial estimate of dynamics matrix
%               .C [n x m] - initial estimate of observation matrix
%               .Q [m x m] - latent noise covariance
%          opts - optimization structure for fminunc (optional)
%
% OUTPUTS
% -------
%   mm [struct] - new model struct with fitted params
%   logEv_final - final value of log-evidence

if nargin < 3
    opts = optimset('display', 'iter','MaxFunEvals',1e4);
end

% Extract sizes
csize = size(mm.C);  % # observed dimension; # latents
prs0 = [mm.A(:); mm.C(:)]; % initial params

% Make neg-log-posterior function
floss = @(prs)(neglogev_LDSBernoulli(prs,yy,mm));

% Compute MAP estimate 
[prshat,neglogEv] = fminunc(floss,prs0,opts); 
%[prshat,neglogEv] = fminsearch(floss,prs0,opts); 

% Insert fitted params into struct
[Ahat,Chat] = unvecLDSprs(prshat,csize);
mm.A = Ahat;
mm.C = Chat;

% Log-evidence at optimum
logEv_final = -neglogEv;

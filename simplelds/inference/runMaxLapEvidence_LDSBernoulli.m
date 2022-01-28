function mm = runMaxLapEvidence_LDSBernoulli(yy,mm,opts)
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
%   mm [struct] - model struct with fields 'A', 'C', 'Q', and 'logEvidence'

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

%if nargin < 3
%     opts = optimoptions('fminunc','algorithm','trust-region',...
%         'SpecifyObjectiveGradient',true,'HessianFcn','objective','display','off');
%end

if nargin < 3
    opts = optimset('display', 'iter','MaxFunEvals',1e4);
end

% Extract sizes
csize = size(mm.C);  % # observed dimension; # latents
prs0 = [mm.A(:); mm.C(:)]; % initial params

% Set posterior arguments
postargs = {yy,csize,mm.Q}; % arguments to neg-log evidence

% Make neg-log-posterior function
floss = @(prs)(neglogev_LDSBernoulli(prs,postargs{:}));

% Compute MAP estimate 
[prshat,neglogEv] = fminunc(floss,prs0,opts); 
[Ahat,Chat] = unvecLDSprs(prshat,csize);
mm.A = Ahat;
mm.C = Chat;
mm.logEvidence = -neglogEv;


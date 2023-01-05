function mm = runMaxLapEvidence_LDSBernoulli_wInputs(yy,ss,mm,opts)
% mm = runMaxLapEvidence_LDSBernoulli_wInputs(yy,ss,mm,opts)
%
% Maximum Laplace-Approximation based evidence estimate for LDS-Bernoulli
% model parameters  
%
% INPUTS
% -------
%     yy [n x T] - Bernoulli observations- design matrix
%     ss [d x T] - inputs
%    mm [struct] - model struct with fields
%               .A [m x m] - initial estimate of dynamics matrix
%               .B [m x d] - initial estimate of latent input weights
%               .C [n x m] - initial estimate of observation matrix
%               .D [n x d] - initial estimate of observation input weights
%               .Q [m x m] - latent noise covariance
%          opts - optimization structure for fminunc (optional)
%
% OUTPUTS
% -------
%   mm [struct] - model struct with fields 'A', 'C', 'Q', and 'logEvidence'

% Basic equations:
% -----------------
% X_t = A*X_{t-1} + B*S_t + w_t,  w_t ~ N(0,Q)   % latent dynamics
% Y_t ~ Bernoulli(C*X_t + D*S_t)                 % observations
%
% Matrix version:
% ---------------
% Model can be rewritten as a pair of matrix equations:
% 
%    Am*Xvec + B*Svec = Wvec
%    Yvec ~ Ber(Cm*Xvec + D*Svec)
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

if nargin < 4
    opts = optimset('display', 'iter','MaxFunEvals',1e4);
end

% Extract sizes
csize = size(mm.C);  % # observed dimension; # latents
ns = size(ss,1); % # input dimensions
prs0 = [mm.A(:); mm.C(:); mm.B(:); mm.D(:)]; % initial params

% Set posterior arguments
postargs = {yy,ss,csize,ns,mm.Q}; % arguments to neg-log evidence

% Make neg-log-posterior function
floss = @(prs)(neglogev_LDSBernoulli_wInputs(prs,postargs{:}));

% Compute MAP estimate 
[prshat,neglogEv] = fminunc(floss,prs0,opts); 

% Put fitted params into struct
[Ahat,Chat,Bhat,Dhat] = unvecLDSprs(prshat,csize,ns);
mm.A = Ahat;
mm.B = Bhat;
mm.C = Chat;
mm.D = Dhat;
mm.logEvidence = -neglogEv;


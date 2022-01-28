function [mm,logEvTrace] = runVLEM_LDSBernoulli(yy,mm,optsEM)
% [mm,logEvTrace] = runVLEM_LDSBernoulli(yy,mm,optsEM)
%
% Variational Laplace-EM for LDS-Bernoulli model 
%
% INPUTS
% -------
%     yy  [n x T] - Bernoulli observations- design matrix
%     mm  [struct] - model struct with fields
%                .A [m x m] - initial estimate of dynamics matrix
%                .C [n x m] - initial estimate of observation matrix
%                .Q [m x m] - latent noise covariance
%  optsEM [struct] - optimization params (optional)
%       .maxiter - maximum # of iterations 
%       .dlogptol - stopping tol for change in log-likelihood 
%       .display - how often to report log-li
%
% OUTPUTS
% -------
%          mm [struct]      - model struct with fields 'A', 'C', 'Q', and 'logEvidence'
%  logEvTrace [1 x maxiter] - trace of log-likelihood during EM

% Set EM optimization params if necessary
if nargin < 3
    optsEM.maxiter = 200;
    optsEM.dlogptol = 0.01;
    optsEM.display = inf;
end
if ~isfield(optsEM,'display') || isempty(optsEM.display)
    optsEM.display = inf;
end

% Extract initial params
A = mm.A;
C = mm.C;
Q = mm.Q;
[nobs,nlatent] = size(C);

% Set up variables for EM
logEvTrace = zeros(optsEM.maxiter,1); % trace of log-likelihood
dlogp = inf; % change in log-likelihood
logpPrev = -inf; % prev value of log-likelihood
jj = 0; % counter

% Set options for optimizing of latents (in E step)
MAPopts = optimoptions('fminunc','algorithm','trust-region',...
    'SpecifyObjectiveGradient',true,'HessianFcn','objective','display','off');

while (jj < optsEM.maxiter) && (dlogp>optsEM.dlogptol)
    jj = jj+1; % increment counter
    
    % --- run E step  -------
    [zzmap,H,logev] = computeMAP_LDSBernoulli([],yy,A,C,Q,MAPopts);
    logEvTrace(jj) = logev;

    % --- run M step  -------
    
    % A updates
    [~,Sblkdiag,Sblksub] = invblktridiag_sym(H,nlatent); % compute diag and off-diag blocks of posterior cov
    
    % ---  Display progress ----
    if mod(jj,optsEM.display)==0
        fprintf('EM iter %d:  logli = %-.6g\n',jj,logp);
    end
    
    % Update log-likelihood change
    dlogp = logp-logpPrev; % change in log-likelihood
    logpPrev = logp; % previous log-likelihood

    if dlogp<-1e-6
        warning('Log-likelihood decreased during EM!');
        fprintf('dlogp = %.5g\n', dlogp);
    end

end

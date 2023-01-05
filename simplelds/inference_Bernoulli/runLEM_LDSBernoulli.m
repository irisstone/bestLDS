function [mm,logEvTrace,logev_final,zzmu,zzHess] = runLEM_LDSBernoulli(yy,mm,uu,optsEM)
% [mm,logEvTrace] = runLEM_LDSBernoulli(yy,mm,ss,optsEM)
%
% Maximum marginal likelihood fittin of LDS-Bernoulli model via Laplace-EM
%
% INPUTS
% -------
%     yy [ny x T] - Bernoulli observations- design matrix
%     mm [struct] - model structure with fields
%            .A [nz x nz] - dynamics matrix
%            .B [nz x ns] - input matrix (optional)
%            .C [ny x nz] - latents-to-observations matrix
%            .D [ny x ns] - input-to-observations matrix (optional)
%            .Q [nz x nz] - latent noise covariance
%      uu [nu x T]     - external inputs
%  optsEM [struct] - optimization params (optional)
%       .maxiter - maximum # of iterations 
%       .dlogptol - stopping tol for change in log-likelihood 
%       .display - how often to report log-li
%       .update  - specify which params to update during M step
%
% OUTPUTS
% -------
%          mm [struct]      - model struct with fields 'A', 'C', 'Q', and 'logEvidence'
%  logEvTrace [1 x maxiter] - trace of log-likelihood during EM

% Set EM optimization params if necessary
if nargin < 4 || isempty(optsEM)
    optsEM.maxiter = 100;
    optsEM.dlogptol = 0.01;
    optsEM.display = 10;
    optsEM.update = struct;
end
if ~isfield(optsEM,'display') || isempty(optsEM.display)
    optsEM.display = inf;
end

% Set data structure for external inputs (if not passed in)
if nargin < 3 
    uu = []; % inputs to latents zz
end

% Determine which parameters to update during M step
update.A = ~isfield(optsEM.update,'A') || optsEM.update.A;
update.C = ~isfield(optsEM.update,'C') || optsEM.update.C;
update.Q = ~isfield(optsEM.update,'Q') || optsEM.update.Q;
update.B = isfield(mm,'B') && (~isfield(optsEM.update,'B') || optsEM.update.B);
update.D = isfield(mm,'D') && (~isfield(optsEM.update,'D') || optsEM.update.D);
update.Dynam = (update.A || update.B || update.Q); % update dynamics params
update.Obs = (update.C || update.D); % update observation params
optsEM.update = update;

% Set up variables for EM
logEvTrace = zeros(optsEM.maxiter,1); % log-likelihood over EM iterations
%logpPrev = -inf; % prev value of log-likelihood
jj = 0; % counter

% Set options for optimizing of latents (in E step)
optsFminunc = optimoptions('fminunc','algorithm','trust-region',...
    'SpecifyObjectiveGradient',true,'HessianFcn','objective','display','off');

% Compute MAP estimate of latents using initial params
zzmap = computeZmap_LDSBernoulli(yy,mm,uu,optsFminunc);  %BUG HERE -- should be using B,D,S

while (jj < optsEM.maxiter) %&& (dlogp>optsEM.dlogptol)
    jj = jj+1; % increment counter
    
    % --- run E step  -------
    [zzmap,H,logp] = computeZmap_LDSBernoulli(yy,mm,uu,optsFminunc,zzmap);
    logEvTrace(jj) = logp;
    
    % --- run M step  -------
    mm = runMstep_LDSBernoulli(yy,mm,uu,zzmap,H,optsEM);
    %mm = optsEM.runMstep(yy,mm,ss,zzmap,H,optsEM);
    
    % ---  Display progress ----
    if mod(jj,optsEM.display)==0
        fprintf('--- EM iter %d: logEv= %.3f ---\n', jj,logp);
    end
    
    % ---------------------------------------------------------------------
    % NOT checking for decreases, since they are common with LEM 

    % % Update log-evidence change
    % dlogp = logp-logpPrev; % change in log-likelihood
    % logpPrev = logp; % previous log-likelihood

    % Check if log-evidence decreased
    %if dlogp<-1e-2
    % fprintf('EM iter %d (logEv = %.1f): LOG-EV DECREASED (dlogEv = %-.3g)\n',jj,logp,dlogp);
    %end
    % ---------------------------------------------------------------------

end

% Compute MAP latents and log-evidence at optimum
if nargout > 2
    [zzmu,zzHess,logev_final] =computeZmap_LDSBernoulli(yy,mm,uu,optsFminunc,zzmap);
end



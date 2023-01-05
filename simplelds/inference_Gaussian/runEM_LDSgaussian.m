function [mm,logEvTrace] = runEM_LDSgaussian(yy,mm,uu,optsEM)
% [mm,logEvTrace] = runEM_LDSgaussian(yy,mm,ss,optsEM)
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
%            .R [ny x ny] - latent noise covariance
%      uu [ns x T]     - external inputs (optional)
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
    optsEM.dlogptol = 1e-4;
    optsEM.display = 10;
    optsEM.update = struct;
end
if ~isfield(optsEM,'display') || isempty(optsEM.display)
    optsEM.display = 1;
end

% Set data structure for external inputs (if not passed in)
if nargin < 3 || sum(abs(uu(:)))==0
    uu = []; % inputs to latents zz
end

% Determine which parameters to update during M step
update.A = ~isfield(optsEM.update,'A') || optsEM.update.A;
update.C = ~isfield(optsEM.update,'C') || optsEM.update.C;
update.Q = ~isfield(optsEM.update,'Q') || optsEM.update.Q;
update.R = ~isfield(optsEM.update,'R') || optsEM.update.R;
update.B = isfield(mm,'B') && (~isfield(optsEM.update,'B') || optsEM.update.B);
update.D = isfield(mm,'D') && (~isfield(optsEM.update,'D') || optsEM.update.D);
update.Dynam = (update.A || update.B || update.Q ); % update dynamics params
update.Obs = (update.C || update.D || update.R); % update observation params
optsEM.update = update;

% Set up variables for EM
logEvTrace = zeros(optsEM.maxiter,1); % trace of log-likelihood
logpPrev = -inf; % prev value of log-likelihood
dlogp = inf; % change in log-li 
jj = 0; % counter

while (jj < optsEM.maxiter) && (dlogp>optsEM.dlogptol)
    jj = jj+1; % iteration counter
    
    % --- run E step  -------
    [zzmu,logp,zzcov,zzcov_d1] = runKalmanSmooth(yy,uu,mm);
    logEvTrace(jj) = logp;

    dlogp = logp-logpPrev; % change in log-likelihood
    logpPrev = logp; % update previous log-likelihood

    % Stop if LL decreased (for debugging purposes)
    if dlogp<-1e-6
        warning('EM iter %d (logEv = %.1f): LOG-EV DECREASED (dlogEv = %-.3g)\n',jj,logp,dlogp);
    end

    % --- run M step  -------
    if isempty(uu)
        mm = runMstep_LDSgaussian(yy,mm,zzmu,zzcov,zzcov_d1,optsEM); % if no inputs
    else
        mm = runMstep_LDSgaussian_wInputs(yy,uu,mm,zzmu,zzcov,zzcov_d1,optsEM);
    end
        
    % ---  Display progress ----
    if mod(jj,optsEM.display)==0
        fprintf('--- EM iter %d: logEv= %.3f ---\n', jj,logp);
    end
    
end

% ---- Report EM termination stats ----------
if optsEM.display < inf
    if dlogp<optsEM.dlogptol
        fprintf('EM finished in %d iters (dlogli=%f)\n',jj,dlogp);
    else
        fprintf('EM stopped at MAXITERS=%d iters (dlogli=%f)\n',jj,dlogp);
    end
end

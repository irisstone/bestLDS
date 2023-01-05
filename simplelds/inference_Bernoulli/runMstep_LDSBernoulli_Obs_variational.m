function mm = runMstep_LDSBernoulli_Obs_variational(yy,mm,uu,zzmap,zzcov,optsEM)
% mm = runMstep_Obs_LDSBernoulli_variational(yy,mm,uu,zzmap,Lblkdiag,optsEM
%
% Run variational M-step updates for observation parameters C and D 
%
% Inputs
% =======
%        yy [ny x T] - Bernoulli observations- design matrix
%        mm [struct] - model structure with fields
%           .A [nz x nz] - dynamics matrix
%           .B [nz x ns] - input matrix (optional)
%           .C [ny x nz] - latents-to-observations matrix
%           .D [ny x ns] - input-to-observations matrix (optional)
%           .Q [nz x nz] - latent noise covariance
%        uu [nu x T]     - external inputs
%     zzmap [nz x T]     - MAP estimate of latents
%  Lblkdiag [nz x nz x T] - tensor with [nlatent x nlatent] marginal cov for each time bin
%
% Output
% =======
%  mmnew - new model struct with updated parameters

% Extract sizes
nz = size(mm.A,1); % number of latents
nobs = size(mm.C,1);    % number of observation dimensions
nt = size(zzmap,2);     % number of time bins

% Set hyperparameters for optimization
nMCsamps = optsEM.nMCsamps; % Set number of MCMC samples to use
a = 0; % set amount of smoothing for C updates

% Draw samples from the marginal posterior over latents in each time step
zSamps = zeros(nz,nMCsamps,nt); 
for jj = 1:nt
    Schol = chol(zzcov(:,:,jj));  % cholesky decomposition of covariance
    zSamps(:,:,jj) = Schol'*randn(nz,nMCsamps) + zzmap(:,jj); % samples of latent for this time step
end
zSamps = reshape(permute(zSamps,[3,2,1]),nt*nMCsamps,nz); % reshape to be (nt*nmcmc,nd) matrix

% Set options for optimizing parameters numerically using fminunc
optsMstepfminunc = optimoptions('fminunc','algorithm','trust-region',...
    'SpecifyObjectiveGradient',true,'HessianFcn','objective','display','off');


if isempty(uu) % ---- M-step update to C if no inputs present ------
    
    % Update 'loadings' weights C
    if optsEM.update.C
        
        % update each row of C (could be parallelized)
        for jrow = 1:nobs
            
            % Make neg-log-posterior function
            yvec = repmat(yy(jrow,:)',nMCsamps,1); % stacked outputs for single output dimension
            fneglogli = @(wts)(neglogli_bernoulliGLM(wts,zSamps,yvec));
            %HessCheck(fneglogli,mm.C(jrow,:)'); % OPTIONAL: check Hessian numerically
            
            % maximize log-likelihood using Bernoulli GLM log-li
            crow_new = fminunc(fneglogli,mm.C(jrow,:)',optsMstepfminunc);

            % insert new weights into C
            mm.C(jrow,:) = a*mm.C(jrow,:) + (1-a)*crow_new';
        end
        
    end
    
else  % ---- M-step update to C and D if inputs are present ----------
        
    % update obs matrix C & input matrix D
    if optsEM.update.C && optsEM.update.D
        
        
        uumat = repmat(uu',nMCsamps,1); % replicate the inputs nMCsamps times
        Xdesign = [zSamps,uumat]; % design matrix
        
        % update each row of C & D(could be parallelized)
        for jrow = 1:nobs
            
            % Make neg-log-posterior function
            yvec = repmat(yy(jrow,:)',nMCsamps,1); % stacked outputs for single output dimension
            fneglogli = @(wts)(neglogli_bernoulliGLM(wts,Xdesign,yvec));
            %HessCheck(fneglogli,randn(size(Xdesign,2),1)); % OPTIONAL: check Hessian numerically
            
            w_init = [mm.C(jrow,:)';mm.D(jrow,:)']; % current value of C and D weights for this output
            
            % maximize log-likelihood using Bernoulli GLM log-li
            cd_rownew = fminunc(fneglogli,w_init,optsMstepfminunc); 
            
            % insert new weights into C and D
            mm.C(jrow,:) = a*mm.C(jrow,:) + (1-a)*cd_rownew(1:nz)';
            mm.D(jrow,:) = a*mm.D(jrow,:) + (1-a)*cd_rownew(nz+1:end)';
            
        end
        
    elseif optsEM.update.C
        warning('NOT AVAILABLE: updates for C alone (without D) when inputs present');
    elseif optsEM.update.D
        warning('NOT AVAILABLE: updates for D alone (without C) when inputs present');
    end
    
end


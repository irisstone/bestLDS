function mm = runMstep_LDSBernoulli(yy,mm,uu,zzmap,H,optsEM)
% mm = runMstep_LDSBernoulli(yy,mm,uu,zzmap,H,optsEM)
%
% Run variational M-step updates for Gaussian observation model
%
% Inputs
% =======
%     yy [ny x T] - Bernoulli observations- design matrix
%     mm [struct] - model structure with fields
%           .A [nz x nz] - dynamics matrix
%           .B [nz x ns] - input matrix (optional)
%           .C [ny x nz] - latents-to-observations matrix
%           .D [ny x ns] - input-to-observations matrix (optional)
%           .Q [nz x nz] - latent noise covariance
%     ss [nu x T]     - external inputs
%  zzmap [nz x T]     - MAP estimate of latents
%      H [nz*T x nz*T] - sparse Hessian for latents around zzmap
%
% Output
% =======
%  mmnew - new model struct with updated parameters

% Extract sizes
nz = size(mm.A,1); % number of latents
nt = size(zzmap,2);     % number of time bins

% Compute inverse of diagonal and off-diagonal blocks of covariance from Hessian
[~,zzcov,zzcov_d1] = invblktridiag_sym(H,nz); 

% =============== Update dynamics parameters ==============
if optsEM.update.Dynam

    % Compute sufficient statistics
    Mz1 = sum(zzcov(:,:,1:nt-1),3) + zzmap(:,1:nt-1)*zzmap(:,1:nt-1)'; % E[zz*zz'] for 1 to T-1
    Mz2 = sum(zzcov(:,:,2:nt),3) + zzmap(:,2:nt)*zzmap(:,2:nt)'; % E[zz*zz'] for 2 to T
    Mz12 = sum(zzcov_d1,3) + (zzmap(:,1:nt-1)*zzmap(:,2:nt)');   % E[zz_t*zz_{t+1}'] (above-diag)

    if isempty(uu) % ---- M-step update if no inputs present ------
    
        % update dynamics matrix A
        if optsEM.update.A
            Anew = Mz12'/Mz1;
            mm.A = Anew;
        end
        
        % Update noise covariance Q
        if optsEM.update.Q
            Qnew = (Mz2 + mm.A*Mz1*mm.A' - mm.A*Mz12 - Mz12'*mm.A')/(nt-1);
            Qnew = (Qnew + Qnew')/2;  % symmetrize (neccessary to avoid floating point asymmetries creeping in)
            mm.Q = Qnew;
        end

    else  % ---- M-step update if inputs are present ----------
        
        % Compute sufficient statistics for inputs x latents
        Mu = uu(:,2:nt)*uu(:,2:nt)';     % E[uu*uu'] for 2 to T
        Muz2 = uu(:,2:nt)*zzmap(:,2:nt)'; % E[uu*zz'] for 2 to T
        Muz21 = uu(:,2:nt)*zzmap(:,1:nt-1)'; % E[uu_t*zz_{t-1} for 2 to T
        
        % update dynamics matrix A & input matrix B
        if optsEM.update.A && optsEM.update.B
            % do a joint update for A and B
            Mlin = [Mz12;Muz2]; % from linear terms
            Mquad = [Mz1 Muz21'; Muz21 Mu]; % from quadratic terms
            ABnew = Mlin'/Mquad; % new A and B from regression
            mm.A = ABnew(:,1:nz); % new A
            mm.B = ABnew(:,nz+1:end); % new B
        elseif optsEM.update.A  % update dynamics matrix A only
            Anew = (Mz12'-mm.B*Muz21)/Mz1;  % new A
            mm.A = Anew;
        elseif optsEM.update.B  % update input matrix B only
            Bnew = (Muz2'-mm.A*Muz21')/Mu; % new B
            mm.B = Bnew;
        end
              
        % Update noise covariance Q
        if optsEM.update.Q
            Qnew = (Mz2 + mm.A*Mz1*mm.A' + mm.B*Mu*mm.B' ...
                - mm.A*Mz12 - Mz12'*mm.A' ...
                - mm.B*Muz2 - Muz2'*mm.B' ...
                + mm.A*Muz21'*mm.B' + mm.B*Muz21*mm.A' )/(nt-1);
            mm.Q = (Qnew+Qnew')/2;  % symmetrize (to avoid floating point errors)
        end
    end
    
        
end

% =============== Update observation parameters ==============
if optsEM.update.Obs
    
    % Run variational updates for C and D
    mm = runMstep_LDSBernoulli_Obs_variational(yy,mm,uu,zzmap,zzcov,optsEM);

end


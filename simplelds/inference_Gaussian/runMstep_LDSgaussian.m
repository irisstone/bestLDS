function mm = runMstep_LDSgaussian(yy,mm,zzmu,zzcov,zzcov_d1,optsEM)
% mm = runMstep_LDSgaussian(yy,mm,ss,zzmu,zzcov,zzcov_d1,optsEM)
%
% Run M-step updates for LDS-Gaussian model
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
%           .Q0 [ny x ny] - latent noise covariance
%    zzmu [nz x T]     - posterior mean of latents
%   zzcov [nz*T x nz*T] - sparse Hessian for latents around zzmap
%
% Output
% =======
%  mmnew - new model struct with updated parameters

% Extract sizes
nt = size(zzmu,2);     % number of time bins

% =============== Update dynamics parameters ==============
if optsEM.update.Dynam

    % Compute sufficient statistics
    Mz1 = sum(zzcov(:,:,1:nt-1),3) + zzmu(:,1:nt-1)*zzmu(:,1:nt-1)'; % E[zz*zz'] for 1 to T-1
    Mz2 = sum(zzcov(:,:,2:nt),3) + zzmu(:,2:nt)*zzmu(:,2:nt)'; % E[zz*zz'] for 2 to T
    Mz12 = sum(zzcov_d1,3) + (zzmu(:,1:nt-1)*zzmu(:,2:nt)');   % E[zz_t*zz_{t+1}'] (above-diag)

    % update dynamics matrix A
    if optsEM.update.A  
        Anew = Mz12'/Mz1;
        mm.A = Anew;
    end
    
    % Update noise covariance Q 
    if optsEM.update.Q
        Qnew = (Mz2 + mm.A*Mz1*mm.A' - mm.A*Mz12 - Mz12'*mm.A')/(nt-1);
        %Qnew = (Qnew + Qnew')/2;  % symmetrize
        mm.Q = Qnew;
    end

end

% =============== Update observation parameters ==============
if optsEM.update.Obs
    
    % Compute sufficient statistics
    if optsEM.update.Dynam
        Mz = Mz1 + zzcov(:,:,nt) + zzmu(:,nt)*zzmu(:,nt)';  % re-use Mz1 if possible
    else
        Mz = sum(zzcov,3) + zzmu*zzmu'; % E[zz*zz']
    end
    Mzy = zzmu*yy'; % E[zz*yy']
    
    % update obs matrix C
    if optsEM.update.C  
        Cnew = Mzy'/Mz;
        mm.C = Cnew;
    end
    
    % update obs noise covariance R
    if optsEM.update.R
        My = yy*yy';   % compute suff stat E[yy*yy']
        
        Rnew = (My + mm.C*Mz*mm.C' - mm.C*Mzy - Mzy'*mm.C')/nt;
        %Rnew = (Rnew + Rnew')/2;  % symmetrize
        mm.R = Rnew;

    end

end


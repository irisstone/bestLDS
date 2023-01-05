function mm = runMstep_LDSgaussian_wInputs(yy,uu,mm,zzmu,zzcov,zzcov_d1,optsEM)
% mm = runMstep_LDSgaussian(yy,uu,mm,zzmu,zzcov,zzcov_d1,optsEM)
%
% Run M-step updates for LDS-Gaussian model
%
% Inputs
% =======
%     yy [ny x T] - Bernoulli observations- design matrix
%     uu [ns x T] - external inputs
%     mm [struct] - model structure with fields
%              .A [nz x nz] - dynamics matrix
%              .B [nz x ns] - input matrix (optional)
%              .C [ny x nz] - latents-to-observations matrix
%              .D [ny x ns] - input-to-observations matrix (optional)
%              .Q [nz x nz] - latent noise covariance
%              .Q0 [ny x ny] - latent noise covariance for first latent sample
%     zzmu [nz x T]        - posterior mean of latents
%    zzcov [nz*T x nz*T]   -  diagonal blocks of posterior cov over latents
% zzcov_d1 [nz*T x nz*T-1] - above-diagonal blocks of posterior covariance
%   optsEM [struct] - optimization params (optional)
%       .maxiter - maximum # of iterations 
%       .dlogptol - stopping tol for change in log-likelihood 
%       .display - how often to report log-li
%       .update  - specify which params to update during M step
%
% Output
% =======
%  mmnew - new model struct with updated parameters

% Extract sizes
nz = size(mm.A,1); % number of latents
nt = size(zzmu,2);     % number of time bins

% =============== Update dynamics parameters ==============
if optsEM.update.Dynam

    % Compute sufficient statistics for latents
    Mz1 = sum(zzcov(:,:,1:nt-1),3) + zzmu(:,1:nt-1)*zzmu(:,1:nt-1)'; % E[zz*zz'] for 1 to T-1
    Mz2 = sum(zzcov(:,:,2:nt),3) + zzmu(:,2:nt)*zzmu(:,2:nt)'; % E[zz*zz'] for 2 to T
    Mz12 = sum(zzcov_d1,3) + (zzmu(:,1:nt-1)*zzmu(:,2:nt)');   % E[zz_t*zz_{t+1}'] (above-diag)

    % Compute sufficient statistics for inputs x latents
    Mu = uu(:,2:nt)*uu(:,2:nt)';     % E[uu*uu'] for 2 to T
    Muz2 = uu(:,2:nt)*zzmu(:,2:nt)'; % E[uu*zz'] for 2 to T
    Muz21 = uu(:,2:nt)*zzmu(:,1:nt-1)'; % E[uu_t*zz_{t-1} for 2 to T
        
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
        mm.Q = (Mz2 + mm.A*Mz1*mm.A' + mm.B*Mu*mm.B' ...
            - mm.A*Mz12 - Mz12'*mm.A' ...
            - mm.B*Muz2 - Muz2'*mm.B' ...
            + mm.A*Muz21'*mm.B' + mm.B*Muz21*mm.A' )/(nt-1);
    end
    
end

% =============== Update observation parameters ==============
if optsEM.update.Obs
    
    % Compute sufficient statistics
    if optsEM.update.Dynam
        Mz = Mz1 + zzcov(:,:,nt) + zzmu(:,nt)*zzmu(:,nt)';  % re-use Mz1 if possible
        Mu = Mu + uu(:,1)*uu(:,1)';  % reuse Mu
        Muz = Muz2 + uu(:,1)*zzmu(:,1)'; % reuse Muz
    else
        Mz = sum(zzcov,3) + zzmu*zzmu'; % E[zz*zz'] for 1 to T
        Mu = uu*uu';     % E[uu*uu'] for 1 to T
        Muz = uu*zzmu'; % E[uu*zz'] for 1 to T
    end
    Mzy = zzmu*yy'; % E[zz*yy']
    Muy = uu*yy';   % E[uu*yy']    
    
    % update obs matrix C & input matrix D
    if optsEM.update.C && optsEM.update.D  
        % do a joint update to C and D
        Mlin = [Mzy;Muy]; % from linear terms
        Mquad = [Mz Muz'; Muz Mu]; % from quadratic terms
        CDnew = Mlin'/Mquad; % new A and B from regression
        mm.C = CDnew(:,1:nz); % new A
        mm.D = CDnew(:,nz+1:end); % new B
    elseif optsEM.update.C  % update C only
        Cnew = (Mzy'-mm.D*Muz)/Mz;  % new A
        mm.C = Cnew;
    elseif optsEM.update.D  % update D only
        Dnew = (Muy'-mm.C*Muz')/Mu; % new B
        mm.D = Dnew;        
    end
    
    % update obs noise covariance R
    if optsEM.update.R
        My = yy*yy';   % compute suff stat E[yy*yy']

        mm.R = (My + mm.C*Mz*mm.C' + mm.D*Mu*mm.D' ...
            - mm.C*Mzy - Mzy'*mm.C' ...
            - mm.D*Muy - Muy'*mm.D' ...
            + mm.C*Muz'*mm.D' + mm.D*Muz*mm.C' )/nt;
        
    end

end


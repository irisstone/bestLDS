function [yy,zz] = sampleLDSgauss(mm,nT,uu)
% [yy,zz] = sampleLDSgauss(mm,nT,uu)
%
% Sample data from a latent LDS-Bernoulli model
%
% INPUTS
% -------
%     mm [struct] - model structure with fields
%            .A [nz x nz] - dynamics matrix
%            .B [nz x ns] - input matrix (optional)
%            .C [ny x nz] - latents-to-observations matrix
%            .D [ny x ns] - input-to-observations matrix (optional)
%            .Q [nz x nz] - latent noise covariance
%            .Q0 [nz x nz] - latent noise covariance for first time bin
%            .R [ny x ny] - observation noise
%     nT [1 x 1]   - number of time samples
%     uu [nu x nT] - external inputs (optional)
%
% OUTPUTS
% -------
%      yy [ny x nT] - binary outputs from ny neurons
%      zz [nz x nT] - sampled latents

[ny,nz] = size(mm.C);  % get # of neurons and # of latents

% Initialize latents and outputs
zz = zeros(nz,nT); 
yy = zeros(ny,nT);

% Process inputs
if (nargin < 3) || isempty(uu)
    zin = zeros(1,nT);
    yin = zeros(1,nT);
else
    zin = [zeros(nz,1), mm.B*uu(:,2:end)];  % additive intput to latents
    yin = mm.D*uu;  % additive intput to observations
end

% Sample data for first time bin
zz(:,1) = mvnrnd(zeros(1,nz),mm.Q0)'; % 1st latent (Note: no input in 1st time bin)
yy(:,1) = mm.C*zz(:,1) + yin(:,1) + mvnrnd(zeros(1,ny),mm.R)'; % 1st observation

% Sample data for remaining bins
for jj = 2:nT
    zz(:,jj) = mm.A*zz(:,jj-1) + zin(:,jj) + mvnrnd(zeros(1,nz),mm.Q)'; % latents
    yy(:,jj) = mm.C*zz(:,jj)   + yin(:,jj) + mvnrnd(zeros(1,ny),mm.R)'; % observations
end



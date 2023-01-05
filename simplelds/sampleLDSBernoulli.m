function [yy,zz,yyprob] = sampleLDSBernoulli(mm,nT,uu)
% [yy,zz,yyprob] = sampleLDSBernoulli(mm,nT,uu)
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
%     nT [1 x 1]   - number of time samples
%     uu [nu x nT] - external inputs (optional)
%
% OUTPUTS
% -------
%      yy [ny x nT] - binary outputs from ny neurons
%      zz [nz x nT] - sampled latents
%  yyprob [ny x nT] - projected latents plus external inputs

[ny,nz] = size(mm.C);  % get # of neurons and # of latents

% function handle for logistic
flogistic = @(x)(1./(1+exp(-x)));

% Initialize latents and outputs
zz = zeros(nz,nT); 
yyprob = zeros(ny,nT); 
yy = zeros(ny,nT);

% Process inputs
if (nargin < 3) || isempty(uu)
    zin = zeros(1,nT);
    yin = zeros(1,nT);
else
    zin = [zeros(nz,1), mm.B*uu(:,2:end)];  % additive intput to latents
    yin = mm.D*uu;  % additive intput to observations
end

zz(:,1) = mvnrnd(zeros(1,nz),mm.Q0)';  % latents (Note: no input during 1st time bin)
yyprob(:,1) = flogistic(mm.C*zz(:,1) + yin(:,1)); % projected latents + inputs
yy(:,1) = rand(ny,1)<yyprob(:,1); % observations

for jj = 2:nT
    zz(:,jj) = mm.A*zz(:,jj-1) + zin(:,jj) + mvnrnd(zeros(1,nz),mm.Q)'; % latent dynamics
    yyprob(:,jj) = flogistic(mm.C*zz(:,jj) + yin(:,jj)); % project latents to output dims
    yy(:,jj) = rand(ny,1)<yyprob(:,jj); % sample Bernoulli outputs
end


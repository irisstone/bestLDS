function [neglogEv,zzmap] = neglogev_LDSBernoulli(prs0,yy,mm)
% [neglogEv,zzmap] = neglogev_LDSBernoulli(prs0,mm)
%
% Computes negative log evidence for LDS model
%
% INPUTS
% -------
%   prs0 [m x T] - initial guess at params (A and C as vectors)
%     yy [n x T] - Bernoulli observationsk- design matrix
%   Qinv [m x m] - latent noise covariance
%      C [n x m] - observation matrix
% opts - optimization structure for fminunc (optional)
%
% OUTPUTS
% -------
%      wmap - MAP estimate of weights
%  Claplace - inverse Hessian of log-posterior at wmap
%   mstruct - model struct with likelihood, prior & data embedded


% extract params
[A,C] = unvecLDSprs(prs0,size(mm.C));

% insert into fitting struct
mm.A = A;
mm.C = C;

% compute log-evidence using Laplace approximation (and MAP estimate of latent)
[zzmap,~,logev] = computeZmap_LDSBernoulli(yy,mm);

% Flip sign of log-evidence
neglogEv = -logev;


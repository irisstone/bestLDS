function [neglogEv,zzmap] = neglogev_LDSBernoulli_wInputs(prs0,yy,ss,csize,ns,Q)
% [neglogEv,zzmap] = neglogev_LDSBernoulli_wInputs(prs0,yy,ss,csize,ns,Q)
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



[A,C,B,D] = unvecLDSprs(prs0,csize,ns);
MAPopts = optimoptions('fminunc','algorithm','trust-region',...
    'SpecifyObjectiveGradient',true,'HessianFcn','objective','display','off');

[zzmap,~,logev] = computeMAP_LDSBernoulli_wInputs([],yy,ss,A,B,C,D,Q,MAPopts);

neglogEv = -logev;


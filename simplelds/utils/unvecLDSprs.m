function [A,C] = unvecLDSprs(prs,csize)
% [A,C] = unvecLDSprs(prs,csize)
% 
% Takes LDS model parameters as a vector and outputs them as matrices A and C
%
% INPUT:
%    v - vector of LDS model parameters [A(:); C(:)]

ny = csize(1);
nz = csize(2);

A = reshape(prs(1:nz^2),nz,nz);
C = reshape(prs(nz^2+1:end),ny,nz);

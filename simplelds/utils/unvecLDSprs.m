function [A,C,B,D] = unvecLDSprs(prs,csize,ns)
% [A,C] = unvecLDSprs(prs,csize)
%   or
% [A,C,B,D] = unvecLDSprs(prs,csize,ns)
% 
% Takes LDS model parameters as a vector and outputs them as matrices A, C, B, D
%
% INPUT:
%    v - vector of LDS model parameters [A(:); C(:)]
%        or [A(:); B(:); C(:); D(:)]

if nargin < 3
    ns = 0;
end

ny = csize(1); % # of observations
nz = csize(2); % # of latents

nAprs = nz^2; % # params in A
nACprs = nAprs + nz*ny; % # params in A and C
nACBprs = nAprs + nz*ny + nz*ns; % # params in A, C, and B

if (nargin == 2)
    ns = 0;
end
A = reshape(prs(1:nAprs),nz,nz);
C = reshape(prs(nAprs+1:nACprs),ny,nz);

B = reshape(prs(nACprs+1:nACBprs),nz,ns);
D = reshape(prs(nACBprs+1:nACBprs+ny*ns),ny,ns);

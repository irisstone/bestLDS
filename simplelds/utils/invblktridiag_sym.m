function [MinvDiag,MinvDiagBlocks,MinvAboveDiagBlocks] = invblktridiag_sym(M,nn)
% [MinvDiag,MinvDiagBlocks,MinvAboveDiagBlocks] = invblktridiag_sym(M,nn)
%
% Find central blocks of the inverse of a square symmetric block tri-diagonal matrix
% using O(N) recursive method of Rybicki & Hummer (1991).
%
% If M is the inverse covariance matrix, then L0 contains the central
% blocks of the marginal covariance 
% of the 
%
% Inputs:  
% ------
%   M = symmetric, block-tridiagonal matrix
%  nn = size of blocks
%
% Output: 
% -------
%  MinvDiag - vector with diagonal elements of inverse of M
%  MinvBlocks - central blocks of inverse of M (nn x nn x T tensor)
%  MinvAboveDiagBlocks - above-diag blocks (nn x nn x T-1 tensor)

nblocks = size(M,1)/nn; % number of total blocks

% Matrices to store during recursions
A = zeros(nn,nn,nblocks); % for below-diagonal blocks
B = zeros(nn,nn,nblocks); % for diagonal blocks
D = zeros(nn,nn,nblocks); % quantity to compute
E = zeros(nn,nn,nblocks); % quantity to compute

% Initialize first D block
inds = 1:nn; % indices for 1st block
B(:,:,1) = M(inds,inds);
A(:,:,2) = M(inds+nn,inds);
D(:,:,1) = B(:,:,1)\A(:,:,2)';

% Initialize last E block
inds = (nblocks-1)*nn+(1:nn);  % indices for last block
A(:,:,nblocks) = M(inds,inds-nn);
B(:,:,nblocks) = M(inds,inds);
E(:,:,nblocks) = B(:,:,nblocks)\A(:,:,nblocks);

% Extract blocks A & B
for ii = 2:nblocks-1
    inds = (ii-1)*nn+1:ii*nn; % indices for center block
    A(:,:,ii) = M(inds,inds-nn); % below-diagonal block
    B(:,:,ii) = M(inds,inds); % middle diagonal block
end
    
% Make a pass through data to compute D and E
for ii = 2:nblocks-1
    % Forward recursion
    D(:,:,ii) = (B(:,:,ii)-A(:,:,ii)*D(:,:,ii-1))\A(:,:,ii+1)'; 
    
    % backward recursion
    jj = nblocks-ii+1;
    E(:,:,jj) = (B(:,:,jj)-A(:,:,jj+1)'*E(:,:,jj+1))\A(:,:,jj); 
end
    
% Now form blocks of inverse covariance
I = eye(nn);
MinvDiagBlocks = zeros(nn,nn,nblocks);
MinvAboveDiagBlocks = zeros(nn,nn,nblocks-1);
MinvDiagBlocks(:,:,1) = (B(:,:,1)*(I-D(:,:,1)*E(:,:,2)))\I;
MinvDiagBlocks(:,:,nblocks) = (B(:,:,nblocks)-A(:,:,nblocks)*D(:,:,nblocks-1))\I;
for ii = 2:nblocks-1
    % compute diagonal blocks of inverse
    MinvDiagBlocks(:,:,ii) = ((B(:,:,ii)-A(:,:,ii)*D(:,:,ii-1))*(I-D(:,:,ii)*E(:,:,ii+1)))\I;
    % compute above-diagonal blocks
    MinvAboveDiagBlocks(:,:,ii-1) = -(D(:,:,ii-1)*MinvDiagBlocks(:,:,ii));
end
MinvAboveDiagBlocks(:,:,nblocks-1) = -(D(:,:,nblocks-1)*MinvDiagBlocks(:,:,nblocks)); 

% Extract just the diagonal elements
MinvDiag = zeros(nn*nblocks,1);
for ii = 1:nblocks
    MinvDiag((ii-1)*nn+1:ii*nn,1) = diag(MinvDiagBlocks(:,:,ii));
end




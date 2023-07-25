function SIG = generateCovariancesWithInput(seq,hS,varargin)
%
% computes cov([u(t); ... ; u(t+2*hS-1); y(t); ... ; y(t+2*hS-1)])
%
%

useHflag = false;
assignopts(who,varargin);

if useHflag 
  U = [seq.h];
else
  U = [seq.s];
end

Y = [seq.y];
[uDim N] = size(U);
yDim = size(Y,1);

U = bsxfun(@minus, U, mean(U,2));
Y = bsxfun(@minus, Y, mean(Y,2));

SIGUU = zeros(uDim*2*hS);
SIGUY = zeros(uDim*2*hS,yDim*2*hS);
SIGYY = zeros(yDim*2*hS);


Ushift = U;
Yshift = Y;

covUU = U*Ushift';
covUY = U*Yshift';
covYY = Y*Yshift';

for k=1:(2*hS)
  
  uIDX = (k-1)*uDim+1:k*uDim;
  yIDX = (k-1)*yDim+1:k*yDim;
  
  SIGUU(uIDX,uIDX) = covUU/2;
  SIGUY(uIDX,yIDX) = covUY/2;
  SIGYY(yIDX,yIDX) = covYY/2;
    
end

for k=1:(2*hS-1)
  
  Ushift = circshift(Ushift,[0 1]);
  Yshift = circshift(Yshift,[0 1]);
  
  covUU  = U*Ushift';
  covUY  = U*Yshift';
  covUYm = Ushift*Y';
  covYY  = Y*Yshift';
  
  for kk=1:(2*hS-k)
    
    uIDX = (kk-1)*uDim+1:kk*uDim;
    yIDX = (kk-1)*yDim+1:kk*yDim;
    SIGUU(uIDX+k*uDim,uIDX) = covUU;
    SIGYY(yIDX+k*yDim,yIDX) = covYY;
    SIGUY(uIDX+k*uDim,yIDX) = covUY/2;
    SIGUY(uIDX,yIDX+k*yDim) = covUYm/2;
    
  end
  
end

SIG = [SIGUU SIGUY; SIGUY' SIGYY];
SIG = (SIG+SIG')./(N);

SIG = SIG+eye(size(SIG,1))*1e-5;
function [A C Q R Pi] = ssidSVD(SIGfp,SIGyy,xDim,varargin)
%
% ssid based in the alogrithm D3, p351 in Katayama
%
% Input: 
%
%   - SIGfp: Hankel matrix of covariance cov([y_{t+1};...;y_{t+k+1}],[y_{t};...;y_{t-k}])
%     where k = hankelSize
%   - SIGyy = cov(y_t,y_t)
%   - xDim: model dimension <= hankelSize !
%   - minVar for psd Q,R;
%
% Output:
%
%   - model parameters A,C,Q,R
%   - stationary covariance Pi

usePiHack = 0; 
minVar    = 1e-5;
minVarPi  = 1e-5;

assignopts(who,varargin);

yDim = size(SIGyy,1);


% SVD on Hankel matrix SIGfp

[UU,SS,VV] = svd(SIGfp);
UU  = UU(:,1:xDim); SS = SS(1:xDim,1:xDim); VV = VV(:,1:xDim);
SSs = sqrt(SS);


% A,C

Obs   = UU*SS;
Con   = VV';

A = Obs(1:end-yDim,:)\Obs(yDim+1:end,:);
C = Obs(1:yDim,:);
Chat = Con(1:xDim,1:yDim)';

% covariances Pi,Q,R

try
  Pi = dare(A',-C',zeros(xDim),-SIGyy,Chat');

catch
  warning('Cannot solve DARE, using heuristics; this might lead to poor estimates of Q and Q0')
  Pi = A\Chat'*pinv(C');
    
if usePiHack
  disp('/\ -----try new Pi hack----- /\')
  Pi    = pinv(C)*SIGyy*pinv(C');
    
end
end

[V D] = eig(Pi); D=diag(D); D(D<minVarPi)=minVarPi;
Pi = V*diag(D)*V';
Pi = real((Pi+Pi')/2);

Q = Pi-A*Pi*A';
[V D] = eig(Q); D=diag(D); D(D<minVar)=minVar;
Q = V*diag(D)*V';
Q = (Q+Q')/2;

R = diag(SIGyy-C*Pi*C');
R(R<minVar)=minVar;
R = diag(R);

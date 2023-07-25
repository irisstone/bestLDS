function [A C Q R] = ssidCCA(SIGfp,SIGff,SIGpp,xDim,yDim)
%
% ssid based in the alogrithm D4, p352 in Katayam
%
% Input: 
%
%   - SIGfp: Hankel matrix of coraviance cov([y_{t+1};...;y_{t+k+1}],[y_{t};...;y_{t-k}])
%     where k = hankelSize
%   - SIGff = cov([y_{t+1};...;y_{t+k+1}])
%   - SIGpp = cov([y_{t};...;y_{t-k}])
%   - xDim: model dimension < hankelSize !
%   - minVar for psd Q,R;
%
% Output:
%
%   - model parameters A,C,Q,R


hankelSize = size(SIGfp,1)./yDim;

% CCA on SIGfp

[Uf,Sf,Vf] = svd(SIGff);
Sf   = sqrt(Sf);  
L    = Uf*Sf*Vf'; 
Linv = Vf*diag(1./diag(Sf))*Uf';

[Up,Sp,Vp] = svd(SIGpp);
Sp   = sqrt(Sp);
M    = Up*Sp*Vp';
Minv = Vp*diag(1./diag(Sp))*Up';

OC = Linv*SIGfp*Minv';
[UU,SS,VV] = svd(OC);
UU = UU(:,1:xDim); SS = SS(1:xDim,1:xDim); VV = VV(:,1:xDim);
SSs = sqrt(SS);

% A,C

Obs = L*UU*SSs;
Con = SSs*VV'*M';

A = Obs(1:end-yDim,:)\Obs(yDim+1:end,:);
C = Obs(1:yDim,:);
Q = SS-A*SS*A';
R = SIGff(1:yDim,1:yDim)-C*SS*C';

function [A C Q Rc S] = ssidN4SIDnoInput(SIG,xDim,yDim,hS,varargin)
%
%
% function [A C Q Rc S] = ssidN4SIDnoInput(SIG,xDim,yDim,hS,varargin)
%
% "simple" N4SID without input, p90, fig 3.13 from  
% SUBSPACE IDENTIFICATION FOR LINEAR SYSTEMS
% Peter VAN OVERSCHEE, Bart DE MOOR
%
% Identifies parameters of:
%
% x(t+1) = Ax(t) + v(t)
% y(t)   = Cx(t) + w(t)
%
% cov([v;w]) = [Q S; S' R]'
%
%
% Input:
%
%       - SIG = var([y(t); ... ; y(t+2*hS-1)])
%       - xDim: latent dimension
%       - yDim: observed dimension
%       - hS:   Hankel-size >! xDim
%
%     
% Output:
%
%       - System matrices: A,B,C,D
%       - covariances: Q,Rc,S
%

econFlag = true;
assignopts(who,varargin);

% Cholesky decomposition of SIG, corresponds to QR of data matrix

R = chol(SIG)';


% first projection

O = R(yDim*hS+1:end,1:yDim*hS);
if econFlag
  [U S V] = svds(O,xDim); 
else
  [U S V] = svd(O,0); 
end

U = U(:,1:xDim);
S = diag(S); S = S(1:xDim); Ss = sqrt(S); Ss = diag(Ss);
V = V(:,1:xDim);

Gam   = U*Ss;
Gam_  = Gam(1:end-yDim,:);

clear('U','S','V');


% second projection

Op = R(yDim*(hS+1)+1:end,1:yDim*(hS+1));

clear('Lupp','Lufp','Lypp');

X  = Gam\O;
Xp = Gam_\Op;

clear('O','Op','Gam','Gam_');


% doing the regression
%keyboard
S0 = [X zeros(xDim,yDim)];
S1 = [Xp; R(yDim*hS+1:yDim*(hS+1),1:yDim*(hS+1))];

clear('R','X','Xp');

  
AC = S1/S0;


A = AC(1:xDim,1:xDim);
C = AC(1+xDim:end,1:xDim);


Res = S1 - AC*S0;
QRS = Res*Res';
Q   = QRS(1:xDim,1:xDim);
Rc  = QRS(1+xDim:end,1+xDim:end);
S   = QRS(1:xDim,1+xDim:end);


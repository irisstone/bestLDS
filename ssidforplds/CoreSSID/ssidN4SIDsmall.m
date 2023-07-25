function [A B C D Q Rc S O] = ssidN4SIDsmall(SIG,xDim,uDim,yDim,hS)
%
% "robust" N4SID, p131, fig 4.8 from  
% SUBSPACE IDENTIFICATION FOR LINEAR SYSTEMS
% Peter VAN OVERSCHEE, Bart DE MOOR
%
% Input:
%
%       - SIG = cov([u(t); ... ; u(t+2*hS-1); y(t); ... ; y(t+2*hS-1)])
%       - system dimensions xDim,uDim,yDim
%       - Hankel-size hS
%
%


% Cholesky decomposition of SIG, corresponds to QR of data matrix

R = chol(SIG)';


% 1

Rtmp = R(end+1-yDim*hS:end,1:hS*(2*uDim+yDim))/R(1:hS*(2*uDim+yDim),1:hS*(2*uDim+yDim));
Lup  = Rtmp(:,1:uDim*hS);
Luf  = Rtmp(:,1+uDim*hS:2*uDim*hS);
Lyp  = Rtmp(:,1+uDim*hS*2:end);


% 2

R2313 = R(uDim*hS+1:2*hS*uDim,1:2*hS*uDim);
%PI = eye(2*uDim*hS)-R2313'*inv(R2313*R2313')*R2313;% check das
PI = eye(2*uDim*hS)-R2313\R2313; 


% 3

O = (Lup*R(1:uDim*hS,1:uDim*hS*2)+Lyp*R(2*uDim*hS+1:2*uDim*hS+yDim*hS,1:uDim*hS*2))*PI;
O = [O Lyp*R(2*uDim*hS+1:2*uDim*hS+yDim*hS,2*uDim*hS+1:2*uDim*hS+yDim*hS)];
[U S V] = svd(O);
U = U(:,1:xDim); S = S(1:xDim,1:xDim); V = V(:,1:xDim); 
Ssqrt = sqrt(S);

Gam  = U*Ssqrt;
Gam_ = Gam(1:end-yDim,:);


% 4

Tl = [Gam_\R(end+1-yDim*(hS-1):end,1:hS*(2*uDim+yDim)+yDim);R(hS*(2*uDim+yDim)+1:hS*(2*uDim+yDim)+yDim,1:hS*(2*uDim+yDim)+yDim)];
Tr = [Gam\ R(end+1-yDim*hS:   end,1:hS*(2*uDim+yDim)),zeros(xDim,yDim);R(uDim*hS+1:2*hS*uDim,1:hS*(2*uDim+yDim)+yDim)];    

AC = Tl/Tr;
A  = AC(1:xDim,1:xDim);
C  = AC(xDim+1:end,1:xDim);


% 5

Gam  = zeros(yDim*hS,xDim);
for k=1:hS
  Gam((k-1)*yDim+1:k*yDim,:) = C*A^(k-1);
end
Gam_ = Gam(1:end-yDim,:);
Tl = [Gam_\R(end+1-yDim*(hS-1):end,1:hS*(2*uDim+yDim)+yDim);R(hS*(2*uDim+yDim)+1:hS*(2*uDim+yDim)+yDim,1:hS*(2*uDim+yDim)+yDim)];
Tr = [Gam\ R(end+1-yDim*hS:   end,1:hS*(2*uDim+yDim)),zeros(xDim,yDim);R(uDim*hS+1:2*hS*uDim,1:hS*(2*uDim+yDim)+yDim)];    

Pcal = Tl-[A;C]*Tr(1:xDim,:);
Pcal = Pcal(:,1:2*hS*uDim);
Qcal = R(uDim*hS+1:2*hS*uDim,1:hS*2*uDim);

GamInv = pinv(Gam);
Gam_Inv = pinv(Gam_);
L1 = A * GamInv;
L2 = C * GamInv;
M  = [zeros(xDim,yDim),Gam_Inv];
X  = [eye(yDim),zeros(yDim,xDim);zeros(yDim*(hS-1),yDim),Gam_];
  
totm=0;
for k=1:hS
    % Calculate N and the Kronecker products (page 126)
    N = [...
	[M(:,(k-1)*yDim+1:yDim*hS)-L1(:,(k-1)*yDim+1:yDim*hS),zeros(xDim,(k-1)*yDim)]
	[-L2(:,(k-1)*yDim+1:yDim*hS),zeros(yDim,(k-1)*yDim)]];
    if k == 1;
      N(xDim+1:xDim+yDim,1:yDim) = eye(yDim) + N(xDim+1:xDim+yDim,1:yDim);
    end
    N = N*X;
    totm = totm + kron(Qcal((k-1)*uDim+1:k*uDim,:)',N);
end
 


DB = totm\vec(Pcal);
DB = reshape(DB,yDim+xDim,uDim);
D  = DB(1:yDim,:);
B  = DB(1+yDim:end,:);


% 6


Tlr = (Tl-AC*Tr);
QSR = Tlr*Tlr';
%keyboard
Q   = QSR(1:xDim,1:xDim);
Rc  = QSR(xDim+1:end,xDim+1:end);
S   = QSR(1:xDim,xDim+1:end);

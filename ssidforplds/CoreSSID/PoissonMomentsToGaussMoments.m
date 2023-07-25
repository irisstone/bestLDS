function [gamma,Lambda] = PoissonMomentsToGaussMoments(mu,Sigma,FFmin,epseig,minmoment);
%
% function [gamma,Lambda] = PoissonMomentsToGaussMoments(mu,Sigma,FFmin,epseig,minmoment)
%

if nargin<=2
    FFmin=1.02;
end
if nargin<=3
    epseig=1e-5;
end
if nargin<=4
    minmoment=1e-3;
end

assert(min(mu)>0, 'Some of the firing rates are 0, please remove those dimensions');

% !!! take this out and put it into the function description
%minmoment=1e-3; %short data sets
%minmoment=1e-4; %long data sets 
%minmoment=1e-5;  %very long data sets

% rescale covariance matrix such that minimal Fano factor is at least FFmin
FF       = diag(Sigma)./mu;
upscaler = max(1,FFmin./FF);
dd       = repmat(sqrt(upscaler),1,size(Sigma,1));
Sigma    = Sigma.*dd.*(dd');


n = numel(mu);    
M = zeros(size(Sigma)); % 2nd moments

for i=1:n
    M(i,i) = max(minmoment,Sigma(i,i)+mu(i)^2);
    alpha(i,1) = mu(i)^2/sqrt(M(i,i)-mu(i));
    beta(i,1)  = sqrt(M(i,i)-mu(i))/mu(i);
end

gamma  = log(alpha);
Lambda = 2*diag(log(beta));

for i=1:n
    for j=i+1:n
        if nnz(Sigma(i,j))>0
            M(i,j) = max(Sigma(i,j)+mu(i)*mu(j),minmoment);
            Lambda(j,i) = (2*(log(M(i,j))-gamma(i)-gamma(j))-Lambda(i,i)-Lambda(j,j))/2;
            Lambda(i,j) = Lambda(j,i);
        end
        if ~isreal(Lambda(i,j))
            %keyboard
	    %!!! throw error here
        end
    end
end



% check if minmal eigenvalue conditions are fulfilled
if epseig>0
   [a,b]=eig(Lambda);
   if min(diag(b))<epseig;
   %    keyboard
   	 b=diag(max(diag(b),epseig));
    	 Lambda=a*b*a';
    	 Lambda=(Lambda+Lambda')/2;
   end
end

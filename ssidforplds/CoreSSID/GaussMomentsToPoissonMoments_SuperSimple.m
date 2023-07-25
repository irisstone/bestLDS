function [mu,Sigma]=GaussMomentsToPoissonMoments_SuperSimple(gamma,Lambda);
%function [mu,Sigma]=GaussMomentsToPoissonMoments(gamma,Lambda);
% a super simple function that does essentially the same thing as
% GaussMomentsToPoissonMoments.

n=numel(gamma);

mu=exp(gamma+0.5*diag(Lambda));

M=zeros(n);

for i=1:n
    M(i,i)=exp(gamma(i)+0.5*Lambda(i,i))+exp(2*gamma(i)+2*Lambda(i,i));
    for j=i+1:n
        M(j,i)=exp(gamma(i)+gamma(j)+0.5*(2*Lambda(i,j)+Lambda(i,i)+Lambda(j,j)));
        M(i,j)=M(j,i);
    end
end

Sigma=M-mu*mu';

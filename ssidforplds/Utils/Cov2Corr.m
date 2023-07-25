function A=Cov2Corr(A)
%takes in a covariance matrix,  returns the corresponding correlation
%matrix
%std(A)=diag(A)
stdA=repmat(sqrt(diag(A)),1,size(A,1));
A=A./stdA./(stdA');
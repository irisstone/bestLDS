function D=OffDiag(M, nonan)

if  numel(size(M))==2
if  size(M,1)==size(M,2);
take=eye(size(M))==0;
elseif mod(size(M,2)/size(M,1),1)<1e-10 %second dimension is multiple of first
    take=repmat(eye(size(M,1)),1,round(size(M,2)/size(M,1)))==0;
elseif mod(size(M,1)/size(M,2),1)<1e-10
    take=repmat(eye(size(M,2)),round(size(M,1)/size(M,2)),1)==0;
else
take=eye(size(M))==0;    
end
elseif numel(size(M))==3
    take=repmat(eye(size(M,1)),[1,1,size(M,3)])==0;
end

    

D=M(take);

if nargin==2 & nonan==true
    D=D(~isnan(D));
end

function [xx,ww,b,polyc]=gausshe(nk)

%
% Calculation of the zeros and weights of Gauss-Hermite quadrature 
%
% xx= zeros of the polynomial 
% ww= associated weights
%
% Called: factf.m and  roots.m
%
% Thomas Vallée  May 1999
% LEN-C3E, Economics Faculty, University of Nantes
% vallee@sc-eco.univ-nantes.fr
%
%JHM, modified function to change from integration of int f(x) exp(-x.^2)
%dx to int f(x) exp(-x.^2/2) dx /sqrt{2pi}


format long;


n=nk;
if nk>60
    error('took code from someone else, seems to become nonsensical for larger values')
end


if rem(n,2)==0, % n even
	nn=(n/2)+1;
	b=zeros(nn-1,nn-1);
	% calculation of the b(i,j) coefficients
	b(1,1)=-2; 
	for i=2:nn-1,
		b(i,1)=((-1)^(i))*(2+4*(i-1))*abs(b(i-1,1));
	end;
	b(1,2)=4; 
	for i=2:nn-1,
		b(i,i+1)=4*b(i-1,i);
	end;
	for j=2:nn-1,
		valmm=abs(b(j-1,j)/b(j-1,j-1));
		valm=valmm+valmm;
		for i=j:nn-1,
			b(i,j)=(-1)*b(i,j-1)*valm;
			valm=valm+valmm;
		end;
	end;	
	kk=1;
	for i=1:nn,
		poly=b(nn-1,nn-(i-1));
		polyc(kk)=poly;
		polyc(kk+1)=0;
		kk=kk+2;
	end;
	ssp=size(polyc,2);
	polycc=polyc(1:ssp-1)*((-1)^n);
	
	xx=roots(polycc);

	%calculation of the weights
	for i=1:n,
		polyd(i)=polycc(i)*(n+1-i);
	end;
	for i=1:n,
		x=xx(i);
		solde=0;
		for k=1:n,
			solde=solde+polyd(k)*(x^(n-k));
		end;
		ww(i,1)=((2^(n+1))*factf(n)*(pi^(.5)))/(solde^2);
	end;

end;





if rem(n,2)==1, % n odd
	nn=(n+1)/2;
	b=zeros(nn,nn);
	b(1,1)=-2; 
	for i=2:nn,
		b(i,1)=((-1)^(i))*(2+4*(i-1))*abs(b(i-1,1));
	end;
	b(2,2)=-8; 
	for i=3:nn,
		b(i,i)=4*b(i-1,i-1);
	end;
	for j=2:nn-1,
		valmm=abs(b(j,j)/b(j,j-1));
		valm=valmm+valmm;
		for i=j+1:nn,
			b(i,j)=(-1)*b(i,j-1)*valm;
			valm=valm+valmm;
		end;
	end;	
	kk=1;
	for i=1:nn,
		poly=b(nn,nn-(i-1)); 
		polyc(kk)=poly;
		polyc(kk+1)=0;
		kk=kk+2;
	end;
	polycc=polyc*((-1)^n);

	xx=roots(polycc);

	%calculation of the weights
	for i=1:n,
		polyd(i)=polycc(i)*(n+1-i);
	end;

	for i=1:n,
		x=xx(i);
		solde=0;
		for k=1:n,
			solde=solde+polyd(k)*(x^(n-k));
		end;
		ww(i,1)=((2^(n+1))*factf(n)*(pi^(.5)))/(solde^2);
	end;
		
end;

%keyboard
ww=ww/sum(ww);
xx=xx*sqrt(2);
[xx,index]=sort(xx);
ww=ww(index);


function factf=factf(x)

factf=factorial(x);

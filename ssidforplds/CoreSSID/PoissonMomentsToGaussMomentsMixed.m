function [gamma,Lambda]=PoissonMomentsToGaussMomentsMixed(mu,Sigma,FFmin,epseig,sdimBIG,minMoment);


%apply no transformation to upper sdimbig*sdimbig block, apply
%GaussToPoissonTransformation to lower block, apply
%MixedGaussToPoissonTransformation to Off-Diagonal Blocks. In the end,
%check for positive definiteness, but in a computaationally efficient way,
%and surpress the Off-Diagonal entries (if needed) until the whole matrix
%is positive definite

ydim=size(Sigma,1)-sdimBIG;
disp(size(Sigma))

SigYY=Sigma(end-ydim+1:end,end-ydim+1:end);
SigSS=Sigma(1:sdimBIG,1:sdimBIG);
SigYS=Sigma(end-ydim+1:end,1:sdimBIG);

muYY=mu(end-ydim+1:end);
[gammaY,LambdaYY]=PoissonMomentsToGaussMoments(muYY,SigYY,FFmin,epseig,minMoment);
LambdaSS=SigSS;

LambdaYS=SigYS;
for i=1:sdimBIG
    LambdaYS(i,:)=SigYS(i,:)*exp(-0.5*LambdaYY(i,i)-gammaY(i));
end

gamma=mu;
gamma(end-ydim+1:end)=gammaY;

Lambda=[LambdaSS, LambdaYS'; LambdaYS, LambdaYY];

if epseig>0
[a,b]=eig(Lambda);
if min(diag(b))<epseig;
%    keyboard
    b=diag(max(diag(b),epseig));
    Lambda=a*b*a';
    Lambda=(Lambda+Lambda')/2;
end
end


% $$$ if epseig>0;
% $$$     mineig=eigs(Lambda,1,-10000);
% $$$     %do bijection search to find  rescaler of LambdaYS which is close to
% $$$     %one as possible but gives pd matrix
% $$$     if mineig<0
% $$$     scalerhigh=1;
% $$$     scalerlow=0;
% $$$     scaler=0.5;
% $$$     counter=0;
% $$$     while (mineig<0 || mineig>epseig) && counter<10001;
% $$$         fprintf('\n Matrix not pd, need to supress off-diagonal block %g %g ', mineig, scaler)
% $$$         %keyboard
% $$$         
% $$$         if counter==1000;
% $$$             scaler=0;
% $$$             warning('Fixing matrix did not work, make it block-diagonal');
% $$$         end
% $$$         Lambda=[LambdaSS, scaler*LambdaYS'; scaler*LambdaYS, LambdaYY];
% $$$         
% $$$         mineig=eigs(Lambda,1,-10000);
% $$$         if mineig<=0
% $$$             scaler=0.5*(scaler+scalerlow);
% $$$             scalerlow=scalerlow;
% $$$             scalerhigh=scaler;
% $$$         elseif mineig>epseig
% $$$             scaler=0.5*(scaler+scalerhigh);
% $$$             scalerlow=scaler;
% $$$             scalerhigh=scalerhigh;
% $$$         end
% $$$         %scalernew=scalernew*0;
% $$$         %scalerold=scalerold*0;
% $$$         
% $$$         
% $$$     end
% $$$     end
% $$$ end

%keyboard
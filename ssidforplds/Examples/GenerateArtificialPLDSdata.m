function [seq, params] = GenerateArtificialPLDSdata(xDim,yDim,Trials,T,Bdim,varargin)
%
% [seq, params] = GenerateArtificialPLDSdata(varargin)
%


alpha      = 0.972; 
alphaR     = 0.005;
saveflag   = false;
tauB       = 50;
models     = [];

assignopts(who,varargin);

useB = Bdim>0;

if ~isempty(models)
    use_flex=true;
    models=set_default_options(models,yDim);
else
    use_flex = false;
end


%%%%%%%%%%%%%%%%%%%%%%%%%% generate PLDS params %%%%%%%%%%%%%%%%%%%%%%%%%%

Or = randn(xDim); Or = (Or-Or')/2; Or = expm(alphaR*Or);
Qr = randn(xDim); Qr = (Qr-Qr')/2; Qr = expm(alphaR*Qr);
params.A  = Or*diag(rand(xDim,1)*(1-alpha)+alpha)*Qr';
params.d  = randn(yDim,1)*0.25-1.5;
params.Q  = eye(xDim)-params.A*params.A';
params.C  = randn(yDim,xDim); 
%[params.C jk1 jk2] = svd(params.C);     %projection matrix between latent and observed state
params.C  = params.C./xDim.*7.5; 
params.Q0 = dlyap(params.A,params.Q);

params.R  = 1e-10*eye(yDim); %covariance of observation noise

if useB; params.B  = randn(xDim,Bdim)./sqrt(Bdim); end;


%%%%%%%%%%%%%%%%%%%%%%%%%% generate Data %%%%%%%%%%%%%%%%%%%%%%%%%%    


if useB
   taus = ((1:T)/tauB)';
   Bchol = (exp(taus*taus'-0.5*repmat(taus.^2,1,T)-0.5*repmat(taus.^2',T,1))+eye(T)*1e-3);
   Bchol = (Bchol+Bchol')/2;
   Bchol=chol(Bchol);
end

cholQ  = chol(params.Q);
cholQ0 = chol(params.Q0); 

for tr=1:Trials
  seq(tr).x = zeros(xDim,T);
  seq(tr).x(:,1) = cholQ0'*randn(xDim,1);
  for t=2:T
      seq(tr).x(:,t) = params.A*seq(tr).x(:,t-1)+cholQ'*randn(xDim,1);
  end
  if useB
      seq(tr).h = randn(Bdim,T)*Bchol;
      seq(tr).x = seq(tr).x+params.B*seq(tr).h;
  end
  seq(tr).yr = params.C*seq(tr).x+repmat(params.d,1,T);
  if ~use_flex
  seq(tr).y  = poissrnd(exp(seq(tr).yr));
  else
    %  keyboard
  seq(tr).y  = GeneratePoissonData(seq(tr).yr,models);
  end    
end


if saveflag
  trueparams = params;
  save('./Examples/ArtificialPLDSdata.mat','seq','trueparams')
end


%%%%%%%%%%%%%%%%%%%%%%%%%% visualization %%%%%%%%%%%%%%%%%%%%%%%%%%%


%figure
%imagesc(seq(1).y)
%xlabel('t');ylabel('neuron no');

%figure
%plot(seq(1).x','linewidth',2)
%xlabel('t');ylabel('x(t)');

%figure
%plot(seq(1).h','linewidth',2)
%xlabel('t');ylabel('x(t)');


%%%%%%%%%%%%%% !!! put the analysis into the example file
% analyse future-past cross-covariance
% $$$ 
% $$$ 
% $$$ mean(vec([seq.y]))
% $$$ 
% $$$ EigA = eig(tp.A);
% $$$ 
% $$$ SIGzz = tp.C*tp.Pi*tp.C';
% $$$ mY    = exp(0.5*diag(SIGzz)+tp.d);
% $$$ 
% $$$ SIGyy = diag(mY)*(exp(SIGzz)-1)*diag(mY)+diag(mY);
% $$$ 
% $$$ SIGzzFP = generateCovariancesFP(seqZ,hankelSize);
% $$$ SIGyyFP = generateCovariancesFP(seq,hankelSize);
% $$$ svdZ    = svd(SIGzzFP);
% $$$ svdY    = svd(SIGyyFP);
% $$$ 
% $$$ Y  = [seq.y];
% $$$ eY = mean(Y,2);
% $$$ covY = cov(Y');
% $$$ 
% $$$ 
% $$$ figure;hold on
% $$$ plot(eY)
% $$$ plot(mY,'r')
% $$$ 
% $$$ figure
% $$$ imagesc(tp.Pi)
% $$$ 
% $$$ figure
% $$$ scatter(real(EigA),imag(EigA))
% $$$ 
% $$$ figure
% $$$ imagesc([covY SIGyy])
% $$$ 
% $$$ figure
% $$$ imagesc([covY-diag(diag(covY)) SIGyy-diag(diag(SIGyy))])
% $$$ 
% $$$ figure
% $$$ imagesc(tp.C*tp.Pi*tp.C')
% $$$ 
% $$$ figure
% $$$ imagesc(tp.C*tp.A*tp.Pi*tp.C')
% $$$ 
% $$$ figure
% $$$ hist(vec([seq.y]),100)
% $$$ 
% $$$ figure
% $$$ imagesc(Y)
% $$$ 
% $$$ figure; hold on
% $$$ plot(svdZ)
% $$$ scatter(1:hankelSize*yDim,svdZ,'r')
% $$$ figure; hold on
% $$$ plot(svdY)
% $$$ scatter(1:hankelSize*yDim,svdY,'r')

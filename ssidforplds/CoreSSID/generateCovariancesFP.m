function [SIGfp SIGff SIGpp SIGtot] = generateCovariancesFP(seqIn,hankelSize,varargin)
%
% function [SIGfp SIGff SIGpp] = generateCovariances(seqIn,hankelSize,varargin)
%
% Build Hankel & Toeplitz matrices of time-lagges covariances of observations
% taking into account trial structure of data
%
%
% Input: 
%
%
% Opt Input: 
%
%   - trialFlag (default false)  : take trial structure of data into account
%   - circFlag  (default true)   : use all data to estimate all delays 
%   - checkEVFlag (default true) : check if SIGtot is psd, might be slow
%
% Output:
%
%   - SIGtot: Hankel matrix of coraviance cov([y_{t+1};...;y_{t+k}],[y_{t};...;y_{t-k+1}])
%     where k = hankelSize
%     SIGtot = [SIGff SIGfp; SIGfp' SIGpp]
%
%

checkEVFlag = false;
circFlag    = true;
trialFlag   = false;

assignopts(who,varargin);

yDim = size(seqIn(1).y,1);

SIGfp  = zeros(yDim*hankelSize,yDim*hankelSize);
SIGff  = zeros(yDim*hankelSize,yDim*hankelSize);
SIGpp  = zeros(yDim*hankelSize,yDim*hankelSize);


% Build covariance hankelmatrix

if circFlag
  
  Ytot   = [seqIn.y];  
  Ttot   = size(Ytot,2);
  Ytot   = bsxfun(@minus, Ytot, mean(Ytot,2));
  Yshift = Ytot;
  
  lamK = Ytot*Yshift'/2;
  for k=1:hankelSize
    indX = 1:yDim;
    SIGff(indX+(k-1)*yDim,indX+(k-1)*yDim) = lamK; 
  end
  
  SIGpp = SIGff;
  
  %keyboard
  
  for k=1:(2*hankelSize-1)
    
    Yshift = circshift(Yshift,[0 1]);
    lamK   = Ytot*Yshift';
    indX   = 1:yDim;
    
    if k<(hankelSize-0.5)
      for kk=k:(hankelSize-1)
	SIGff(indX+kk*yDim,indX+(kk-k)*yDim) = lamK;
	SIGpp(indX+kk*yDim,indX+(kk-k)*yDim) = lamK';
      end            
    end
    
    if k<(hankelSize+0.5)
      for kk=1:k
	SIGfp(indX+(k-kk)*yDim,indX+(kk-1)*yDim) = lamK;
      end
    else
      for kk=1:(2*hankelSize-k)
	SIGfp(indX+(hankelSize-kk)*yDim,indX+(kk+k-hankelSize-1)*yDim) = lamK;
      end
    end
  end
  
  SIGfp = SIGfp./Ttot;
  SIGff = (SIGff+SIGff')./Ttot;
  SIGpp = (SIGpp+SIGpp')./Ttot;
  
else
  
  if ~trialFlag
    Ytot = [seqIn.y];
    seq(1).y = Ytot;
  else
    seq = seqIn;
  end

  Trials = size(seq,2);
  DhsTot = 0;
  for tr=1:Trials
    
    T   = size(seq(tr).y,2);
    Dhs = T-2*hankelSize+1;
    
    if Dhs>2
      
      DhsTot  = DhsTot+Dhs;
      Yf = zeros(hankelSize*yDim,Dhs);
      Yp = zeros(hankelSize*yDim,T-2*hankelSize+1);
      
      for kk=1:hankelSize
	
	seq(tr).y = bsxfun(@minus, seq(tr).y, mean(seq(tr).y,2));
	
	Yf((kk-1)*yDim+1:kk*yDim,:) = seq(tr).y(:,hankelSize+kk:T-hankelSize+kk);
	Yp((kk-1)*yDim+1:kk*yDim,:) = seq(tr).y(:,hankelSize+1-kk:T-hankelSize-kk+1);
	
      end
      
      SIGfp = SIGfp + Yf*Yp';
      SIGff = SIGff + Yf*Yf';
      SIGpp = SIGpp + Yp*Yp';
      
    end
  end
  
  if DhsTot>2
    SIGfp = SIGfp./DhsTot;
    SIGff = SIGff./DhsTot;
    SIGpp = SIGpp./DhsTot;
  else
    warning('hankelSize too large, cannot estimate Hankel matrix!')
  end
  
end


SIGtot = [SIGff SIGfp; SIGfp' SIGpp];

if checkEVFlag
  [V D] = eig(SIGtot); D = diag(D);
  if min(real(D))<0
    warning('smth wrong, future-past cov not psd; fixing it')
    fprintf('\n min EV: %d \n \n', min(real(D)))
    D(D<0)=0;
    SIGtot = V*diag(D)*V';
    SIGff = SIGtot(1:hankelSize*yDim,1:hankelSize*yDim);
    SIGfp = SIGtot(1:hankelSize*yDim,1+hankelSize*yDim:end);
    SIGpp = SIGtot(1+hankelSize*yDim:end,1+hankelSize*yDim:end);
  end
end

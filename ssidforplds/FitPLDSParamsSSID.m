function [params,SIGBig] = FitPLDSParamsSSID(seq,xDim,varargin)
%
% function [params,SIGBig] = FitParamsPPLDSID(seq,xDim,varargin)
%
% fit parameters of a Poisson LDS using subspace identification
% output-- parameter-struct as usual
%
%
%


algo              = 'SVD';      % select from available algos 'SVD', 'CCA', 'N4SID'
hS                = xDim;  	% Hankel size, recommended minimal value is latent dimension xDim
minFanoFactor     = 1.01;  	% regularizes covariance matrix such that all units have this minimal Fano factor
minEig            = .0001; 	% !!! still in use?
params            = [];    	% initialize parameter array
saveHankel        = 0;     	% store the full Hankel matrix
doNonlinTransform = 1;          % yes if data is Poisson, no if data is Gaussian
useB     	  = 0;	        % external stimulus

yall              = [seq.y];    
[yDim,allT]       = size(yall);
minMoment         = 5/allT;     % regularizes covariance matrix such that minimal 2nd order moments exceed this value
 
extraOptsIn       = assignopts(who, varargin);

disp('Fitting data with PLDS-SSID')
disp('---------------------------')
fprintf('using HankelSize = %i\n',hS);
fprintf('useB = %i \n',useB)
fprintf('doNonlinTransform = %i \n',doNonlinTransform);
  
if useB; algo='N4SID';end; % only N4SID can handle external input


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate future-past covariance matrix & do moment converison
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if ~useB 
  
  [SIGfp SIGff SIGpp] = generateCovariancesFP(seq,hS);  % generate future-past Henkel matrix
     
  mu       = mean(yall,2);
  muBig    = repmat(mu,2*hS,1);
  gammaBig = muBig;    
  SIGBig   = [SIGff, SIGfp; SIGfp', SIGpp];

  % do moment conversion if data is Poisson     
  if doNonlinTransform
       
       [gammaBig,SIGBig] = PoissonMomentsToGaussMoments(muBig,SIGBig,minFanoFactor,minEig,minMoment);

  end
              
  SIGyy  = SIGBig(1:yDim,1:yDim);
  SIGfp  = SIGBig(1:yDim*hS,yDim*hS+1:end);
  SIGff  = SIGBig(1:yDim*hS,1:yDim*hS);
  SIGpp  = SIGBig(yDim*hS+1:end,yDim*hS+1:end);
  SIGBig = [SIGff, SIGfp; SIGfp', SIGpp];
  
  params.d = gammaBig(1:yDim);
  Bdim = 0;


else
  
  Bdim     = size(seq(1).h,1);
  SIGBig   = generateCovariancesWithInput(seq,hS,'useHflag',1);
  MUh      = mean([seq.h],2);
  MUy      = mean([seq.y],2);
  params.d = MUy;
  MUyy     = [repmat(MUh,2*hS,1);repmat(MUy,2*hS,1)];
  
  SIGyy    = SIGBig(Bdim*2*hS+1:2*Bdim*hS+yDim,2*Bdim*hS+1:2*Bdim*hS+yDim);
   
  if doNonlinTransform

    SIGyy_orig = SIGyy;
    [gammaBig,SIGBig] = PoissonMomentsToGaussMomentsMixed(MUyy,SIGBig,minFanoFactor,minEig,Bdim*2*hS,minMoment);
    
    params.d = gammaBig(2*hS*Bdim+1:2*hS*Bdim+yDim);
    SIGyy = SIGBig(Bdim*hS*2+1:Bdim*hS*2+yDim,Bdim*hS*2+1:Bdim*hS*2+yDim);      
   
  end
    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do ssid on converted future-past Hankel matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



switch algo
 
 case 'SVD'
  fprintf('PPLDSID: Using SVD with hS %d \n',hS);
  [params.A params.C params.Q params.R params.Q0] = ssidSVD(SIGfp,SIGyy,xDim,extraOptsIn{:});

 case 'CCA'
  fprintf('PPLDSID: Using CCA with hS %d \n',hS);
    [params.A params.C params.Q params.R] = ssidCCA(SIGfp,SIGff,SIGpp,xDim,yDim);

 case 'N4SID'
  if ~useB 
    fprintf('PPLDSID: Using N4SID with hS %d \n',hS);
    [params.A params.C params.Q params.R] = ssidN4SIDnoInput(SIGBig,xDim,yDim,hS);
  else
    fprintf('PPLDSID: Using N4SID with input with hS %d \n',hS);
    [params.A params.B params.C params.D params.Q params.R] = ssidN4SIDsmall(SIGBig,xDim,Bdim,yDim,hS);
  end

 otherwise
  warning('Unknown SSID algorithm')

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% some parameter post-processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if any(~isreal(params.Q))
    params.Q=real(params.Q);
    %!!! prob want to throw an error message here
end

params.Q=(params.Q+params.Q')/2;

if min(eig(params.Q))<0
    [a,b]=eig(params.Q);
    params.Q=a*max(b,1d-10)*a';
    %!!! project onto pos def matrices
end

if ~isfield(params,'Q0')
   params.Q0 = real(dlyap(params.A,params.Q));
end

if min(eig(params.Q0))<0
    [a,b]=eig(params.Q0);
    params.Q0=a*max(b,1d-10)*a';
    %!!! project onto pos def matrices
end

params.x0     = zeros(xDim,1);
params.R      = diag(diag(params.R));

if saveHankel
    params.SIGBig = SIGBig;
end

disp('Done!');
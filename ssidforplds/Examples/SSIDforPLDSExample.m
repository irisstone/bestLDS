clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%% generate artifical system and sample from it %%%%%%%%%%%%%%%%%%%%%%%%

Trials     = 100;
T          = 1000;	% length of each time-series, i.e. trial
xDim       = 5;     	% dimensionality of latent input
yDim       = 25;    	% dimensionality of observable
Bdim       = 0;     	% dimensionality of stimulus innovations
algo       = 'SVD'; 	% available algorithms 'SVD','CCA','N4SID'

[seq, trueparams] = GenerateArtificialPLDSdata(xDim,yDim,Trials,T,Bdim);


%%%%%%%%%%%%%%%%%%%%%%%%%% estimate parameters from data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[params,SIGBig] = FitPLDSParamsSSID(seq,xDim,'algo',algo);


%%%%%%%%%%%%%%%%%%%%%%%%%% some analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('-----------------')
disp('Some analysis:')
fprintf('\nSubspace angle between true and estimated parameters: %d \n\n',subspace(trueparams.C,params.C))

disp('True eigenspectrum:')
sort(eig(trueparams.A))
disp('Estimated spectrum')
sort(eig(params.A))

disp('Plot of true and estimated data covariance matrix')
figure()
imagesc([trueparams.C*trueparams.Q0*trueparams.C' params.C*params.Q0*params.C'])



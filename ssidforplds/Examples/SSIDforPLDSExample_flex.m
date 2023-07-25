clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%% generate artifical system and sample from it %%%%%%%%%%%%%%%%%%%%%%%%

Trials     = 100;
T          = 1000;	% length of each time-series, i.e. trial
xDim       = 2;     	% dimensionality of latent input
yDim       = 15;    	% dimensionality of observable
Bdim       = 0;     	% dimensionality of stimulus innovations
algo       = 'SVD'; 	% available algorithms 'SVD','CCA','N4SID'

for k=1:floor(yDim/3)
    models(k,1).name='exp_poisson';
end
for k=floor(yDim/3)+1:floor(yDim*2/3);
    models(k,1).name='custom_poisson';
    models(k,1).nonlinearity=@(x)(10./(1+exp(-x)));
end
for k=floor(yDim*2/3)+1:yDim
    models(k,1).name='logexp_poisson';
end
[seq, trueparams] = GenerateArtificialPLDSdata(xDim,yDim,Trials,T,Bdim,'models',models);


%%%%%%%%%%%%%%%%%%%%%%%%%% estimate parameters from data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
[params,SIGBig] = FitPLDSParamsSSID(seq,xDim,'algo',algo);
toc
tic
[params_2,SIGBig_2] = FitPLDSParamsSSID_flex(seq,xDim,'algo',algo,'models',models);
toc

%% %%%%%%%%%%%%%%%%%%%%%%%%%% some analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('-----------------')
disp('Some analysis:')
fprintf('\nSubspace angle between true and estimated parameters: %d \n\n',subspace(trueparams.C,params.C))

disp('True eigenspectrum:')
sort(eig(trueparams.A))
disp('Estimated spectrum, using wrong function')
sort(eig(params.A))
disp('Estimated spectrum, using correct function')
sort(eig(params_2.A))


disp('Plot of true and estimated data covariance matrix, using fast but wrong function')
figure(1)
imagesc([trueparams.C*trueparams.Q0*trueparams.C' params.C*params.Q0*params.C'])


disp('Plot of true and estimated data covariance matrix, using slow but correct function')
figure(2)
imagesc([trueparams.C*trueparams.Q0*trueparams.C' params_2.C*params_2.Q0*params_2.C'])


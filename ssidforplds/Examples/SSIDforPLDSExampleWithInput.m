clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%% generate artifical system and sample from it %%%%%%%%%%%%%%%%%%%%%%%%

Trials     = 1000;
T          = 1000;	% length of each time-series, i.e. trial
xDim       = 2;     	% dimensionality of latent input
yDim       = 5;    	% dimensionality of observable
Bdim       = 3;     	% dimensionality of stimulus innovations
algo       = 'N4SID'; 	% available algorithms 'SVD','CCA','N4SID'

%[seq, trueparams] = GenerateArtificialPLDSdata(xDim,yDim,Trials,T,Bdim);

% instead of simulating data, load from file
folder = 'datasetD';
dataset = 'data-small';
data = load(sprintf('data/em-inits/%s/%s.mat',folder,dataset));

for i=1:1
seq.y = transpose(squeeze(data.y_train(i,:,:)));
seq.h = transpose(squeeze(data.u_train(i,:,:)));
%seq.y = transpose(squeeze(data.y_train(:,:,i)));
%seq.h = transpose(squeeze(data.u_train(:,:,i)));
trueparams.A = data.A;
trueparams.B = data.B;
trueparams.C = data.C;
trueparams.D = data.D;
trueparams.Q = data.Q;
trueparams.Q0 = data.Q0;


%%%%%%%%%%%%%%%%%%%%%%%%%% estimate parameters from data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[params,SIGBig] = FitPLDSParamsSSID(seq,xDim,'algo',algo,'useB',true);
save(sprintf('data/em-inits/%s/poisson-small.mat',folder), "params");

%%%%%%%%%%%%%%%%%%%%%%%%%% some analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('-----------------')
disp('Some analysis:')
fprintf('\nSubspace angle between true and estimated parameters: %d \n\n',subspace(trueparams.C,params.C))

%true_gain = trueparams.C * inv(eye(xDim) - trueparams.A) * trueparams.B + trueparams.D;
%est_gain = params.C * inv(eye(xDim) - params.A) * params.B + params.D;
%fprintf('\nAvg. gain difference between true and estimated parameters: %d \n\n',mean(mean(abs(est_gain - true_gain))))

disp('True eigenspectrum:')
sort(eig(trueparams.A))
disp('Estimated spectrum')
sort(eig(params.A))

disp('Plot of true and estimated data covariance matrix')
figure()
hcov   = cov([seq.h]');
tpCov  = trueparams.C*trueparams.Q0*trueparams.C';
tpCov  = tpCov+trueparams.C*trueparams.B*hcov*trueparams.B'*trueparams.C';
estCov = params.C*params.Q0*params.C';
estCov = estCov + params.C*params.B*hcov*params.B'*params.C';
imagesc([tpCov estCov])
end


% check moment conversion for flexible function, i.e. for the one which
% accepts a different nonlinearity for each dimension
close all, clear all, clc
rng('default')
rng(1)

%work with 20-dimensional data, and with 5000 samples: 
N=20; 
T=10000;


%make random mean and covariance:
mean_gauss=rand(N,1)*4-4;
A=randn(N,3)/5;
cov_gauss=Cov2Corr(A*A'+eye(N)/50);
varo=diag(rand(N,1)+2);
cov_gauss=sqrt(varo)*cov_gauss*sqrt(varo);
var_gauss=diag(cov_gauss);

G=chol(cov_gauss)';

N=numel(mean_gauss);
z=bsxfun(@plus,G*randn(N,T),mean_gauss);


%% scenario 1: Exponential Poisson;

models.name='exp_poisson';
[x]=GeneratePoissonData(z,models,T);
mean_obs_sampled=mean(x')';
cov_obs_sampled=cov(x');
var_obs_sampled=diag(cov_obs_sampled);

[mean_obs_calc,var_obs_calc,cov_obs_calc]=GaussMomentsToPoissonMoments_flex(mean_gauss,var_gauss,cov_gauss,models);
[mean_gauss_comp,var_gauss_comp,cov_gauss_comp]=PoissonMomentsToGaussMoments_flex(mean_obs_sampled,var_obs_sampled,cov_obs_sampled,models);
[mean_gauss_comp_2,cov_gauss_comp_2] = PoissonMomentsToGaussMoments(mean_obs_sampled,cov_obs_sampled,1);


h(1)=figure;
subplot(2,3,1)
plot(mean_obs_sampled,mean_obs_calc,'.')
xlabel('sampled')
ylabel('calculated')
title('Sampled Poisson mean vs calculated one')
eqline

subplot(2,3,2)
plot(var_obs_sampled,var_obs_calc,'.')
xlabel('sampled')
ylabel('calculated')
title('Sampled Poisson variance vs calculated one')
eqline


subplot(2,3,3)
plot(OffDiag(cov_obs_sampled),OffDiag(cov_obs_calc),'.')
xlabel('sampled')
ylabel('calculated')
title('Sampled Poisson covariance vs calculated one')

eqline

subplot(2,3,4)
plot(mean_gauss,mean_gauss_comp,'.')
hold on
plot(mean_gauss,mean_gauss_comp_2,'or')
xlabel('sampled')
ylabel('calculated')
title('True Gaussian mean vs reconstructed one')
legend('Flex function','fast function')
eqline
%(minor differences could be a consequence of the fact that min factor
%factor etc are applied ina slightly different way)

subplot(2,3,5)
plot(var_gauss,var_gauss_comp,'.')
hold on
plot(var_gauss,diag(cov_gauss_comp_2),'or')
xlabel('sampled')
ylabel('calculated')
title('True Gaussian variance vs reconstructed one')
eqline


subplot(2,3,6)
plot(OffDiag(cov_gauss),OffDiag(cov_gauss_comp),'.')
hold on
plot(OffDiag(cov_gauss),OffDiag(cov_gauss_comp_2),'or')
xlabel('sampled')
ylabel('calculated')
title('True Gaussian covariance vs reconstructed one')
eqline





%% scenario 2: Poisson with soft threshold
models.name='softthresh_poisson';
[x]=GeneratePoissonData(z,models,T);
mean_obs_sampled=mean(x')';
cov_obs_sampled=cov(x');
var_obs_sampled=diag(cov_obs_sampled);

[mean_obs_calc,var_obs_calc,cov_obs_calc]=GaussMomentsToPoissonMoments_flex(mean_gauss,var_gauss,cov_gauss,models);
[mean_gauss_comp,var_gauss_comp,cov_gauss_comp]=PoissonMomentsToGaussMoments_flex(mean_obs_sampled,var_obs_sampled,cov_obs_sampled,models);
[mean_gauss_comp_2,cov_gauss_comp_2] = PoissonMomentsToGaussMoments(mean_obs_sampled,cov_obs_sampled,1);


h(2)=figure;
subplot(2,3,1)
plot(mean_obs_sampled,mean_obs_calc,'.')
xlabel('sampled')
ylabel('calculated')
title('Sampled Poisson mean vs calculated one')
eqline

subplot(2,3,2)
plot(var_obs_sampled,var_obs_calc,'.')
xlabel('sampled')
ylabel('calculated')
title('Sampled Poisson variance vs calculated one')
eqline


subplot(2,3,3)
plot(OffDiag(cov_obs_sampled),OffDiag(cov_obs_calc),'.')
xlabel('sampled')
ylabel('calculated')
title('Sampled Poisson covariance vs calculated one')

eqline

subplot(2,3,4)
plot(mean_gauss,mean_gauss_comp,'.')
hold on
plot(mean_gauss,mean_gauss_comp_2,'or')
xlabel('sampled')
ylabel('calculated')
title('True Gaussian mean vs reconstructed one')
legend('Flex function','fast function')
eqline
%(minor differences could be a consequence of the fact that min factor
%factor etc are applied ina slightly different way)

subplot(2,3,5)
plot(var_gauss,var_gauss_comp,'.')
hold on
plot(var_gauss,diag(cov_gauss_comp_2),'or')
xlabel('sampled')
ylabel('calculated')
title('True Gaussian variance vs reconstructed one')
eqline


subplot(2,3,6)
plot(OffDiag(cov_gauss),OffDiag(cov_gauss_comp),'.')
hold on
plot(OffDiag(cov_gauss),OffDiag(cov_gauss_comp_2),'or')
xlabel('sampled')
ylabel('calculated')
title('True Gaussian covariance vs reconstructed one')
eqline



%% scenario 3: Different kind of nonlinearities

Ngo=ceil(N/4);
clear models
for k=1:Ngo,
models(k).name='exp_poisson';
end
for k=Ngo+1:2*Ngo
models(k).name='custom_poisson';
models(k).nonlinearity=@(x)(exp(x));
end
for k=2*Ngo+1:3*Ngo
models(k).name='softthresh_poisson';
end
for k=3*Ngo+1:4*Ngo
models(k).name='logexp_poisson';
end

models=models(1:N);



[x]=GeneratePoissonData(z,models,T);
mean_obs_sampled=mean(x')';
cov_obs_sampled=cov(x');
var_obs_sampled=diag(cov_obs_sampled);

[mean_obs_calc,var_obs_calc,cov_obs_calc]=GaussMomentsToPoissonMoments_flex(mean_gauss,var_gauss,cov_gauss,models);
[mean_gauss_comp,var_gauss_comp,cov_gauss_comp]=PoissonMomentsToGaussMoments_flex(mean_obs_sampled,var_obs_sampled,cov_obs_sampled,models);
[mean_gauss_comp_2,cov_gauss_comp_2] = PoissonMomentsToGaussMoments(mean_obs_sampled,cov_obs_sampled,1);


h(3)=figure;
subplot(2,3,1)
plot(mean_obs_sampled,mean_obs_calc,'.')
xlabel('sampled')
ylabel('calculated')
title('Sampled Poisson mean vs calculated one')
eqline

subplot(2,3,2)
plot(var_obs_sampled,var_obs_calc,'.')
xlabel('sampled')
ylabel('calculated')
title('Sampled Poisson variance vs calculated one')
eqline


subplot(2,3,3)
plot(OffDiag(cov_obs_sampled),OffDiag(cov_obs_calc),'.')
xlabel('sampled')
ylabel('calculated')
title('Sampled Poisson covariance vs calculated one')

eqline

subplot(2,3,4)
plot(mean_gauss,mean_gauss_comp,'.')
hold on
plot(mean_gauss,mean_gauss_comp_2,'or')
xlabel('sampled')
ylabel('calculated')
title('True Gaussian mean vs reconstructed one')
legend('Flex function','fast function')
eqline
%(minor differences could be a consequence of the fact that min factor
%factor etc are applied ina slightly different way)

subplot(2,3,5)
plot(var_gauss,var_gauss_comp,'.')
hold on
plot(var_gauss,diag(cov_gauss_comp_2),'or')
xlabel('sampled')
ylabel('calculated')
title('True Gaussian variance vs reconstructed one')
eqline


subplot(2,3,6)
plot(OffDiag(cov_gauss),OffDiag(cov_gauss_comp),'.')
hold on
plot(OffDiag(cov_gauss),OffDiag(cov_gauss_comp_2),'or')
xlabel('sampled')
ylabel('calculated')
title('True Gaussian covariance vs reconstructed one')
eqline

%% scenario 3: Different kind of nonlinearities

Ngo=ceil(N/4);
clear models
for k=1:Ngo,
models(k).name='exp_poisson';
end
for k=Ngo+1:2*Ngo
models(k).name='custom_poisson';
models(k).nonlinearity=@(x)(exp(x));
end
for k=2*Ngo+1:3*Ngo
models(k).name='softthresh_poisson';
end
for k=3*Ngo+1:4*Ngo
models(k).name='logexp_poisson';
end

models=models(1:N);



[x]=GeneratePoissonData(z,models,T);
mean_obs_sampled=mean(x')';
cov_obs_sampled=cov(x');
var_obs_sampled=diag(cov_obs_sampled);

[mean_obs_calc,var_obs_calc,cov_obs_calc]=GaussMomentsToPoissonMoments_flex(mean_gauss,var_gauss,cov_gauss,models);
[mean_gauss_comp,var_gauss_comp,cov_gauss_comp]=PoissonMomentsToGaussMoments_flex(mean_obs_sampled,var_obs_sampled,cov_obs_sampled,models);
[mean_gauss_comp_2,cov_gauss_comp_2] = PoissonMomentsToGaussMoments(mean_obs_sampled,cov_obs_sampled,1);

%% scenario 4: custom nonlinearity
h(4)=figure;

clear models;
models.name='custom_poisson';
models.nonlinearity=@(x)(1./(1+exp(-x)));


[x]=GeneratePoissonData(z,models,T);
mean_obs_sampled=mean(x')';
cov_obs_sampled=cov(x');
var_obs_sampled=diag(cov_obs_sampled);

[mean_obs_calc,var_obs_calc,cov_obs_calc]=GaussMomentsToPoissonMoments_flex(mean_gauss,var_gauss,cov_gauss,models);
[mean_gauss_comp,var_gauss_comp,cov_gauss_comp]=PoissonMomentsToGaussMoments_flex(mean_obs_sampled,var_obs_sampled,cov_obs_sampled,models);
[mean_gauss_comp_2,cov_gauss_comp_2] = PoissonMomentsToGaussMoments(mean_obs_sampled,cov_obs_sampled,1);


subplot(2,3,1)
plot(mean_obs_sampled,mean_obs_calc,'.')
xlabel('sampled')
ylabel('calculated')
title('Sampled Poisson mean vs calculated one')
eqline

subplot(2,3,2)
plot(var_obs_sampled,var_obs_calc,'.')
xlabel('sampled')
ylabel('calculated')
title('Sampled Poisson variance vs calculated one')
eqline


subplot(2,3,3)
plot(OffDiag(cov_obs_sampled),OffDiag(cov_obs_calc),'.')
xlabel('sampled')
ylabel('calculated')
title('Sampled Poisson covariance vs calculated one')

eqline

subplot(2,3,4)
plot(mean_gauss,mean_gauss_comp,'.')
hold on
plot(mean_gauss,mean_gauss_comp_2,'or')
xlabel('sampled')
ylabel('calculated')
title('True Gaussian mean vs reconstructed one')
legend('Flex function','fast function')
eqline
%(minor differences could be a consequence of the fact that min factor
%factor etc are applied ina slightly different way)

subplot(2,3,5)
plot(var_gauss,var_gauss_comp,'.')
hold on
plot(var_gauss,diag(cov_gauss_comp_2),'or')
xlabel('sampled')
ylabel('calculated')
title('True Gaussian variance vs reconstructed one')
eqline


subplot(2,3,6)
plot(OffDiag(cov_gauss),OffDiag(cov_gauss_comp),'.')
hold on
plot(OffDiag(cov_gauss),OffDiag(cov_gauss_comp_2),'or')
xlabel('sampled')
ylabel('calculated')
title('True Gaussian covariance vs reconstructed one')
eqline





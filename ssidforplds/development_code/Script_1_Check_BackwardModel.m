laddpath ../functions/
addpath ~/ME/Proj/matlab/functions/
close all, clear all, clc
%simple check script: debug forward models (i.e. no inversion yet)

%mean and covari
N=20;
mean_gauss=rand(N,1)*4-4;
rng('default')
rng(1)
A=randn(N,3)/5;
cov_gauss=Cov2Corr(A*A'+eye(N)/100);
varo=diag(rand(N,1)+1);
cov_gauss=sqrt(varo)*cov_gauss*sqrt(varo);
var_gauss=diag(cov_gauss);
indices_1=1:N;
indices_2=1:N;
T=50000;

%% scenario 1: Everything is gaussian
%
models.name='lin_gauss';
[x,z]=SSIDforPLDS_sample_static(mean_gauss,cov_gauss,models,T);
mean_obs_sampled=mean(x');
cov_obs_sampled=cov(x');

cov_in=cov_gauss(indices_1,indices_2);
[mean_obs,var_obs,cov_obs]=SSIDforPLDS_gauss_2_obs(mean_gauss,var_gauss,cov_in,models,'indices_1',indices_1,'indices_2',indices_2);
[mean_gauss_recomp,var_gauss_recomp,cov_gauss_recomp]=SSIDforPLDS_obs_2_gauss(mean_obs,var_obs,cov_obs,models,'indices_1',indices_1,'indices_2',indices_2);


h(1)=figure;
subplot(1,3,1)
plot(mean_gauss_recomp,mean_gauss,'.'); axis equal; axis square;
eqline
title(models.name)
subplot(1,3,2)
plot(var_gauss_recomp,var_gauss,'.'); axis equal; axis square
eqline
subplot(1,3,3)
plot(OffDiag(cov_gauss_recomp),OffDiag(cov_gauss),'.'); axis equal, axis square;
eqline

error_1=mean((cov_gauss_recomp(:)-vec(cov_gauss).^2)/mean(cov_gauss(:).^2));




%% scenario 2: Exponential Poisson;
models.name='exp_poisson';
[x,z]=SSIDforPLDS_sample_static(mean_gauss,cov_gauss,models,T);
mean_obs_sampled=mean(x');
cov_obs_sampled=cov(x');

cov_in=cov_gauss(indices_1,indices_2);
[mean_obs,var_obs,cov_obs]=SSIDforPLDS_gauss_2_obs(mean_gauss,var_gauss,cov_in,models,'indices_1',indices_1,'indices_2',indices_2);
[mean_gauss_recomp,var_gauss_recomp,cov_gauss_recomp]=SSIDforPLDS_obs_2_gauss(mean_obs,var_obs,cov_obs,models,'indices_1',indices_1,'indices_2',indices_2);

h(2)=figure;
subplot(1,3,1)
plot(mean_gauss_recomp,mean_gauss,'.'); axis equal; axis square;
eqline
title(models(1).name)
subplot(1,3,2)
plot(var_gauss_recomp,var_gauss,'.'); axis equal; axis square
eqline
subplot(1,3,3)
plot(OffDiag(cov_gauss_recomp),OffDiag(cov_gauss),'.'); axis equal, axis square;
eqline

error_2=mean((cov_gauss_recomp(:)-vec(cov_gauss).^2)/mean(cov_gauss(:).^2));



%% scenario 3: Poisson with soft threshold
models.name='softthresh_poisson';
[x,z]=SSIDforPLDS_sample_static(mean_gauss,cov_gauss,models,T);
mean_obs_sampled=mean(x');
cov_obs_sampled=cov(x');

cov_in=cov_gauss(indices_1,indices_2);
[mean_obs,var_obs,cov_obs]=SSIDforPLDS_gauss_2_obs(mean_gauss,var_gauss,cov_in,models,'indices_1',indices_1,'indices_2',indices_2);
[mean_gauss_recomp,var_gauss_recomp,cov_gauss_recomp]=SSIDforPLDS_obs_2_gauss(mean_obs,var_obs,cov_obs,models,'indices_1',indices_1,'indices_2',indices_2);

h(3)=figure;
subplot(1,3,1)
plot(mean_gauss_recomp,mean_gauss,'.'); axis equal; axis square;
eqline
title(models.name)
subplot(1,3,2)
plot(var_gauss_recomp,var_gauss,'.'); axis equal; axis square
eqline
subplot(1,3,3)
plot(OffDiag(cov_gauss_recomp),OffDiag(cov_gauss),'.'); axis equal, axis square;
eqline

error_3=mean((cov_gauss_recomp(:)-vec(cov_gauss).^2)/mean(cov_gauss(:).^2));


%% scenario 4: Poisson with custom threshold
models.name='custom_poisson';
models.nonlinearity=@(x)(exp(x));
[x,z]=SSIDforPLDS_sample_static(mean_gauss,cov_gauss,models,T);
mean_obs_sampled=mean(x');
cov_obs_sampled=cov(x');

cov_in=cov_gauss(indices_1,indices_2);
[mean_obs,var_obs,cov_obs]=SSIDforPLDS_gauss_2_obs(mean_gauss,var_gauss,cov_in,models,'indices_1',indices_1,'indices_2',indices_2);
[mean_gauss_recomp,var_gauss_recomp,cov_gauss_recomp]=SSIDforPLDS_obs_2_gauss(mean_obs,var_obs,cov_obs,models,'indices_1',indices_1,'indices_2',indices_2);

h(4)=figure;
subplot(1,3,1)
plot(mean_gauss_recomp,mean_gauss,'.'); axis equal; axis square;
eqline
title(models.name)
subplot(1,3,2)
plot(var_gauss_recomp,var_gauss,'.'); axis equal; axis square
eqline
subplot(1,3,3)
plot(OffDiag(cov_gauss_recomp),OffDiag(cov_gauss),'.'); axis equal, axis square;
eqline

error_4=mean((cov_gauss_recomp(:)-vec(cov_gauss).^2)/mean(cov_gauss(:).^2));

%% scenario 5: Combination of Exponential Poisson and Gaussian
model1.name='exp_poisson';
model2.name='lin_gauss';
models=[repmat(model1,N/2,1);repmat(model2,N/2,1)];
[x,z]=SSIDforPLDS_sample_static(mean_gauss,cov_gauss,models,T);
mean_obs_sampled=mean(x');
cov_obs_sampled=cov(x');

cov_in=cov_gauss(indices_1,indices_2);
[mean_obs,var_obs,cov_obs]=SSIDforPLDS_gauss_2_obs(mean_gauss,var_gauss,cov_in,models,'indices_1',indices_1,'indices_2',indices_2);
[mean_gauss_recomp,var_gauss_recomp,cov_gauss_recomp]=SSIDforPLDS_obs_2_gauss(mean_obs,var_obs,cov_obs,models,'indices_1',indices_1,'indices_2',indices_2);

h(5)=figure;
subplot(1,3,1)
plot(mean_gauss_recomp,mean_gauss,'.'); axis equal; axis square;
eqline
title(models(1).name)
subplot(1,3,2)
plot(var_gauss_recomp,var_gauss,'.'); axis equal; axis square
eqline
subplot(1,3,3)
plot(OffDiag(cov_gauss_recomp),OffDiag(cov_gauss),'.'); axis equal, axis square;
eqline

error_5=mean((cov_gauss_recomp(:)-vec(cov_gauss).^2)/mean(cov_gauss(:).^2));
%% scenario 6: Combination of Exponential Poisson and Gaussian
model1.name='exp_poisson';
model2.name='lin_gauss';
models=[repmat(model2,N/2,1);repmat(model1,N/2,1)];
[x,z]=SSIDforPLDS_sample_static(mean_gauss,cov_gauss,models,T);
mean_obs_sampled=mean(x');
cov_obs_sampled=cov(x');

cov_in=cov_gauss(indices_1,indices_2);
[mean_obs,var_obs,cov_obs]=SSIDforPLDS_gauss_2_obs(mean_gauss,var_gauss,cov_in,models,'indices_1',indices_1,'indices_2',indices_2);
[mean_gauss_recomp,var_gauss_recomp,cov_gauss_recomp]=SSIDforPLDS_obs_2_gauss(mean_obs,var_obs,cov_obs,models,'indices_1',indices_1,'indices_2',indices_2);

h(6)=figure;
subplot(1,3,1)
plot(mean_gauss_recomp,mean_gauss,'.'); axis equal; axis square;
eqline
title(models(1).name)
subplot(1,3,2)
plot(var_gauss_recomp,var_gauss,'.'); axis equal; axis square
eqline
subplot(1,3,3)
plot(OffDiag(cov_gauss_recomp),OffDiag(cov_gauss),'.'); axis equal, axis square;
eqline

error_6=mean((cov_gauss_recomp(:)-vec(cov_gauss).^2)/mean(cov_gauss(:).^2));

%% scenario 7: Combination of Soft Threshold Poisson and Gaussian
model1.name='softthresh_poisson';
model2.name='lin_gauss';
models=[repmat(model2,N/2,1);repmat(model1,N/2,1)];
[x,z]=SSIDforPLDS_sample_static(mean_gauss,cov_gauss,models,T);
mean_obs_sampled=mean(x');
cov_obs_sampled=cov(x');

cov_in=cov_gauss(indices_1,indices_2);
[mean_obs,var_obs,cov_obs]=SSIDforPLDS_gauss_2_obs(mean_gauss,var_gauss,cov_in,models,'indices_1',indices_1,'indices_2',indices_2);
[mean_gauss_recomp,var_gauss_recomp,cov_gauss_recomp]=SSIDforPLDS_obs_2_gauss(mean_obs,var_obs,cov_obs,models,'indices_1',indices_1,'indices_2',indices_2);

h(7)=figure;
subplot(1,3,1)
plot(mean_gauss_recomp,mean_gauss,'.'); axis equal; axis square;
eqline
title(models(1).name)
subplot(1,3,2)
plot(var_gauss_recomp,var_gauss,'.'); axis equal; axis square
eqline
subplot(1,3,3)
plot(OffDiag(cov_gauss_recomp),OffDiag(cov_gauss),'.'); axis equal, axis square;
eqline

error_7=mean((cov_gauss_recomp(:)-vec(cov_gauss).^2)/mean(cov_gauss(:).^2));
%% scenario 8: Combination of Soft Threshold Poisson and Exponential Threshold Poisson
model1.name='exp_poisson';
model2.name='softthresh_poisson';
models=[repmat(model2,N/2,1);repmat(model1,N/2,1)];
[x,z]=SSIDforPLDS_sample_static(mean_gauss,cov_gauss,models,T);
mean_obs_sampled=mean(x');
cov_obs_sampled=cov(x');

cov_in=cov_gauss(indices_1,indices_2);
[mean_obs,var_obs,cov_obs]=SSIDforPLDS_gauss_2_obs(mean_gauss,var_gauss,cov_in,models,'indices_1',indices_1,'indices_2',indices_2);
[mean_gauss_recomp,var_gauss_recomp,cov_gauss_recomp]=SSIDforPLDS_obs_2_gauss(mean_obs,var_obs,cov_obs,models,'indices_1',indices_1,'indices_2',indices_2);

h(8)=figure;
subplot(1,3,1)
plot(mean_gauss_recomp,mean_gauss,'.'); axis equal; axis square;
eqline
title(models(1).name)
subplot(1,3,2)
plot(var_gauss_recomp,var_gauss,'.'); axis equal; axis square
eqline
subplot(1,3,3)
plot(OffDiag(cov_gauss_recomp),OffDiag(cov_gauss),'.'); axis equal, axis square;
eqline

error_8=mean((cov_gauss_recomp(:)-vec(cov_gauss).^2)/mean(cov_gauss(:).^2));




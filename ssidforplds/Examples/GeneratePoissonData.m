function [x]=GeneratePoissonData(z,models,T);
%function [x]=GeneratePoissonData(z,models,T);
%
%a simple function to sample from a multivariate Poisson model
%Generates T samples, based on nonlinearities etc specified in 'models'. 
%
%inputs:
%
%mean_gauss, cov_gauss: mean and covariance of input gaussian:
%models: struct describing the observation models (see check_models for
%details)
%T: number of samples:
%
%outputs: 
%x: Poisson samples;
%y: Gaussian samples:
%
%JH Macke and L Buesing, 01/2014
%
%also see SSIDforPLDS_gauss_2_obs, SSIDforPLDS_obs_2_gauss

[a,b]=size(z);

if numel(models)==1
    models=repmat(models,a,1);
end


x=zeros(size(z));



for k=1:numel(models);
    switch models(k).name
        case 'lin_gauss'
            x(k,:)=z(k,:);
        case 'exp_poisson'
            x(k,:)=poissrnd(exp(z(k,:)));
        case 'logexp_poisson';
            x(k,:)=poissrnd(log(1+exp(z(k,:))));
        case 'softthresh_poisson'
            xx=z(k,:);
            x(k,:)=poissrnd((exp(xx).*(xx<=0)+(1+xx).*(xx>0)));
        case'custom_poisson'
            %error('not implemented yet')
           % keyboard
            x(k,:)=poissrnd(models(k).nonlinearity(z(k,:)));
        %case 'probit_binary';
        %    x(k,:)=(z(k,:)+randn(size(z(k,:))))>0;
        case {'custom_numerical','custom_analytic'}
            error('not implemented yet');     
    end
end






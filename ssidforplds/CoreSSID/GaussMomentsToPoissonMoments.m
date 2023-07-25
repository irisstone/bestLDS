function [mean_obs,cov_obs,models]=GaussMomentsToPoissonMoments(mean_gauss,cov_gauss,varargin)
%function [mean_obs,cov_obs,models]=GaussMomentsToPoissonMoments(mean_gauss,cov_gauss,varargin)
%
%calculate mean and covariance of a Poisson-model driven by a multivariate
%Gaussian with specified mean and covariance, assuming an exponential
%nonlinearity.
%
%inputs: 
% mean_gauss:   vector of size N by 1, means of input-Gaussian
% cov_gauss:    matrix of size N by N (or smaller, see below), covariance of input-Gaussian
%
%outputs:
%
% mean_obs:   vector of size N by 1, means of observed model
% cov_obs:    matrix of size N by N (or smaller, see below), covariance of
% observed Poisson model

%for a description of the underlying algorithms, see L Buesing, JH Macke,
%M Sahani, Advances in Neural Information Processing Systems (NIPS) 25,
%2012
%
% c/o JH Macke and L Buesing, 01/2014

%% check input arguments and parse optional arguments: 
p=inputParser;
p.StructExpand=false;
N=numel(mean_gauss);
p.addRequired('mean_gauss',@(x)(validateattributes(x,{'numeric'},{'vector','size',[N,1]})));
p.addRequired('cov_gauss',@(x)(validateattributes(x,{'numeric'},{'2d'})));
p.addParamValue('minmoment',1e-3);
p.addParamValue('FFmin',1e-3);
p.parse(mean_gauss,cov_gauss,varargin{:});


models=struct('name','exp_poisson');
[models,precomp]=set_default_options(models,N);

%pre-allocate outputs:
mean_obs=zeros(size(mean_gauss));
cov_obs=zeros(size(cov_gauss));
var_gauss=diag(cov_gauss);

%% perform transformation on variances and means:
for k=1:N
    [meanvar_obs]=gauss_2_obs_meanvar(mean_gauss(k),var_gauss(k),models(k),precomp);
    mean_obs(k)=meanvar_obs(1);
    var_obs(k)=meanvar_obs(2);
end



%% perform transformations on covariances.
indices_1=1:N;
indices_2=1:N;
for k=1:numel(indices_1);
    k_1=indices_1(k);
    for kk=1:numel(indices_2);
        k_2=indices_2(kk);
        if k_1==k_2
            %covariances are variances, so can just copy them:
            cov_obs(k,kk)=var_obs(k_1);
        elseif k_1==k && k_2==kk && k_1>k_2
            %exploit symmetry of covariance (in fact, code is a bit stupid
            %in that it only exploits it if indices_1 and indices_2 are
            %1:N, this could easily be improved if necessary)
            cov_obs(k,kk)=cov_obs(kk,k);
        else
            %actual transformation happens in gauss-2_obs_cov
            cov_obs(k,kk)=gauss_2_obs_cov(mean_gauss([k_1,k_2]),var_gauss([k_1,k_2]),cov_gauss(k,kk),mean_obs([k_1,k_2]),var_obs([k_1,k_2]),models([k_1,k_2]),precomp);
        end
    end
end














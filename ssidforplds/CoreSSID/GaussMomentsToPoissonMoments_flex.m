function [mean_obs,var_obs,cov_obs,models]=GaussMomentsToPoissonMoments_flex(mean_gauss,var_gauss,cov_gauss,models,varargin)
%function [mean_obs,var_obs,cov_obs,models]=GaussMomentsToPoissonMoments_flex(mean_gauss,var_gauss,cov_gauss,models,varargin)
%
%calculate mean and covariance of a Poisson-model driven by a multivariate
%Gaussian with specified mean and covariance
%
%inputs: 
% mean_gauss:   vector of size N by 1, means of input-Gaussian
% var_gauss:    vector of size N by 1, variances of input-Gaussian;
% cov_gauss:    matrix of size N by N (or smaller, see below), covariance of input-Gaussian
% models:       a struct with 1 or N elements describing the details of the
% model used for each dimension. If 'models' only has one element, then
% same model is used for each dimension. Set models.name='exp_poisson' for
% Poisson with exponentional nonlinarity, 'softthresh_poisson' for soft
% threshold, or 'lin_gauss' for a gaussian model. 
%See 'check_models' for details, also on how to use other nonlinearities
%
%outputs:
%
% mean_obs:   vector of size N by 1, means of observed model
% var_obs:    vector of size N by 1, variances of observed Poisson model
% cov_obs:    matrix of size N by N (or smaller, see below), covariance of
% observed Poisson model
%
%optional parameters:
%indices_1, indices_2: index-vectors with at most N elements each. If
%supplied, then cov_gauss corresponds to those covariances of the element
%specified by the index vectors, i.e. if COV is the N by N covariance
%matrix, then cov_gauss=COV(indices_1,indices_2);
%integrate_options: options to passed on to the integration routines in
%gauss_2_obs_meanvar and gauss_2_obs_cov, see these function for details.
%At the moment, only a very naive integration routine is implemented.
%
%
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
p.addRequired('var_gauss',@(x)(validateattributes(x,{'numeric'},{'vector','size',[N,1]})));
p.addRequired('cov_gauss',@(x)(validateattributes(x,{'numeric'},{'2d'})));
p.addRequired('models',@(x)(isempty(x) || (isstruct(x) && (numel(x)==1 || numel(x)==N))));
%only evaluate covariance for block cov(indices_1,indices_2). Default is
%1:N, i.e. to evaluate full covariance matrix. Number of entrices in
%indices_1 must match size(cov_gauss,1) and likewise for indices_2
p.addParamValue('indices_1',1:size(cov_gauss,1),@(x)(validateattributes(x(:),{'numeric'},{'vector','size',[size(cov_gauss,1),1]})));
p.addParamValue('indices_2',1:size(cov_gauss,2),@(x)(validateattributes(x(:),{'numeric'},{'vector','size',[size(cov_gauss,2),1]})));
%structures handed over to the optimization or integration routines: If
%left empty, defaults are inserted in the lower-level functions
p.addParamValue('integrate_options',struct,@(x)(isstruct(x)));
p.addParamValue('minmoment',1e-3);
p.addParamValue('FFmin',1e-3);


p.parse(mean_gauss,var_gauss,cov_gauss,models,varargin{:});
indices_1=p.Results.indices_1;
indices_2=p.Results.indices_2;
integrate_options=p.Results.integrate_options;



%% check models, add more fields if necessary, and precompute grids for numerical optimization: 
[models,precomp]=set_default_options(models,N,integrate_options);


%pre-allocate outputs:
mean_obs=zeros(size(mean_gauss));
var_obs=zeros(size(var_gauss));
cov_obs=zeros(size(cov_gauss));


%% perform transformation on variances and means:
for k=1:N
    [meanvar_obs]=gauss_2_obs_meanvar(mean_gauss(k),var_gauss(k),models(k),precomp);
    mean_obs(k)=meanvar_obs(1);
    var_obs(k)=meanvar_obs(2);
end



%% perform transformations on (selected) covariances.
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














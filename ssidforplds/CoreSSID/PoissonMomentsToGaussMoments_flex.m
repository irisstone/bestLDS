function [mean_gauss,var_gauss,cov_gauss,models,fitinfo]=PoissonMomentsToGaussMoments_flex(mean_obs,var_obs,cov_obs,models,varargin)
%function [mean_gauss,var_gauss,cov_gauss,models,fitinfo]=PoissonMomentsToGaussMoments_flex(mean_obs,var_obs,cov_obs,models,varargin)
%
%given mean and covarince of Poisson model driven by a multivariate
%Gaussian, reconstruct the mean and covariance of the latent Gaussians
%
%inputs: 
% mean_obs:   vector of size N by 1, observed means
% var_obs:    vector of size N by 1, observed variances
% cov_obs:    matrix of size N by N (or smaller, see below), observed covariance 
% models:       a struct with 1 or N elements describing the details of the
% model used for each dimension. If 'models' only has one element or is empty, then
% same model is used for each dimension. Set models.name='exp_poisson' for
% Poisson with exponentional nonlinarity, 'softthresh_poisson' for soft
% threshold, or 'lin_gauss' for a gaussian model. 
%See 'check_models' for details, also on how to use other nonlinearities
%
%outputs:
%
% mean_gauss:   vector of size N by 1, means of latent Gaussian model
% var_gauss:    vector of size N by 1, variances of latent Gaussian model
% cov_gauss:    matrix of size N by N (or smaller, see below), covariance of
% latent Gaussian
%models: modified input structure is returned
%fitinfo: various information about fitting accuracy etc.
%
%optional parameters:
%FFmin: minimum fano factor enforced for Poisson observation models 
%minmoment: minimum second moment enforced for Poisson model;
%
%indices_1, indices_2: index-vectors with at most N elements each. If
%supplied, then cov_gauss corresponds to those covariances of the element
%specified by the index vectors, i.e. if COV is the N by N covariance
%matrix, then cov_gauss=COV(indices_1,indices_2);
%optim_options: options to passed on to the matlab-optimization routines in
%obs_2_gauss_meanvar and obs_2_gauss_cov, see these functions for details
%integrate_options: options to passed on to the integration routines in
%gauss_2_obs_meanvar and gauss_2_obs_cov, see these function for details.
%At the moment, only a very naive integration routine is implemented.

%
%for a description of the underlying algorithms, see L Buesing, JH Macke,
%M Sahani, Advances in Neural Information Processing Systems (NIPS) 25,
%2012
%
% c/o JH Macke and L Buesing, 01/2014

%% check input arguments and parse optional arguments:
p=inputParser;
p.StructExpand=false;
N=numel(mean_obs);
p.addRequired('mean_obs',@(x)(validateattributes(x,{'numeric'},{'vector','size',[N,1]})));
p.addRequired('var_obs',@(x)(validateattributes(x,{'numeric'},{'vector','size',[N,1]})));
p.addRequired('cov_obs',@(x)(validateattributes(x,{'numeric'},{'2d'})));
p.addRequired('models',@(x)(isempty(x) || (isstruct(x) && (numel(x)==1 || numel(x)==N))));

%enforce minimum fano factor:
p.addParamValue('FFmin',1.02,@(x)(isscalar(x) && x>=1));
%enforce minimum second moment:
p.addParamValue('minmoment',1e-4,@(x)(isscalar(x) && x>=0));
%if full covariance is estimated, enforce minimal eigenvalue:
p.addParamValue('epseig',1e-5,@(x)(isscalar(x) && x>=0));
%only evaluate covariance for block cov(indices_1,indices_2). Default is
%1:N, i.e. to evaluate full covariance matrix. Number of entrices in
%indices_1 must match size(cov_gauss,1) and likewise for indices_2
p.addParamValue('indices_1',1:size(cov_obs,1),@(x)(validateattributes(x(:),{'numeric'},{'vector','size',[size(cov_obs,1),1]})));
p.addParamValue('indices_2',1:size(cov_obs,2),@(x)(validateattributes(x(:),{'numeric'},{'vector','size',[size(cov_obs,2),1]})));
p.addParamValue('optim_options',struct);
p.addParamValue('integrate_options',struct);


p.parse(mean_obs,var_obs,cov_obs,models,varargin{:});
indices_1=p.Results.indices_1;
indices_2=p.Results.indices_2;
FFmin=p.Results.FFmin;
minmoment=p.Results.minmoment;
epseig=p.Results.epseig;
optim_options=p.Results.optim_options;
integrate_options=p.Results.integrate_options;


%% check models, add more fields if necessary, and precompute grids for numerical optimization: 
[models,precomp]=set_default_options(models,N,integrate_options,optim_options);

%pre-allocate outputs:
mean_gauss=zeros(size(mean_obs));
var_gauss=zeros(size(var_obs));
cov_gauss=zeros(size(cov_obs));

%ensure that variances for poisson model are large enough to achieve min_fano_factor, inflate if
%necessary: 
fanos_original=var_obs./mean_obs;
is_poisson=[models.is_poisson];
is_poisson=is_poisson(:);
fano_violated=(fanos_original<FFmin) & is_poisson;
if any(fano_violated)
    warning('At least one fano-factor to small to be consistent with a Poisson model, variance increased'); 
end
var_obs(fano_violated)= mean_obs(fano_violated).*FFmin;
fitinfo.fano_violated=fano_violated;
fitinfo.var_obs_modified=var_obs;

%% perform transformation on variances and means:
for k=1:N
    [meanvar_gauss,fitinfo.accuracy_meanvar(k,:)]=obs_2_gauss_meanvar(mean_obs(k),var_obs(k),models(k),precomp);
    mean_gauss(k)=meanvar_gauss(1);
    var_gauss(k)=meanvar_gauss(2);
end



%% perform transformations on (selected) covariances.
for k=1:numel(indices_1);
    k_1=indices_1(k);
    for kk=1:numel(indices_2);
        k_2=indices_2(kk);
        if k_1==k_2
            cov_gauss(k,kk)=var_gauss(k_1);
        elseif k_1==k && k_2==kk && k_1>k_2
                        %exploit symmetry of covariance (in fact, code is a bit stupid
            %in that it only exploits it if indices_1 and indices_2 are
            %1:N, this could easily be improved if necessary)
            cov_gauss(k,kk)=cov_gauss(kk,k);
        else
            %all the heavy lifting for covariances happens here:
            cov_obs_corrected=max(cov_obs(k,kk)+prod(var_obs([k_1,k_2])),minmoment)-prod(var_obs([k_1,k_2]));
            [cov_gauss(k,kk),fitinfo.accuracy_cov(k,kk)]=obs_2_gauss_cov(mean_obs([k_1,k_2]),var_obs([k_1,k_2]),cov_obs_corrected,mean_gauss([k_1,k_2]),var_gauss([k_1,k_2]),models([k_1,k_2]),precomp);
        end
    end
end

fitinfo.integrate_options=precomp.integrate_options;
fitinfo.optim_options=precomp.optim_options;
fitinfo.FFmin=FFmin;


% if full covariance matrix has been estimated, check if minmal eigenvalue conditions are fulfilled
if min(size(cov_gauss))== numel(var_gauss) &    epseig>0
   [a,b]=eig(cov_gauss);
   if min(diag(b))<epseig;
   %    keyboard
   	 b=diag(max(diag(b),epseig));
    	 cov_gauss=a*b*a';
    	 cov_gauss=(cov_gauss+cov_gauss')/2;
   end
end


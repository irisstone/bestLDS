function cov_obs=gauss_2_obs_cov(mean_gauss,var_gauss,cov_gauss,mean_obs,var_obs,model,precomp)
%function cov_obs=gauss_2_obs_cov(mean_gauss,var_gauss,cov_gauss,mean_obs,var_obs,model,precomp)
%
%given a bivariate gaussian and two observation models, calculate the
%covariance of the Poisson outputs
%
%inputs:
%mean_gauss, var_gauss: 2 x 1 vectors giving the mean and variance of the
%                       gaussian
%cov_gauss:             1 scalar giving the cross-covariannce in the gaussian terms.
%mean_obs, var_obs:     mean and variance of the observation model, which can
%                       be calculated using gauss_2_obs_meanvar (and, in fact, var_obs is not
%                       actually used, so it does not matter what is being passed in. Also, it is
%                       easy to modify the function such that this is no longer needed)
%model:                 a 2 by 1 struct defining the models. 
%precomp:               precomp: a struct with grids used for numerical
%                       integration 
%
%outputs:               cov_obs, a scalar giving the cross-covariance of
%                       the two outputs
%
%
%
% c/o JH Macke and L Buesing, 01/2014



switch [model(1).name,'.',model(2).name]
    case 'exp_poisson.exp_poisson'; %exponentional Poisson with exponential Poisson: closed form!
        cov_obs=exp(cov_gauss)*prod(mean_obs)-prod(mean_obs);
    case 'exp_poisson.lin_gauss'%exponential Poisson with linear Gaussian: closed form!
        cov_obs=mean_obs(1)*(mean_gauss(2)+cov_gauss)-prod(mean_obs);
    case 'lin_gauss.exp_poisson'; %closed form again, just in other direction
        cov_obs=mean_obs(2)*(mean_gauss(1)+cov_gauss)-prod(mean_obs);
    case 'lin_gauss.lin_gauss'; %two gaussians: nothing to do
        cov_obs=cov_gauss;
    otherwise
        %in the general case, need to do numerical integration: 
        %transform grid so that it is centered and shaped as the input
        %gaussian:
        Sigma=[var_gauss(1),cov_gauss;cov_gauss,var_gauss(2)];
        G=chol(Sigma)';
        x_2d=bsxfun(@plus,G*precomp.x_2d,mean_gauss);
        %evaluate E(x| input) and E(y|input) for the two models, so that we
        %can evaluate E(x*y|input) and integrate this over all inputs to
        %get cov_obs:
        x=model(1).meanvar_given_x_fun(x_2d(1,:),model(1).params{:});
        x=x(1,:);
        y=model(2).meanvar_given_x_fun(x_2d(2,:),model(2).params{:});
        y=y(1,:);
        xy=x.*y;
        cov_obs=xy*precomp.weights_2d'-prod(mean_obs);
                
end
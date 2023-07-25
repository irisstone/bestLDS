function [cov_gauss,delta,output]=obs_2_gauss_cov(mean_obs,var_obs,cov_obs,mean_gauss,var_gauss,model,precomp)
%function [cov_gauss,delta,output]=obs_2_gauss_cov(mean_obs,var_obs,cov_obs,mean_gauss,var_gauss,model,precomp)
%
%
%
%inputs:
%mean_obs,var_obs:      2 by 1 vectors giving the observed means and variances
%cov_obs:               scalar, observed cross-covariance
%mean_gauss, var_gauss: 2 by 1 vectors of the mean and variance of the
%                       gaussian inputs (which can be obtained from
%                       obs_2_gauss_meanvar)
%model:                 a 2 by 1 struct model specifying which models to use (see
%                       check_models for details)
%precomp:               a struct with grid for numerical computation (only
%                       needed for models which require numerical
%                       computation, obviously)
%
%
%outputs:
%cov_gauss:            scalar, inferred cross-covariance of latent gaussian
%delta:                difference between specified observed covariance
%                       (cov_obs) and covariance that one gets from
%                       cov_gauss. Ideally close to 0.
%output:                a struct with information about the numerical
%                       optimization (exitflags etc)
%
%JH Macke and L Buesing, 01/2014
%
%also see: obs_2_gauss_meanvar

output=[];
delta=nan;
optim_options=precomp.optim_options;

switch [model(1).name,'.',model(2).name]
    case 'exp_poisson.exp_poisson'; %exponentional Poisson with exponential Poisson: closed form!
        cov_gauss=log(cov_obs+prod(mean_obs))-log(prod(mean_obs));
        
    case 'exp_poisson.lin_gauss'%exponential Poisson with linear Gaussian: closed form!
        cov_gauss=cov_obs./mean_obs(1);
        
    case 'lin_gauss.exp_poisson'; %closed form again, just in other direction
        cov_gauss=cov_obs./mean_obs(2);
        
        %case 'probit_binary.probit_binary';    %binary with binary, semi-closed form
        % keyboard
        %    gamma=mvncdf([0;0],-mean_gauss,[var_gauss(1)+1,cov_gauss;cov_gauss,var_gauss(2)+1]);
    case 'lin_gauss.lin_gauss';
        cov_gauss=cov_obs;
    otherwise
        %need to go to 'fully numerical mode'. Invert forward function using fminbnd
        %minimal and maximal possible covariance (corresponding to
        %correlation coefficient being plus or minus one)
        cov_min=(-1+1e-4)*sqrt(prod(var_gauss));
        cov_max=(1-1e-4)*sqrt(prod(var_gauss));
        

        
        %objective function for minimization: just squared error:
        obj_fun=@(x) ((gauss_2_obs_cov(mean_gauss,var_gauss,x,mean_obs,var_obs,model,precomp)-cov_obs).^2);
        
        %do numerical optimization, and save outputs:
        [cov_gauss,fval,exitflag,output]= fminbnd(obj_fun,cov_min,cov_max,optim_options);
        output.fval=fval;
        output.exitflag=exitflag;
        output.info='used fminbnd';
        %calculate resulting error:
        delta=gauss_2_obs_cov(mean_gauss,var_gauss,cov_gauss,mean_obs,var_obs,model,precomp)-cov_obs;
end
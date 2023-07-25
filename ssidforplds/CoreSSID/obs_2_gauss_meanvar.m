function [meanvar_gauss,delta,output]=obs_2_gauss_meanvar(mean_obs,var_obs,model,precomp)
%function [meanvar_gauss,delta,output]=obs_2_gauss_meanvar(mean_obs,var_obs,model,precomp)
%
%
%
%inputs:
%mean_gauss, var_gauss: 2 x 1 vectors giving the mean and variance of the
%                       gaussian
%model:                 a 1 by 1 struct defining the models. 
%precomp:               a struct with grid for numerical computation (only
%                       needed for models which require numerical
%                       computation, obviously)
%
%
%outputs:               meanvar_obs, a 2 by 1 vector giving the mean and variance of the output
%delta:                 2 by 1 vector,differences between specified
%                       observed mean and variance and reconstruction
%                       Ideally close to [0;0].
%output:                a struct with information about the numerical
%                       optimization (exitflags etc)
%
%
% c/o JH Macke and L Buesing, 01/2014

if nargin==4
optim_options=precomp.optim_options;
end 
switch model.name
    case 'exp_poisson'
        meanvar_gauss(1,1)=2*log(mean_obs)-0.5*log(var_obs+mean_obs.^2-mean_obs);
        meanvar_gauss(2,1)=log(var_obs+mean_obs.^2-mean_obs)-log(mean_obs.^2);
        delta=[nan,nan];
        output.message='analytical fit';
    case 'lin_gauss'
        meanvar_gauss=[mean_obs;var_obs];
        delta=[nan,nan];
        output.message='analytical fit';
    case {'softthresh_poisson','custom_poisson','logexp_poisson'};
        %need to do brute force optimization of the forward function. Not
        %claiming that this is fast or robust, but it is a first start
        %keyboard
        
        [meanvar_init]=model.obs_2_gauss_meanvar_init(mean_obs,var_obs);
        mean_init=meanvar_init(1);
        var_init=meanvar_init(2);
        log_var_init=log(var_init);
        
        
        fun=@(x)(gauss_2_obs_meanvar(x(1),exp(x(2)),model,precomp)-[mean_obs;var_obs]);
        %fun_min=@(x)(sum(gauss_2_obs_meanvar(x(1),exp(x(2)),model,precomp)-[mean_obs;var_obs]).^2);
        

        
        %options.
        [x_opt,fval,exitflag,output]=fsolve(fun,[mean_init,log_var_init],optim_options);
        %[x_opt_2,fval_2,exitflag_2,output_2]=fminunc(fun_min,[mean_init,log_var_init],options);
        
        output.info='used fminsearch';
        output.fval=fval;
        output.exitflag=exitflag;
        meanvar_gauss(1,1)=x_opt(1);
        meanvar_gauss(2,1)=exp(x_opt(2));
        delta=fval;
        %keyboard
end

function [meanvar_obs]=gauss_2_obs_meanvar(mean_gauss,var_gauss,model,precomp)
%function [meanvar_obs]=gauss_2_obs_meanvar(mean_gauss,var_gauss,model,precomp)
%
%given a univariate gaussian and an observation models, calculate the
%mean and variane of the output.
%
%inputs:
%mean_gauss, var_gauss: 2 x 1 vectors giving the mean and variance of the
%                       gaussian
%model:                 a 1 by 1 struct defining the models. 
%precomp:               precomp: a struct with grids used for numerical
%                       integration 
%
%outputs:               meanvar_obs, a 2 by 1 vector giving the mean and variance of the output
%
%
%
% c/o JH Macke and L Buesing, 01/2014


switch model.name
    case 'exp_poisson'
        %closed form solution:
        mean_obs=exp(0.5*var_gauss+mean_gauss);
        var_obs=mean_obs+exp(diag(var_gauss)).*mean_obs.^2-mean_obs.^2;
        meanvar_obs=[mean_obs;var_obs];
    case 'lin_gauss'
        %trivial solution for gaussian:
        meanvar_obs=[mean_gauss;var_gauss];
    case {'custom_numerical','softthresh_poisson','custom_poisson','logexp_poisson'};
        %shift grid to match mean and variance of gaussian:
        x=precomp.x*sqrt(var_gauss)+mean_gauss;
        w=precomp.weights;
        
        %evaluate mean and variance of model given input:
        dd=model.meanvar_given_x_fun(x,model.params{:});
        mean_given_x=dd(1,:);
        var_given_x=dd(end,:);
        
        %integrate over all inputs to get mean:
        mean_obs=mean_given_x*w';
        %use law of iterated expectation to get variance: 
        %Var(X)=E(Var(X|Y))+ Var(E(X|Y));
        var_mean_given_x=(mean_given_x-mean_obs).^2*w';
        mean_var_given_x=var_given_x*w';
        var_obs=var_mean_given_x+mean_var_given_x;
        
        [meanvar_obs]=[mean_obs;var_obs];
    
        %also give out derivatives to speed up optimization:
        if nargout==3
            error('not implemented yet')
        end
end
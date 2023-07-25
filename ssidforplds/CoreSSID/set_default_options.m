function [models,precomp]=check_models(models,N,integrate_options,optim_options)
%function [models,precomp]=check_models(models,N,integrate_options,optim_options)
%
%perform some book-keeping on model structs, and prepare grids etc for
%numerical computation.

if isempty(models)
    models=struct('name','exp_poisson');
end

if numel(models)>1
    N=numel(models);
else
    models=repmat(models,N,1);
end




for k=1:numel(models);
    switch models(k).name
        case 'lin_gauss'
            models(k).nonlinearity=@(x)(x);
            models(k).meanvar_given_x_fun=@(x)([x;x*0]); % E(y|z)= z; %Var(y|z)=0;
            models(k).obs_2_gauss_meanvar_init=@(x,y)([x;y]); %not really needed because we use closed form solution anyway;
            models(k).need_num_int=false; %do not need numerical integration for this one
            models(k).is_poisson=false; %obviously not a poisson model
        case 'exp_poisson'
            models(k).nonlinearity=@(x)(exp(x));
            models(k).meanvar_given_x_fun=models(k).nonlinearity;
            %models(k).d_nonlinearity=@(x)(exp(x));
            %E(y|z)=nonlinearity(z), as for any Poisson model;
            %Var(y|z)=nonlinearity(z) as well, therefore
            %mean_var_given_x_fun only has a one dimensional output
            models(k).obs_2_gauss_meanvar_init=@(x,y)(obs_2_gauss_meanvar(x,y,struct('name','exp_poisson')));  %not really needed because we use closed form solution anyway;
            models(k).need_num_int=false;
            models(k).is_poisson=true;
        case 'logexp_poisson'
            models(k).nonlinearity=@(x)(log(1+exp(x)));
            models(k).meanvar_given_x_fun=models(k).nonlinearity;
            if ~isfield(models(k),'obs_2_gauss_meanvar_init') || isempty(models(k).obs_2_gauss_meanvar_init);
                models(k).obs_2_gauss_meanvar_init=@(x,y)(obs_2_gauss_meanvar(x,y,struct('name','exp_poisson')));
            end
            models(k).is_poisson=true;
            models(k).need_num_int=true;
            
        case 'softthresh_poisson'
            models(k).nonlinearity=@(x)(exp(x).*(x<=0)+(1+x).*(x>0));
            models(k).meanvar_given_x_fun=models(k).nonlinearity;
            %models(k).d_nonlinarity=@(x)(exp(x)./(1+exp(x)));
            %to invert the mean-var relationship, we use numerical
            %optimization-- this function here specifies the initial guess
            %used--use guess from exponential model, although this is
            %obviously a bit of a poor guess and probably too low;
            % a more intelligent first guess would be to check whether we
            % are in the 'linear' or 'exponential' regime of the function,
            % and then to do a local inversion based on this.
            if ~isfield(models(k),'obs_2_gauss_meanvar_init') || isempty(models(k).obs_2_gauss_meanvar_init);
                models(k).obs_2_gauss_meanvar_init=@(x,y)(obs_2_gauss_meanvar(x,y,struct('name','exp_poisson')));
            end
            models(k).is_poisson=true;
            models(k).need_num_int=true;
        case 'custom_poisson';
            if ~isfield(models(k),'nonlinearity')
                error('Nonlinearity needs to be specified for custom model')
            end
            
            models(k).meanvar_given_x_fun=models(k).nonlinearity;
            if ~isfield(models(k),'obs_2_gauss_meanvar_init') || isempty(models(k).obs_2_gauss_meanvar_init);
                models(k).obs_2_gauss_meanvar_init=@(x,y)(obs_2_gauss_meanvar(x,y,struct('name','exp_poisson')));
            end
            
            models(k).is_poisson=true;
            models(k).need_num_int=true;
            
            % case 'probit_binary'
            %     models(k).mean_given_x_fun=@(x)(normcdf(x));
            %     models(k).var_given_x_fun=@(x)(normcdf(x).*(1-normcdf(x)));
            % case 'dich_gauss'
            %case 'disc_gauss'
            %case 'custom_numeric'
            %case 'custom_analytic';
            
    end
    if ~isfield(models(k),'params') || ~iscell(models(k).params);
        models(k).params={};
    end
    need_num_int(k)=models(k).need_num_int;
end


if nargin<=2
    integrate_options=struct;
end
if nargin<=3
    optim_options=struct;
end


integrate_options=touchField(integrate_options,'mode','naive');
integrate_options=touchField(integrate_options,'limit_1d','5');
integrate_options=touchField(integrate_options,'limit_2d','4');
integrate_options=touchField(integrate_options,'Nint_1d',251);
integrate_options=touchField(integrate_options,'Nint_2d',51);

optim_options=touchField(optim_options,'MaxFunEvals',200);
optim_options=touchField(optim_options,'TolFun',1e-10);
optim_options=touchField(optim_options,'TolX',1e-10);
optim_options=touchField(optim_options,'Display','off');



if any(need_num_int)
    switch integrate_options.mode
        case 'naive'
            precomp.x=linspace(-integrate_options.limit_1d,integrate_options.limit_1d,integrate_options.Nint_1d);
            precomp.x_coarse=linspace(-integrate_options.limit_2d,integrate_options.limit_2d,integrate_options.Nint_2d);
            precomp.weights=exp(-0.5*precomp.x.^2);
            precomp.weights=precomp.weights/sum(precomp.weights);
            
            [x,y]=meshgrid(precomp.x_coarse,precomp.x_coarse);
            precomp.x_2d=[x(:)';y(:)'];
            precomp.weights_2d=exp(-0.5*sum(precomp.x_2d.^2,1));
            precomp.weights_2d=precomp.weights_2d/sum(precomp.weights_2d(:));
        case 'frozen_samples'
            precomp.x=randn(1,integrate_options.Nint_1d);
            precomp.weights=ones(size(precomp.x))/integrate_options.Nint_1d;
            precomp.x_2d=randn(2,integrate_options.Nint_2d^2);
            precomp.weights_2d=ones(1,integrate_options.Nint_2d^2)/integrate_options.Nint_2d^2;
            
        case 'prob_spaced'
            error('not implemented yet');
        case 'hermite'
            error('not implemented yet')
            [precomp.x,precomp.weights,precomp.b,precomp.polyc]=gausshe(10);
    end
else
    precomp=[];
end

precomp.integrate_options=integrate_options;
precomp.optim_options=optim_options;





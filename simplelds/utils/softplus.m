function [f,df,ddf] = softplus(x)
% [f,df,ddf] = softplus(x)
%
% Also known as the soft-rectification function
%
% Computes the function:
%    f(x) = log(1+exp(x))
% and its first and second derivatives

switch nargout
    case 1  % ----------------------------- %
        f = log(1+exp(x));
        
        if any(x(:)<-20) % check for small values to avoid underflow
            iix = (x(:)<-20);
            f(iix) = exp(x(iix));
        end
        if any(x(:)>500) % Check for large values to avoid overflow errors
            iix = (x(:)>500);
            f(iix) = x(iix);
        end
        
    case 2  % ----------------------------- %
        f = log(1+exp(x));
        df = exp(x)./(1+exp(x));
        
        if any(x(:)<-20) % check for small values to avoid underflow
            iix = (x(:)<-20);
            f(iix) = exp(x(iix));
            df(iix) = f(iix);
        end
        if any(x(:)>500) % Check for large values to avoid overflow errors
            iix = (x(:)>500);
            f(iix) = x(iix);
            df(iix) = 1;
        end
        
    case 3 % ----------------------------- %
        f = log(1+exp(x));
        df = exp(x)./(1+exp(x));
        ddf = exp(x)./(1+exp(x)).^2;
        
        if any(x(:)<-20) % check for small values to avoid underflow
            iix = (x(:)<-20);
            f(iix) = exp(x(iix));
            df(iix) = f(iix);
            ddf(iix) = f(iix);
        end
        if any(x(:)>500) % Check for large values to avoid overflow errors
            iix = (x(:)>500);
            f(iix) = x(iix);
            df(iix) = 1;
            ddf(iix) = 0;
        end
end


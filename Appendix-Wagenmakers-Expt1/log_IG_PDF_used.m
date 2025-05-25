function log_density = log_IG_PDF_used(X,a,b)

% Purpose: 
% Density (p.d.f) of inverted gamma distribution
% -----------------------------------
% Density:
% f(x) = 1/(G(a) * b^a) * x^(a-1) * exp(-x/b)
% E(X) = a*b, Var(X) = a*b^2
% -----------------------------------
% Usage:
% X = points of evaluation
% a = degree of freedom parameter (shape parameter) 
% b = scale parameter
% -----------------------------------
% Returns:
% density = density evaluated at points X
% -----------------------------------
% Notes:
% If at least one of the X, a, b is vector/matrix
% It will return a vector/matrix density with conformable size.
%
% Written by Hang Qian, Iowa State University
% Contact me:  matlabist@gmail.com

if any(a<0) || any(b<0)
    error('Parameter a, b must be positive')
end

log_density = -gammaln(a)+a*log(b)-(a+1)*log(X)-(b./X);
density = exp(log_density);
density(X<0)=0;
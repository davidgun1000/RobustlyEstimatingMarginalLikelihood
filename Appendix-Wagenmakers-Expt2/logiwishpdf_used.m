function [logpdf] = logiwishpdf_used(IW,T,v,dim)

log1=-0.5*trace(T/IW);
log2=log(det(T))*(v/2);
log3=((v*dim)/2)*log(2);
log4=log(pi)*((dim*(dim-1))/4);
log5=((v+dim+1)/2)*log(det(IW));

const=0:dim-1;
dkk=(v-const)/2;
gamln=gammaln(dkk);
sumgamln=sum(gamln);

logpdf=log1+log2-(log3+log4+log5+sumgamln);
end

% [k,k2] = size(IW) ;
% 
% logexpterm = -.5*trace(S/IW) ;
% logdetIWterm = log(det(IW))*(d/2) ;
% logdetSterm = log(det(S))*((d-k-1)/2) ;
% logtwoterm = log(2)*((d-k-1)*k/2) ;
% logpiterm = log(pi)*((k-1)*k/4) ;
% 
% klst = 1:k ;
% dkk2 = (d-k-klst)/2 ;
% gamln = gammaln(dkk2) ;
% sumgamln = sum(gamln) ;
% 
% logpdf = logexpterm + logdetSterm - ...
%          (logdetIWterm + logtwoterm + logpiterm + sumgamln  ) ;
% 
% pdf = exp(logpdf) ;

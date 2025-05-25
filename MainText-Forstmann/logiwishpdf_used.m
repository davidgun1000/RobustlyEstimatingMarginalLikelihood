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


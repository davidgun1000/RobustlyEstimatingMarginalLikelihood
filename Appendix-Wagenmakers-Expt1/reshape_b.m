function [rnorm_b_kron]=reshape_b(data_cond_repmat,rnorm_b1_kron,rnorm_b2_kron)
    %match the threshold with the conditions
    ind1=data_cond_repmat==1;
    ind2=data_cond_repmat==2;
    
    rnorm_b_kron(ind1,:)=rnorm_b1_kron(ind1,1);
    rnorm_b_kron(ind2,:)=rnorm_b2_kron(ind2,1);
    

end
function [rnorm_v_kron]=reshape_v(data_response_repmat,data_cond_repmat,rnorm_v11_kron,rnorm_v12_kron,rnorm_v21_kron,rnorm_v22_kron)

     %match the mean of the drift rate with responses and conditions
     ind1=data_response_repmat==1 & data_cond_repmat==1;
     ind2=data_response_repmat==1 & data_cond_repmat==2;
     ind3=data_response_repmat==2 & data_cond_repmat==1;
     ind4=data_response_repmat==2 & data_cond_repmat==2;     
     
     rnorm_v_kron(ind1,:)=[rnorm_v11_kron(ind1,1),rnorm_v21_kron(ind1,1)];
     rnorm_v_kron(ind2,:)=[rnorm_v12_kron(ind2,1),rnorm_v22_kron(ind2,1)];
     rnorm_v_kron(ind3,:)=[rnorm_v21_kron(ind3,1),rnorm_v11_kron(ind3,1)];
     rnorm_v_kron(ind4,:)=[rnorm_v22_kron(ind4,1),rnorm_v12_kron(ind4,1)];
     
     
      

end
function [rnorm_b_kron]=reshape_b(data_cond_repmat,data_stim_repmat,rnorm_b11_kron,rnorm_b12_kron,rnorm_b21_kron,rnorm_b22_kron)
    %match the threshold with the stimulus and conditions
    ind11=data_cond_repmat==1 & data_stim_repmat==1;
    ind12=data_cond_repmat==1 & data_stim_repmat==2;
    ind21=data_cond_repmat==2 & data_stim_repmat==1;
    ind22=data_cond_repmat==2 & data_stim_repmat==2;
    rnorm_b_kron(ind11,:)=[rnorm_b11_kron(ind11,1),rnorm_b12_kron(ind11,1)];
    rnorm_b_kron(ind12,:)=[rnorm_b12_kron(ind12,1),rnorm_b11_kron(ind12,1)];
    rnorm_b_kron(ind21,:)=[rnorm_b21_kron(ind21,1),rnorm_b22_kron(ind21,1)];
    rnorm_b_kron(ind22,:)=[rnorm_b22_kron(ind22,1),rnorm_b21_kron(ind22,1)];


    

end
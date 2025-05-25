function [llh,var_llh]=LBA_MC_IS_real_log_prop_usedmixchol(data,param,...
num_subjects,num_trials,num_particles,mean_theta,covmat_theta)

parfor j=1:num_subjects
    w_mix=0.95;
    u=rand(num_particles,1);
    id1=(u<w_mix);
    id2=1-id1;
    n1=sum(id1);
    n2=num_particles-n1;
    chol_theta_sig2=chol(param.theta_sig2,'lower');
    chol_theta_sig2_1=log(chol_theta_sig2(1,1));
    chol_theta_sig2_2=[chol_theta_sig2(2,1),log(chol_theta_sig2(2,2))];
    chol_theta_sig2_3=[chol_theta_sig2(3,1:2),log(chol_theta_sig2(3,3))];
    chol_theta_sig2_4=[chol_theta_sig2(4,1:3),log(chol_theta_sig2(4,4))];
    chol_theta_sig2_5=[chol_theta_sig2(5,1:4),log(chol_theta_sig2(5,5))];
    chol_theta_sig2_6=[chol_theta_sig2(6,1:5),log(chol_theta_sig2(6,6))];
    chol_theta_sig2_7=[chol_theta_sig2(7,1:6),log(chol_theta_sig2(7,7))];
    chol_theta_sig2_8=[chol_theta_sig2(8,1:7),log(chol_theta_sig2(8,8))];
    xx=[param.theta_mu';chol_theta_sig2_1';chol_theta_sig2_2';chol_theta_sig2_3';...
        chol_theta_sig2_4';chol_theta_sig2_5';chol_theta_sig2_6';chol_theta_sig2_7';chol_theta_sig2_8'];
    cond_mean=mean_theta(j,1:8)'+covmat_theta(1:8,9:end,j)*((covmat_theta(9:end,9:end,j))\(xx-mean_theta(j,9:end)'));
    cond_var=covmat_theta(1:8,1:8,j)-covmat_theta(1:8,9:end,j)*(covmat_theta(9:end,9:end,j)\covmat_theta(9:end,1:8,j));
    cond_var=topdm(cond_var);
    chol_cond_var=chol(cond_var,'lower');
    rnorm1=cond_mean+chol_cond_var*randn(param.num_randeffect,n1);
    chol_covmat=chol(param.theta_sig2,'lower');
    rnorm2=param.theta_mu'+chol_covmat*randn(param.num_randeffect,n2);
    rnorm=[rnorm1,rnorm2];
    rnorm=rnorm';
    
    rnorm_theta_b1=rnorm(:,1);
    rnorm_theta_b2=rnorm(:,2);
    rnorm_theta_A=rnorm(:,3);
    rnorm_theta_v11=rnorm(:,4);
    rnorm_theta_v12=rnorm(:,5);
    
    rnorm_theta_v21=rnorm(:,6);
    rnorm_theta_v22=rnorm(:,7);
    rnorm_theta_tau=rnorm(:,8);

    rnorm_theta_b1_kron=kron(rnorm_theta_b1,ones(num_trials(j,1),1));
    rnorm_theta_b2_kron=kron(rnorm_theta_b2,ones(num_trials(j,1),1));
    rnorm_theta_A_kron=kron(rnorm_theta_A,ones(num_trials(j,1),1));
    rnorm_theta_v11_kron=kron(rnorm_theta_v11,ones(num_trials(j,1),1));
    rnorm_theta_v12_kron=kron(rnorm_theta_v12,ones(num_trials(j,1),1));
    
    rnorm_theta_v21_kron=kron(rnorm_theta_v21,ones(num_trials(j,1),1));
    rnorm_theta_v22_kron=kron(rnorm_theta_v22,ones(num_trials(j,1),1));
    rnorm_theta_tau_kron=kron(rnorm_theta_tau,ones(num_trials(j,1),1));
    
    data_response_repmat=repmat(data.response{j,1}(:,1),num_particles,1);
    data_rt_repmat=repmat(data.rt{j,1}(:,1),num_particles,1);
    data_cond_repmat=repmat(data.cond{j,1}(:,1),num_particles,1);
    [rnorm_theta_b_kron]=reshape_b(data_cond_repmat,rnorm_theta_b1_kron,rnorm_theta_b2_kron);
    [rnorm_theta_v_kron]=reshape_v(data_response_repmat,data_cond_repmat,rnorm_theta_v11_kron,rnorm_theta_v12_kron,rnorm_theta_v21_kron,rnorm_theta_v22_kron);
    lw=real(log(LBA_n1PDF_reparam_real(data_rt_repmat, rnorm_theta_A_kron, rnorm_theta_b_kron, rnorm_theta_v_kron, ones(num_particles*num_trials(j,1),1),rnorm_theta_tau_kron)));
    
    lw_reshape=reshape(lw,num_trials(j,1),num_particles);
    logw_first=sum(lw_reshape);
    
    logw_second=(logmvnpdf([rnorm_theta_b1,rnorm_theta_b2,rnorm_theta_A,...
         rnorm_theta_v11,rnorm_theta_v12,rnorm_theta_v21,rnorm_theta_v22,rnorm_theta_tau],param.theta_mu,param.theta_sig2));
    logw_third=log(w_mix.*mvnpdf([rnorm_theta_b1,rnorm_theta_b2,rnorm_theta_A,...
         rnorm_theta_v11,rnorm_theta_v12,rnorm_theta_v21,rnorm_theta_v22,rnorm_theta_tau],cond_mean',chol_cond_var*chol_cond_var')+...
         (1-w_mix).*mvnpdf([rnorm_theta_b1,rnorm_theta_b2,rnorm_theta_A,...
         rnorm_theta_v11,rnorm_theta_v12,rnorm_theta_v21,rnorm_theta_v22,rnorm_theta_tau],param.theta_mu,param.theta_sig2));
    logw=logw_first'+logw_second'-logw_third;
    
    id=imag(logw)~=0;
    id=1-id;
    id=logical(id);
    logw=logw(id,1); 
    logw=real(logw);

    if sum(isinf(logw))>0 | sum(isnan(logw))>0
     id=isinf(logw) | isnan(logw);
     id=1-id;
     id=logical(id);
     logw=logw(id,1);
    end
    
    max_logw=max(real(logw));
    var_z_i = sum(exp(2*(logw-max_logw)))/(sum(exp(logw-max_logw)))^2-1/num_particles;
    weight=real(exp(logw-max_logw));
    llh_i(j) = max_logw+log(mean(weight)); 
    llh_i(j) = real(llh_i(j));
    var_llh_i(j) = var_z_i;
    weight=weight./sum(weight);
    if sum(weight<0)>0
        id=weight<0;
        id=1-id;
        id=logical(id);
        weight=weight(id,1);
    end
    Nw=length(weight);
    
    if Nw>0 
        ind=randsample(Nw,1,true,weight);
        theta_latent_b1(j,1)=rnorm_theta_b1(ind,1);
        theta_latent_b2(j,1)=rnorm_theta_b2(ind,1);
        theta_latent_A(j,1)=rnorm_theta_A(ind,1);
        theta_latent_v11(j,1)=rnorm_theta_v11(ind,1);
        theta_latent_v12(j,1)=rnorm_theta_v12(ind,1);
        theta_latent_v21(j,1)=rnorm_theta_v21(ind,1);
        theta_latent_v22(j,1)=rnorm_theta_v22(ind,1);
        theta_latent_tau(j,1)=rnorm_theta_tau(ind,1);
    end
        
%----------------------------------------------------------------------------------------------------------------------------------    
    
end
llh=sum(llh_i);
var_llh=sum(var_llh_i);
end



function [llh,var_llh]=LBA_MC_IS2_v1(data,param,...
num_subjects,num_trials,num_particles,mean_theta,covmat_theta,num_randeffect)
%this function provides an estimate of log of estimated likelihood of a LBA
%model

var_llh_opt=1; %the optimal variance of the log of estimated likelihood of the LBA model
var_llh_opt_i=var_llh_opt/num_subjects; %the optimal variance per subject of the log of the estimated likelihood of the LBA model
num_particles_inc=250;% we use adaptive strategy to choose the number of particles used to estimate the log of estimated likelihood. We start with 250 particles, and computing the variance of log of
%estimated likelihood, if the variance was greater than 1, we then increase
%the number of particles.
%we can estimate the log-likelihood contribution of each subject in
%parallel in multiple core machines which is very efficient.
parfor j=1:num_subjects
    N_i=num_particles;
    w_mix=0.95; %set the mixture weights
    %generate the random effects particles from the proposals outlined in
    %the paper Tran et al (2019)
    %-------------------------------------------
    u=rand(N_i,1);
    id1=(u<w_mix);
    id2=1-id1;
    n1=sum(id1);
    n2=N_i-n1;
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
    %--------------------------------------------
    rnorm_theta_b1=rnorm(:,1);
    rnorm_theta_b2=rnorm(:,2);
    rnorm_theta_A=rnorm(:,3);
    rnorm_theta_v11=rnorm(:,4);
    rnorm_theta_v12=rnorm(:,5);
    rnorm_theta_v21=rnorm(:,6);
    rnorm_theta_v22=rnorm(:,7);
    rnorm_theta_tau=rnorm(:,8);
    
    rnorm_theta=[rnorm_theta_b1,rnorm_theta_b2,rnorm_theta_A,rnorm_theta_v11,rnorm_theta_v12,rnorm_theta_v21,rnorm_theta_v22,rnorm_theta_tau];
    
    
    %adjust the size of the vector of each random effect particles
    rnorm_theta_b1_kron=kron(rnorm_theta_b1,ones(num_trials(j,1),1));
    rnorm_theta_b2_kron=kron(rnorm_theta_b2,ones(num_trials(j,1),1));
    rnorm_theta_A_kron=kron(rnorm_theta_A,ones(num_trials(j,1),1));
    rnorm_theta_v11_kron=kron(rnorm_theta_v11,ones(num_trials(j,1),1));
    rnorm_theta_v12_kron=kron(rnorm_theta_v12,ones(num_trials(j,1),1));
    
    rnorm_theta_v21_kron=kron(rnorm_theta_v21,ones(num_trials(j,1),1));
    rnorm_theta_v22_kron=kron(rnorm_theta_v22,ones(num_trials(j,1),1));
    rnorm_theta_tau_kron=kron(rnorm_theta_tau,ones(num_trials(j,1),1));
    
    %adjust the size of the dataset
    data_response_repmat=repmat(data.response{j,1}(:,1),N_i,1);
    data_rt_repmat=repmat(data.rt{j,1}(:,1),N_i,1);
    data_cond_repmat=repmat(data.cond{j,1}(:,1),N_i,1);
    [rnorm_theta_b_kron]=reshape_b(data_cond_repmat,rnorm_theta_b1_kron,rnorm_theta_b2_kron); %choose the threshold particles to match with the conditions of the experiment
    [rnorm_theta_v_kron]=reshape_v(data_response_repmat,data_cond_repmat,rnorm_theta_v11_kron,rnorm_theta_v12_kron,rnorm_theta_v21_kron,rnorm_theta_v22_kron); %set the drift rate particles to match with the responses and the conditions
    %---------------------------------
    %computing the log of the weights
    
    %computing the log density of the LBA given the particles of random
    %effects
    %------------------------------
    lw=real(log(LBA_n1PDF_reparam_real(data_rt_repmat, rnorm_theta_A_kron, rnorm_theta_b_kron, rnorm_theta_v_kron, ones(N_i*num_trials(j,1),1),rnorm_theta_tau_kron)));
    lw_reshape=reshape(lw,num_trials(j,1),N_i);
    logw_first=sum(lw_reshape);
    %------------------------------
    logw_second=(logmvnpdf(rnorm_theta,param.theta_mu,param.theta_sig2)); %computing the log of prior density p(\alpha|\theta) at each of the proposal density
    logw_third=log(w_mix.*mvnpdf(rnorm_theta,cond_mean',chol_cond_var*chol_cond_var')+...
         (1-w_mix).*mvnpdf(rnorm_theta,param.theta_mu,param.theta_sig2)); % computing the log of proposal density at each of the proposal particles
    logw=logw_first'+logw_second'-logw_third; %compute the log of weights
     %---------------------------------------------
    max_logw=max(real(logw));
    var_z_i = sum(exp(2*(logw-max_logw)))/(sum(exp(logw-max_logw)))^2-1/N_i; %compute the variance of log of estimated likelihood for subject j
     %if the var_z_i>var_llh_opt_i, then we need to increase the particle by
    %another 250 particles.
    while (var_z_i>var_llh_opt_i) & (N_i<10000)
        %generate the random effects particles from the proposals outlined in
        %the paper Tran et al (2019)
        %-------------------------------------------
        u=rand(num_particles_inc,1);
        id1=(u<w_mix);
        id2=1-id1;
        n1=sum(id1);
        n2=num_particles_inc-n1;
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
        %--------------------------------------------
        rnorm_theta_b1=rnorm(:,1);
        rnorm_theta_b2=rnorm(:,2);
        rnorm_theta_A=rnorm(:,3);
        rnorm_theta_v11=rnorm(:,4);
        rnorm_theta_v12=rnorm(:,5);
        rnorm_theta_v21=rnorm(:,6);
        rnorm_theta_v22=rnorm(:,7);
        rnorm_theta_tau=rnorm(:,8);
        rnorm_theta=[rnorm_theta_b1,rnorm_theta_b2,rnorm_theta_A,rnorm_theta_v11,rnorm_theta_v12,rnorm_theta_v21,rnorm_theta_v22,rnorm_theta_tau];
        
        %adjust the size of the vector of each random effect particles
        

        rnorm_theta_b1_kron=kron(rnorm_theta_b1,ones(num_trials(j,1),1));
        rnorm_theta_b2_kron=kron(rnorm_theta_b2,ones(num_trials(j,1),1));
        rnorm_theta_A_kron=kron(rnorm_theta_A,ones(num_trials(j,1),1));
        rnorm_theta_v11_kron=kron(rnorm_theta_v11,ones(num_trials(j,1),1));
        rnorm_theta_v12_kron=kron(rnorm_theta_v12,ones(num_trials(j,1),1));
        rnorm_theta_v21_kron=kron(rnorm_theta_v21,ones(num_trials(j,1),1));
        rnorm_theta_v22_kron=kron(rnorm_theta_v22,ones(num_trials(j,1),1));
        rnorm_theta_tau_kron=kron(rnorm_theta_tau,ones(num_trials(j,1),1));
        %adjust the size of the dataset

        data_response_repmat=repmat(data.response{j,1}(:,1),num_particles_inc,1);
        data_rt_repmat=repmat(data.rt{j,1}(:,1),num_particles_inc,1);
        data_cond_repmat=repmat(data.cond{j,1}(:,1),num_particles_inc,1);
        [rnorm_theta_b_kron]=reshape_b(data_cond_repmat,rnorm_theta_b1_kron,rnorm_theta_b2_kron); %choose the threshold particles to match with the conditions of the experiment
        [rnorm_theta_v_kron]=reshape_v(data_response_repmat,data_cond_repmat,rnorm_theta_v11_kron,rnorm_theta_v12_kron,rnorm_theta_v21_kron,rnorm_theta_v22_kron); %set the drift rate particles to match with the response and conditions
        %computing the log of the weights
        %----------------------------------------
        lw=real(log(LBA_n1PDF_reparam_real(data_rt_repmat, rnorm_theta_A_kron, rnorm_theta_b_kron, rnorm_theta_v_kron, ones(num_particles_inc*num_trials(j,1),1),rnorm_theta_tau_kron)));

        lw_reshape=reshape(lw,num_trials(j,1),num_particles_inc);
        logw_first=sum(lw_reshape);

        logw_second=(logmvnpdf(rnorm_theta,param.theta_mu,param.theta_sig2));
        logw_third=log(w_mix.*mvnpdf(rnorm_theta,cond_mean',chol_cond_var*chol_cond_var')+...
             (1-w_mix).*mvnpdf(rnorm_theta,param.theta_mu,param.theta_sig2));
        logw_temp=logw_first'+logw_second'-logw_third;
        %----------------------------------------
        logw=[logw;logw_temp];
        N_i=N_i+num_particles_inc;
    
        max_logw=max(real(logw));
        var_z_i = sum(exp(2*(logw-max_logw)))/(sum(exp(logw-max_logw)))^2-1/N_i;
    end
    
    
    weight=real(exp(logw-max_logw));
    llh_i(j) = max_logw+log(mean(weight));  % obtain the log of estimated likelihood contribution for subject j
    llh_i(j) = real(llh_i(j));
    var_llh_i(j) = var_z_i; %obtain the variance of log of estimated likelihood contribution for subject j
    %----------------------------------------------------------------------------------------------------------------------------------    

    
        
%----------------------------------------------------------------------------------------------------------------------------------    
    
end
llh=sum(llh_i); %the value of the log of estimated likelihood
var_llh=sum(var_llh_i); %the total variance of the log of estimated likelihood
end

%end



function [logw]=compute_logw(prop_theta,data,num_subjects,num_trials,num_particles,num_randeffect,mean_thetalatent,covmat_thetalatent,mixprop_mean,mixprop_Sigma,mixprop_weight,i)
    %this function is to compute the log of importance weights
    v_alpha=2;

    
    param.num_randeffect=8;
    param.theta_mu=[prop_theta(i,1:8)]; %the parameter proposal for \mu_{\alpha}
    chol_theta_sig2_temp=[prop_theta(i,9:44)]'; %the parameter proposal for the cholesky factor of covariance matrix \Sigma_{\alpha}
    %---------------------------------------------------------------
    %reconstruct the proposal of \Sigma_{\alpha} from the proposal of
    %cholesky factorisation of \Sigma_{\alpha}
    chol_theta_sig2_temp(1,1)=exp(chol_theta_sig2_temp(1,1));
    chol_theta_sig2_temp(3,1)=exp(chol_theta_sig2_temp(3,1));
    chol_theta_sig2_temp(6,1)=exp(chol_theta_sig2_temp(6,1));
    chol_theta_sig2_temp(10,1)=exp(chol_theta_sig2_temp(10,1));
    chol_theta_sig2_temp(15,1)=exp(chol_theta_sig2_temp(15,1));
    chol_theta_sig2_temp(21,1)=exp(chol_theta_sig2_temp(21,1));
    chol_theta_sig2_temp(28,1)=exp(chol_theta_sig2_temp(28,1));
    chol_theta_sig2_temp(36,1)=exp(chol_theta_sig2_temp(36,1));
    
    chol_theta_sig2(1,1)=chol_theta_sig2_temp(1,1);
    chol_theta_sig2(2,1:2)=[chol_theta_sig2_temp(2,1),chol_theta_sig2_temp(3,1)];
    chol_theta_sig2(3,1:3)=[chol_theta_sig2_temp(4,1),chol_theta_sig2_temp(5,1),chol_theta_sig2_temp(6,1)];
    chol_theta_sig2(4,1:4)=[chol_theta_sig2_temp(7,1),chol_theta_sig2_temp(8,1),chol_theta_sig2_temp(9,1),chol_theta_sig2_temp(10,1)];
    chol_theta_sig2(5,1:5)=[chol_theta_sig2_temp(11,1),chol_theta_sig2_temp(12,1),chol_theta_sig2_temp(13,1),chol_theta_sig2_temp(14,1),chol_theta_sig2_temp(15,1)];
    chol_theta_sig2(6,1:6)=[chol_theta_sig2_temp(16,1),chol_theta_sig2_temp(17,1),chol_theta_sig2_temp(18,1),chol_theta_sig2_temp(19,1),chol_theta_sig2_temp(20,1),chol_theta_sig2_temp(21,1)];
    chol_theta_sig2(7,1:7)=[chol_theta_sig2_temp(22,1),chol_theta_sig2_temp(23,1),chol_theta_sig2_temp(24,1),chol_theta_sig2_temp(25,1),chol_theta_sig2_temp(26,1),chol_theta_sig2_temp(27,1),chol_theta_sig2_temp(28,1)];
    chol_theta_sig2(8,1:8)=[chol_theta_sig2_temp(29,1),chol_theta_sig2_temp(30,1),chol_theta_sig2_temp(31,1),chol_theta_sig2_temp(32,1),chol_theta_sig2_temp(33,1),chol_theta_sig2_temp(34,1),chol_theta_sig2_temp(35,1),chol_theta_sig2_temp(36,1)];
    
    param.theta_sig2=chol_theta_sig2*chol_theta_sig2'; %the parameter proposal for \Sigma_{\alpha}
    param.a1=exp(prop_theta(i,45)); %the parameter proposal for a_{1}
    param.a2=exp(prop_theta(i,46));%the parameter proposal for a_{2}
    param.a3=exp(prop_theta(i,47));%the parameter proposal for a_{3}
    param.a4=exp(prop_theta(i,48));%the parameter proposal for a_{4}
    param.a5=exp(prop_theta(i,49));%the parameter proposal for a_{5}
    param.a6=exp(prop_theta(i,50));%the parameter proposal for a_{6}
    param.a7=exp(prop_theta(i,51));%the parameter proposal for a_{7}
    param.a8=exp(prop_theta(i,52));%the parameter proposal for a_{8}
    
    [logp,var_logp]=LBA_MC_IS2_v1(data,param,num_subjects,num_trials,num_particles,mean_thetalatent,covmat_thetalatent,num_randeffect); % compute the log of estimated likelihood of the hierarchical LBA model    
    log_prior_mu=logmvnpdf(param.theta_mu,zeros(1,num_randeffect),eye(num_randeffect));  % compute the log of the prior for \mu_{\alpha}
    log_prior_sigma=logiwishpdf_used(param.theta_sig2,2*v_alpha*diag([1/param.a1;1/param.a2;1/param.a3;1/param.a4;1/param.a5;1/param.a6;1/param.a7;1/param.a8]),v_alpha+num_randeffect-1,num_randeffect);% compute the log of the prior for \Sigma_{\alpha}
    log_prior_a=log_IG_PDF_used(param.a1,0.5,1)+log_IG_PDF_used(param.a2,0.5,1)+log_IG_PDF_used(param.a3,0.5,1)+log_IG_PDF_used(param.a4,0.5,1)+...
        log_IG_PDF_used(param.a5,0.5,1)+log_IG_PDF_used(param.a6,0.5,1)+log_IG_PDF_used(param.a7,0.5,1)+log_IG_PDF_used(param.a8,0.5,1);% %compute the log prior of a_1,...,a_7
    %compute the log of importance weights
    %------------------------------------------
    logw_num=logp+log_prior_mu+log_prior_sigma+log_prior_a;
    logw_den1=log(mixprop_weight(1,1).*mvnpdf(prop_theta(i,:),mixprop_mean(1,:),mixprop_Sigma(:,:,1))+...
        (1-mixprop_weight(1,1)).*mvnpdf(prop_theta(i,:),mixprop_mean(2,:),mixprop_Sigma(:,:,2)));
    logw_den2=log(1/param.a1)+log(1/param.a2)+log(1/param.a3)+log(1/param.a4)+log(1/param.a5)+log(1/param.a6)+log(1/param.a7)+log(1/param.a8);
    logw_den3=log(2^num_randeffect)+(num_randeffect-1+2)*log(chol_theta_sig2_temp(1,1))+(num_randeffect-2+2)*log(chol_theta_sig2_temp(3,1))+...
        (num_randeffect-3+2)*log(chol_theta_sig2_temp(6,1))+(num_randeffect-4+2)*log(chol_theta_sig2_temp(10,1))+(num_randeffect-5+2)*log(chol_theta_sig2_temp(15,1))+...
        (num_randeffect-6+2)*log(chol_theta_sig2_temp(21,1))+(num_randeffect-7+2)*log(chol_theta_sig2_temp(28,1))+(num_randeffect-8+2)*log(chol_theta_sig2_temp(36,1));
    logw_den=logw_den1+logw_den2-logw_den3;
    logw=logw_num-logw_den;

 %--------------------------------------------
end
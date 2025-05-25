%obtain proposal 

% num_subjects=19;
% theta_sig2_1_temp=[log(theta_sig2_store1_1(:,1)),theta_sig2_store1_1(:,2:end)];
% theta_sig2_2_temp=[log(theta_sig2_store2_1(:,1)),theta_sig2_store2_1(:,2:end)];
% theta_sig2_3_temp=[log(theta_sig2_store3_1(:,1)),theta_sig2_store3_1(:,2:end)];
% theta_sig2_4_temp=[log(theta_sig2_store4_1(:,1)),theta_sig2_store4_1(:,2:end)];
% theta_sig2_5_temp=[log(theta_sig2_store5_1(:,1)),theta_sig2_store5_1(:,2:end)];
% theta_sig2_6_temp=[log(theta_sig2_store6_1(:,1)),theta_sig2_store6_1(:,2:end)];
% theta_sig2_7_temp=[log(theta_sig2_store7_1(:,1))];
% 
% for j=1:num_subjects
%     theta=[theta_latent_b1_store1(10000:13000,j),theta_latent_b2_store1(10000:13000,j),theta_latent_b3_store1(10000:13000,j),...
%         theta_latent_A_store1(10000:13000,j),theta_latent_v1_store1(10000:13000,j),theta_latent_v2_store1(10000:13000,j),theta_latent_tau_store1(10000:13000,j),...
%         theta_mu_store1(10000:13000,:),theta_sig2_1_temp(10000:13000,:),theta_sig2_2_temp(10000:13000,:),theta_sig2_3_temp(10000:13000,:),...
%         theta_sig2_4_temp(10000:13000,:),theta_sig2_5_temp(10000:13000,:),theta_sig2_6_temp(10000:13000,:),theta_sig2_7_temp(10000:13000,:)];
%     covmat_theta(:,:,j)=cov(theta);
%     mean_theta(j,:)=mean(theta);
% end
% save('proposal_sim_log.mat','mean_theta','covmat_theta');

%obtain proposal cholesky

theta_latent_A_temp=theta_latent_A_store(1:end,:);
theta_latent_b1_temp=theta_latent_b1_store(1:end,:);
theta_latent_b2_temp=theta_latent_b2_store(1:end,:);
theta_latent_b3_temp=theta_latent_b3_store(1:end,:);
theta_latent_v1_temp=theta_latent_v1_store(1:end,:);
theta_latent_v2_temp=theta_latent_v2_store(1:end,:);
theta_latent_tau_temp=theta_latent_tau_store(1:end,:);

theta_mu_temp=theta_mu_store(1:end,:);
theta_sig2_1_temp=theta_sig2_store1(1:end,:);
theta_sig2_2_temp=theta_sig2_store2(1:end,:);
theta_sig2_3_temp=theta_sig2_store3(1:end,:);
theta_sig2_4_temp=theta_sig2_store4(1:end,:);
theta_sig2_5_temp=theta_sig2_store5(1:end,:);
theta_sig2_6_temp=theta_sig2_store6(1:end,:);
theta_sig2_7_temp=theta_sig2_store7(1:end,:);

for i=1:10000
    sigma_temp=[theta_sig2_1_temp(i,:);
        theta_sig2_1_temp(i,2),theta_sig2_2_temp(i,:);
        theta_sig2_1_temp(i,3),theta_sig2_2_temp(i,2),theta_sig2_3_temp(i,:);
        theta_sig2_1_temp(i,4),theta_sig2_2_temp(i,3),theta_sig2_3_temp(i,2),theta_sig2_4_temp(i,:);
        theta_sig2_1_temp(i,5),theta_sig2_2_temp(i,4),theta_sig2_3_temp(i,3),theta_sig2_4_temp(i,2),theta_sig2_5_temp(i,:);
        theta_sig2_1_temp(i,6),theta_sig2_2_temp(i,5),theta_sig2_3_temp(i,4),theta_sig2_4_temp(i,3),theta_sig2_5_temp(i,2),theta_sig2_6_temp(i,:);
        theta_sig2_1_temp(i,7),theta_sig2_2_temp(i,6),theta_sig2_3_temp(i,5),theta_sig2_4_temp(i,4),theta_sig2_5_temp(i,3),theta_sig2_6_temp(i,2),theta_sig2_7_temp(i,:)];
    
    chol_sigma_temp=chol(sigma_temp,'lower');
    chol_theta_sig2_1(i,:)=[log(chol_sigma_temp(1,1))];
    chol_theta_sig2_2(i,:)=[chol_sigma_temp(2,1),log(chol_sigma_temp(2,2))];
    chol_theta_sig2_3(i,:)=[chol_sigma_temp(3,1:2),log(chol_sigma_temp(3,3))];
    chol_theta_sig2_4(i,:)=[chol_sigma_temp(4,1:3),log(chol_sigma_temp(4,4))];
    chol_theta_sig2_5(i,:)=[chol_sigma_temp(5,1:4),log(chol_sigma_temp(5,5))];
    chol_theta_sig2_6(i,:)=[chol_sigma_temp(6,1:5),log(chol_sigma_temp(6,6))];
    chol_theta_sig2_7(i,:)=[chol_sigma_temp(7,1:6),log(chol_sigma_temp(7,7))];
    
end

%proposal for latent
for j=1:num_subjects
    theta_latent=[theta_latent_b1_temp(:,j),theta_latent_b2_temp(:,j),theta_latent_b3_temp(:,j),...
        theta_latent_A_temp(:,j),theta_latent_v1_temp(:,j),theta_latent_v2_temp(:,j),theta_latent_tau_temp(:,j),...
        theta_mu_temp,chol_theta_sig2_1,chol_theta_sig2_2,chol_theta_sig2_3,chol_theta_sig2_4,chol_theta_sig2_5,chol_theta_sig2_6,chol_theta_sig2_7];
    covmat_thetalatent(:,:,j)=cov(theta_latent);
    mean_thetalatent(j,:)=mean(theta_latent);
    
end

%proposal for parameters
theta_param=[theta_mu_temp,chol_theta_sig2_1,chol_theta_sig2_2,chol_theta_sig2_3,chol_theta_sig2_4,chol_theta_sig2_5,chol_theta_sig2_6,chol_theta_sig2_7];
covmat_thetaparam=cov(theta_param);
mean_thetaparam=mean(theta_param);

save('proposal_sim_IS2.mat','mean_thetalatent','covmat_thetalatent','mean_thetaparam','covmat_thetaparam');

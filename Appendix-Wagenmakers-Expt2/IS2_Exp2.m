%this IS^2 code takes an input of MCMC posterior draws of parameters
%\mu_{\alpha} and \Sigma_{\alpha} and the random effects for each subject
% The example of the input is given in the Matlab file 'LBA_Exp2.mat',
% To run this program, you need to make sure that you have appropriate
% input as in 'LBA_Exp2.mat'.
% The dataset is 'LBA_data_Exp2.mat', it has data.response = 1 if response
% is Word and response=2 if the response is Nonword, data.cond
% specifies the conditions of the experiments, data.rt contains the
% response times. 

num_randeffect=8; %number of random effects
load('LBA_Exp2.mat');   %load the posterior MCMC draws of random effects \alpha and parameters \mu_{\alpha} and \Sigma_{\alpha}
load('LBA_data_Exp2.mat'); %load the dataset
num_subjects=length(data.rt);
for j=1:num_subjects
     num_trials(j,1)=length(data.rt{j,1}); % computing the number of trials per subjects
end
num_particles=250; %number of particles in the Monte Carlo algorithm required to estimate the likelihood
v_alpha=2;
tic
%obtain proposal
length_draws=length(theta_latent_A_store); %compute the number of MCMC draws from the input file 'LBA_Exp2.mat'
%parpool(28) %the number of multi-cores.
%--------------------------------------------------------------------------
%In this code, we use only the last 10000 of the MCMC draws
theta_latent_A_temp=theta_latent_A_store(length_draws-9999:end,:);
theta_latent_b11_temp=theta_latent_b11_store(length_draws-9999:end,:);
theta_latent_b12_temp=theta_latent_b12_store(length_draws-9999:end,:);
theta_latent_b21_temp=theta_latent_b21_store(length_draws-9999:end,:);
theta_latent_b22_temp=theta_latent_b22_store(length_draws-9999:end,:);
theta_latent_v1_temp=theta_latent_v1_store(length_draws-9999:end,:);
theta_latent_v2_temp=theta_latent_v2_store(length_draws-9999:end,:);
theta_latent_tau_temp=theta_latent_tau_store(length_draws-9999:end,:);

theta_mu_temp=theta_mu_store(length_draws-9999:end,:);
chol_theta_sig2_1=chol_theta_sig2_store1(length_draws-9999:end,:);
chol_theta_sig2_2=chol_theta_sig2_store2(length_draws-9999:end,:);
chol_theta_sig2_3=chol_theta_sig2_store3(length_draws-9999:end,:);
chol_theta_sig2_4=chol_theta_sig2_store4(length_draws-9999:end,:);
chol_theta_sig2_5=chol_theta_sig2_store5(length_draws-9999:end,:);
chol_theta_sig2_6=chol_theta_sig2_store6(length_draws-9999:end,:);
chol_theta_sig2_7=chol_theta_sig2_store7(length_draws-9999:end,:);
chol_theta_sig2_8=chol_theta_sig2_store8(length_draws-9999:end,:);


a1_half=a1_half_store(length_draws-9999:end,1);
a2_half=a2_half_store(length_draws-9999:end,1);
a3_half=a3_half_store(length_draws-9999:end,1);
a4_half=a4_half_store(length_draws-9999:end,1);
a5_half=a5_half_store(length_draws-9999:end,1);
a6_half=a6_half_store(length_draws-9999:end,1);
a7_half=a7_half_store(length_draws-9999:end,1);
a8_half=a8_half_store(length_draws-9999:end,1);
%--------------------------------------------------------------------------

%Training the proposal for conditional Monte Carlo algorithm

for j=1:num_subjects
    % In the matrix called theta_latent below, you have to list (1) all
    % your random effects in the LBA model, in the case of
    % Forstmann, you have \alpha_{b_11}, \alpha_{b_12}, \alpha_{b_21}, \alpha_{b_22},
    % \alpha_{b_3}, \alpha_A, \alpha_{v_1}, \alpha_{v_2},
    % \alpha_{tau}, (2) followed by the parameters \mu_{\alpha}, and
    % cholesky factor (lower triangular matrix) of the covariance matrix
    % \Sigma_{\alpha}
    theta_latent=[theta_latent_b11_temp(:,j),theta_latent_b12_temp(:,j),theta_latent_b21_temp(:,j),theta_latent_b22_temp(:,j),...
        theta_latent_A_temp(:,j),theta_latent_v1_temp(:,j),theta_latent_v2_temp(:,j),theta_latent_tau_temp(:,j),...
        theta_mu_temp,chol_theta_sig2_1,chol_theta_sig2_2,chol_theta_sig2_3,chol_theta_sig2_4,chol_theta_sig2_5,chol_theta_sig2_6,...
        chol_theta_sig2_7,chol_theta_sig2_8];
    covmat_thetalatent(:,:,j)=cov(theta_latent);
    mean_thetalatent(j,:)=mean(theta_latent);
end

% fitting mixture of normal distribution to the posterior draws of parameters \mu_{\alpha} and
% cholesky factorisation of the covariance matrix \Sigma_{\alpha} using built in function in Matlab (fitgmdist).
%---------------------------------------
X=[theta_mu_temp,chol_theta_sig2_1,chol_theta_sig2_2,chol_theta_sig2_3,chol_theta_sig2_4,chol_theta_sig2_5,chol_theta_sig2_6,...
    chol_theta_sig2_7,chol_theta_sig2_8,log(a1_half),log(a2_half),log(a3_half),log(a4_half),log(a5_half),log(a6_half),...
    log(a7_half),log(a8_half)];
options = statset('MaxIter',5000);
k=2;
mixtwo=fitgmdist(X,k,'Options',options);
mixprop_weight=mixtwo.ComponentProportion;
mixprop_mean=mixtwo.mu;

for i=1:k
    mixprop_Sigma(:,:,i)=mixtwo.Sigma(:,:,i);
end
%generate the parameters from the mixture of normal proposals estimated
%above

IS_samples=10000; % the number of IS^2 samples
num_param=52; % the total number of parameters in the LBA model.
u=rand(IS_samples,1);
id1=(u<=mixprop_weight(1,1));
n1=sum(id1);
n2=IS_samples-n1;
chol_mixprop_Sigma1=chol(mixprop_Sigma(:,:,1),'lower');
prop_theta1=mixprop_mean(1,:)'+chol_mixprop_Sigma1*randn(num_param,n1);
chol_mixprop_Sigma2=chol(mixprop_Sigma(:,:,2),'lower');
prop_theta2=mixprop_mean(2,:)'+chol_mixprop_Sigma2*randn(num_param,n2);
prop_theta=[prop_theta1,prop_theta2];
prop_theta=prop_theta';

% computing the log weight of the IS^2 samples
parfor i=1:IS_samples
    i
    [logw(i,1)]=compute_logw(prop_theta,data,num_subjects,num_trials,num_particles,num_randeffect,mean_thetalatent,covmat_thetalatent,mixprop_mean,mixprop_Sigma,mixprop_weight,i);
end

 max_logw=max(real(logw));
 weight=real(exp(logw-max_logw));
 log_marglik = max_logw+log(mean(weight)); 

 %bootstrap method to get the Monte carlo standard error
 %----------------------------------------------------------------
 
 B=10000;
 for i=1:B
     i
     log_weight_boot=datasample(logw,IS_samples);
     max_logw_boot=max(real(log_weight_boot));
     weight_boot=real(exp(log_weight_boot-max_logw_boot));
     log_marglik_boot(i,1)=max_logw_boot+log(mean(weight_boot));
 end
toc
CPUtime=toc;
%save the output as the Matlab file

save('IS_prop_diffthreshold_twoaccum.mat','logw','log_marglik_boot','CPUtime');
 
 %the log of the marginal liekliehood estimated is
%2*log_marglik-mean(log_marglik_boot)
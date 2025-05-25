### Instructions for estimating the marginal likelihood via IS^2 for Forstmann et al. (2008) data ###

Before using the IS^2 method to estimate the marginal likelihood,
the model parameters are estimated with the Particle Metropolis within
Gibbs (PMwG) algorithm. For details on PMwG, please see: osf.io/5b4w3

The code below assumes posterior samples are already stored though
we provide files to also obtain the samples via PMwG.


* IS2_LBA_Forstmann.m : This is the primary file to run the method. Start here. This file loads subsidiary scripts as required including:
   -> LBA_realdata.mat : Data from Forstmann et al. (2008) in format expected by the algorithm.
   -> LBA_Forstmann.mat : Posterior samples for Forstmann et al. (2008) estimated by PMwG. This is assumed as the input for IS^2. 
   -> LBA_MC_IS2_v1.m : Monte Carlo algorithm for generating initial estimates of the random effects. In this example it's not required as the estimates are saved in LBA_Forstmann.mat

When applying IS^2 to different data sets, the following scripts will
require editing depending on the design of the new data set and the
model that is estimated.
* compute_logw.m : compute the log weight of the IS^2 generated samples. 
* obtain_proposal.m : generate proposals for IS^2. 

When applying PMwG to different data sets, the following scripts may
require editing depending on the design of the new data set and the
model that is estimated. This was also covered in the code at: osf.io/5b4w3
* LBA_n1PDF_reparam_real.m : LBA race equation for two-choice task. 
* LBA_tpdf.m : density function for a single accumulator.
* LBA_tcdf.m : distribution function for a single accumulator.
* reshape_v.m : assign appropriate drift rates (correct, error) to appropriate trials.
* reshape_b.m :  assign appropriate response thresholds (accuracy, neutral, speed) to appropriate trials. 

Subsidiary Matlab functions. These don't require editing for application to different data sets.
* logmvnpdf.m : log density of multivariate normal distribution.
* logiwishpdf_used.m : log density of inverse Wishart distribution.
* log_IG_PDF_used.m : log density of inverse Gamma distribution
* logsumexp.m : helper function to avoid numerical underflow.
* multitransp.m : transposing arrays of matrices.
* multiprod.m : multiplying sub-arrays

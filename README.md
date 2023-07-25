# bestLDS
This package provides code for a spectral method for estimating the system parameters of input-driven Bernoulli Linear Dynamical Systems (LDS) models and performing subsequent analysis of the estimates, as well as for using the estimates as initializations for LDS fitting using the Expectation Maximization (EM) algorithm. The estimator, called bestLDS (BErnoulli SpecTral Linear Dynamical System) is appropriate for applications to a wide variety binary time series data.  

The code can also be used to recreate all the figures in this paper. 

### Package Contents

#### bestlds
* moments.py: functions for executing the moment conversion in bestLDS and as detailed in Subsection 3.3
* simulate.py: functions for simulating data of various distributions, used throughout the paper
* ssid.py: functions for executing subspace identification methods as detailed in Subsection 3.2
* utils.py: miscellaneous helper functions
* real-data-utils.py: miscellaneous helper functions specific to processing the real data shown in Figure 4

#### examples
* EM_inits.py: script for generating the bestLDS initializations used in subsequent EM fitting shown in Figure 3
* error_metrics.py: basic script for generating the types of error metrics shown in Figure 2
* error_metrics.ipynb: code used for running the simulations and computing the error metrics shown in Figure 2
* estimator_run_time.py: script to compute the bestLDS run times as shown in Table 2 and Supplementary Figure 2
* fit_EM.py: various functions for fitting the Expectation Maximization (EM) algorithm using the <code>ssm</code> [package](https://github.com/lindermanlab/ssm) as well as for checking convergence and plotting the ELBOs, as shown in Figure 3 and Table 2
* model_comparisons.py: code for generating the simulated datasets, train/test splits, and bestLDS fits for obtaining the model comparison values shown in Table 1
* run_EM.ipynb: code used for fitting the Expectation Maximization (EM) algorithm using the <code>ssm</code> [package](https://github.com/lindermanlab/ssm) as well as for checking convergence and plotting the ELBOs, as shown in Figure 3 and Table 2

#### data
* em-fits: contains all the results after fitting EM using the parameters in <code>em-inits</code>, including the EM-inferred parameters and the ELBOs/LLs
* em-inits: contains all the inferred parameters (from bestLDS and Gaussian LDS) used to initialize the Expectation Maximization (EM) fits shown in Figure 3 
* model comparisons: contains all the true information about the simulated data as well as the bestLDS, Gaussian, and Poisson LDS inferred parameters
* simulations: contains the saved moment conversion files, all the saved error metrics, and the characteristics of the recovered latents on resimulation shown in Figure 2
* real-data: contains all the real input-output data shown in Figure 4

#### figures
* fig2.ipynb: contains all the plotting code used to produce the panels in Figure 2
* fig3.ipynb: contains all the plotting code used to produce the panels in Figure 3
* fig4.ipynb: contains all the fitting, analysis, and plotting code used to produce the panels in Figure 4
* suppfig1.ipynb: contains all the plotting code used to produce the panels in Supplementary Figure 1
* suppfig2.ipynb: contains all the plotting code used to produce the panels in Supplementary Figure 2
* suppfig3.ipynb: contains all the plotting code used to produce the panels in Supplementary Figure 3
* suppfig4.ipynb: contains all the plotting code used to produce the panels in Supplementary Figure 4
* illustrator_files: folder containing the .ai and .pdf files of the figures shown in the paper
* saved_images: folder containing all the .pdf image files of individual panels produced from python code and arranged into the figures

#### miscellaneous packages
* glmhmm: code for computing the GLM model comparisons for the real data, as shown in Figure 4
* SIPPY-master: code for generating the Gaussian parameter estimates, used for comparisons in Tables 1 and 2 and Figures 3 and 4
* simplelds: folder containing MATLAB scripts used to evaluate the log-evidence of simulated datasets in which q=1, as shown in Figure 3
* ssidforplds: code for generating the Poisson parameter estimates (from [Buesing et. al (2012)](https://proceedings.neurips.cc/paper_files/paper/2012/hash/d58072be2820e8682c0a27c0518e805e-Abstract.html)), used for comparisons in Table 1 and Figure 3

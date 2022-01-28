# best-LDS
This package provides fitting code for fitting a BErnoulli SpecTral Linear Dynamical System (best-LDS) model to binary time series data.  

### Package Contents

#### general scripts
* error_metrics.ipynb: code used for running the simulations and computing the error metrics shown in Figure 2
* fit_EM.py: various functions for fitting the Expectation Maximization (EM) algorithm using the <code>ssm</code> [package] (https://github.com/lindermanlab/ssm) as well as for checking convergence and plotting the ELBOs, as shown in Figure 3 and Table 2
* model_comparisons.py: code for generating the simulated datasets, train/test splits, and best-LDS fits for obtaining the log-evidence values shown in Table 1
* moment_conversion.py: functions for executing the moment conversion in best-LDS and as detailed in Subsection 2.3
* run_EM.py: code used for fitting the Expectation Maximization (EM) algorithm using the <code>ssm</code> [package] (https://github.com/lindermanlab/ssm) as well as for checking convergence and plotting the ELBOs, as shown in Figure 3 and Table 2
* simulate.py: functions for simulating data of various distributions, used throughout the paper
* ssid.py: functions for executing subspace identification methods as detailed in Subsection 2.2
* simplelds: folder containing MATLAB scripts used to evaluate the log-evidence of simulated datasets, as shown in Table 1


#### data
* em-fits: contains all the results after fitting EM using the parameters in <code>em-inits</code>, including the EM-inferred parameters and the ELBOs
* em-inits: contains all the inferred parameters (from best-LDS and gaussian-LDS) used to initialize the Expectation Maximization (EM) fits shown in Figure 3 
* model comparisons: contains all the true information about the simulated data, the best-LDS inferred parameters, and the train/test splits used to evaluate the log-evidence values shown in Table 1
* simulations: contains the saved moment conversion files, all the saved error metrics, and the characteristics of the recovered latents on resimulation shown in Figure 2
* real-data: contains all the real input-output data shown in Figure 4

#### figures
* fig2.ipynb: contains all the plotting code used to produce the panels in Figure 2
* fig3.ipynb: contains all the plotting code used to produce the panels in Figure 3
* fig4.ipynb: contains all the fitting, analysis, and plotting code used to produce the panels in Figure 4
* illustrator_files: folder containing the .ai and .pdf files of the figures shown in the paper
* saved_images: folder containing all the .pdf image files of individual panels produced from python code and arranged into the figures
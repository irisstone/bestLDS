# Import our functions
from moment_conversion import *
from ssid import *
from simulate import *

# Import other things that might be useful
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import subspace_angles
from sklearn.model_selection import KFold
import scipy.io as sio
import ssm as ssm

# set seed
seed = 8656
np.random.seed(seed)

# set data dimensions
q = 8  # dimension of the data
p = 6  # dimension of the state space
m = 4  # dimesion of the inputs
k = 8  # Hankel size
N = 256000 # total number of data points
ntrials = 5  # number of trials

# set primary system parameters
A = generate_dynamics_matrix(p,eig_high=0.9,eig_low=0.5)
B = generate_input_matrix(p,m) * 0.1
D = generate_input_matrix(q,m) * 0.1
noise = np.random.standard_normal(size=(q,p))
U,_,_ = np.linalg.svd(noise,full_matrices=False)
C = U * 10

# set other parameters
d = np.zeros(q)
Q = np.eye(p) * 0.1
Q0 = np.eye(p) * 0.1
R = np.eye(q) * 0.1
x0 = np.zeros(p)
muu = np.zeros(m)
Qu = np.eye(m) * 0.1


### SIMULATE DATA ---------------------------------------------------------------------------
Npertrial = int(N/ntrials)
y, u, z_old, _ = simulate_driven_bernoulli_lds_trials(Npertrial,ntrials,x0,Q0,A,B,Q,C,D,d,muu,Qu,noise=R)
z_reshaped = future_past_Hankel_order_stream(z_old, k, q, flip=True)
sig_z = np.cov(z_reshaped)[: q, q : 2*q]
diag_z = np.diag(sig_z)
y, u, z, C_new = simulate_driven_bernoulli_lds_trials(Npertrial,ntrials,x0,Q0,A,B,Q,C,D,d,muu,Qu,
                                                      diag_z=diag_z,inputs=u,noise=R)

print('data simulated for %sk' %N)

### SPLIT THE DATA
folds = ntrials
train_size = int(N - N/folds)
test_size = int(N/folds)
y_train = np.zeros((folds,train_size,q))
y_test = np.zeros((folds,test_size,q))
u_train = np.zeros((folds,train_size,m))
u_test = np.zeros((folds,test_size,m))
kf = KFold(n_splits=folds)
kf.get_n_splits(y)
for i, (train_index, test_index) in enumerate(kf.split(y)):
    y_train[i,:], y_test[i,:] = y[train_index], y[test_index]
    u_train[i,:], u_test[i,:] = u[train_index], u[test_index]

### SAVE DATA
np.savez('data/model-comparisons/datasetB/datasetB_large.npz',A=A,B=B,C=C_new,D=D,Q=Q,Q0=Q0,R=R,Qu=Qu,muu=muu,
         N=N,trials=ntrials,u=u,y=y,z=z,u_test=u_test,u_train=u_train,y_test=y_test,y_train=y_train,seed=seed)

print('data split for %sk' %N)


### INFER PARAMETERS FOR BEST-LDS FOR EACH TRAINING SET
for i in range(ntrials):
    y_reshaped = future_past_Hankel_order_stream(y_train[i], k, q).T
    u_reshaped = future_past_Hankel_order_stream(u_train[i], k, m).T

    # moment conversion
    mu_zs, mu_us, sigma_zz, sigma_uu, sigma_zu = fit_mu_sigma_bernoulli_driven(y_reshaped, u_reshaped)

    # rearrange sigma, get estimate of covariance w 
    sigma_zz_full = tril_to_full(sigma_zz, 2 * k * q)
    sigma_what = get_sigmaw_driven(sigma_uu, sigma_zz_full, sigma_zu)

    # cholesky decompose R
    R = get_R(sigma_what)

    # run n4sid
    Ahat_bern,Bhat_bern,Chat_bern,Dhat_bern,_,_,_,_ = driven_n4sid(R,k,m,p,q)

    np.savez('data/model-comparisons/datsetB/datasetB_large-best-lds-fit-%s.npz' %(i+1),Ahat=Ahat_bern,Bhat=Bhat_bern,Chat=Chat_bern,
      Dhat=Dhat_bern,u_train=u_train,y_train=y_train,u_test=u_test,y_test=y_test,seed=seed)
    
    print('best-lds for %sk, training set %s' %(N,i+1))
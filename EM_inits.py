# Import our functions
from moment_conversion import *
from ssid import *
from simulate import *

# Import other things that might be useful
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import subspace_angles
import scipy.io as sio
import pickle
import sys
sys.path.insert(0, '..')

### Simulation with driven LDS-BEST
seed = 27642
np.random.seed(seed)

## System orders
q = 5  # dimension of the data
p = 2  # dimension of the state space
k = p  # Hankel parameter
m = 3  # dimension of the inputs

# System parameters
#A = generate_dynamics_matrix(p, eig_high = 0.99, eig_low = 0.9)
theta = np.pi/2
A = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
B = generate_input_matrix(p,m) * .1
D = generate_input_matrix(q,m) * .1
Q = np.eye(p) * .1
gamma = np.eye(q) * .1
d = np.zeros(q)

# method 1 for C
# noise = np.random.standard_normal(size=(q,p))
# U,_,_ = np.linalg.svd(noise,full_matrices=False)
# C = U * 1e4

# method 2 for C
M = np.random.uniform(0,5,size=(q,p))
C,rr = np.linalg.qr(M)
C = C * 1e4

# Prior parameters
x0 = np.zeros(p)
Q0 = np.eye(p) * .1

# Inputs
muu = np.zeros(m)
Qu = np.eye(m) * 2

# Simulated data size
N = 256000

### Run simulation ###

# Get initial diag z for unitizing
# u = np.random.multivariate_normal(muu,Qu,size=N)
#y, x, u, z, _ = simulate_driven_bernoulli_lds(N,x0,Q0,A,B,Q,C,D,d,muu,Qu,inputs=u)
y, x, u, z, _ = simulate_driven_studentt_bernoulli_lds(N,x0,Q0,A,B,Q,C,D,d)

z_reshaped = future_past_Hankel_order_stream(z, k, q, flip=True)
sig_z = np.cov(z_reshaped)[: q, q : 2*q]
diag_z = np.diag(sig_z)

# Resimulate with unitizing data
#y, x, u, z, C_new = simulate_driven_bernoulli_lds(N,x0,Q0,A,B,Q,C,D,d,muu,Qu,diag_z=diag_z,inputs=u)
y, x, u, z, C_new = simulate_driven_studentt_bernoulli_lds(N,x0,Q0,A,B,Q,C,D,d,diag_z=diag_z,inputs=u)

# moment conversion
y_reshaped = future_past_Hankel_order_stream(y, k, q).T
u_reshaped = future_past_Hankel_order_stream(u, k, m).T
mu_zs, mu_us, sigma_zz, sigma_uu, sigma_zu = fit_mu_sigma_bernoulli_driven(y_reshaped, u_reshaped)

# rearrange sigma, get estimate of covariance w 
sigma_zz_full = tril_to_full(sigma_zz, 2 * k * q)
sigma_what = get_sigmaw_driven(sigma_uu, sigma_zz_full, sigma_zu)

# cholesky decompose R
R = get_R(sigma_what)

# run n4sid
Ahat,Bhat,Chat,Dhat,_,_,_,ss = driven_n4sid(R,k,m,p,q)
    
# Store results
np.savez('data/em-inits/datasetE_nonnormal/best-lds.npz', A=A, B=B, C=C, C_new=C_new, D=D, 
            Q=Q, Q0=Q0, x0=x0,muu=muu, Qu=Qu, y=y, z=z, u=u, x=x, Ahat=Ahat, Bhat=Bhat, Chat=Chat, Dhat=Dhat)
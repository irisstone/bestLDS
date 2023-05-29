#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:21:47 2023

@author: istone

Example script to demonstrate how to get averaged error metrics for a simulated dataset over a range of Ns
"""

# Import sys and add folder one level up to path
import sys
sys.path.insert(0, '..')

# Import our functions
from bestlds.moments import *
from bestlds.ssid import *
from bestlds.simulate import *

# Import other things that might be useful
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import subspace_angles
import scipy.io as sio
import pickle
import time

### Simulation with driven LDS-BEST
seed = 9357
np.random.seed(seed) # 8656 # 9347

## System orders
q = 1 # dimension of the data
p = 3  # dimension of the state space
ks = [3] # Hankel parameter
m = 3

## Specific system parameters
# System parameters

A = generate_dynamics_matrix(p, eig_high = 0.99, eig_low = 0.9)
B = generate_input_matrix(p,m) * .1
D = generate_input_matrix(q,m) * .1
Q = np.eye(p) * .1
gamma = np.eye(q) * .1
d = np.zeros(q)

# use this if p < q
# noise = np.random.standard_normal(size=(q,p))
# U,_,_ = np.linalg.svd(noise,full_matrices=False)
# C = U

# use this if p > q
M = np.random.uniform(0,1,size=(p,q))
C,rr = np.linalg.qr(M)
C = C.T * 0.1 

# Prior parameters
x0 = np.zeros(p)
Q0 = np.eye(p) * .1

# Inputs
muu = np.zeros(m)
Qu = np.eye(m) * 2

### Simulation parameters
Ns = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
num_sims = 30

# Storage variables
Aangles = np.zeros((len(Ns), num_sims, len(ks))) # Mean difference in the sorted eigenvalues
Cangles = np.zeros((len(Ns), num_sims, len(ks))) # Subspace angle between true C and inferred C
Dangles = np.zeros((len(Ns), num_sims, len(ks))) # Reconstruction error between true D and inferred D
Gangles = np.zeros((len(Ns), num_sims, len(ks))) # Reconstruction error in the gain matrix
run_times = np.zeros((len(Ns), num_sims, len(ks))) # Time to perform moment conversion
C_news = np.zeros((len(Ns), num_sims, len(ks), q, p))

ys = np.empty((len(Ns), num_sims), dtype=object)
xs = np.empty((len(Ns), num_sims), dtype=object)
us = np.empty((len(Ns), num_sims), dtype=object)
zs = np.empty((len(Ns), num_sims), dtype=object)

Ahats = np.zeros((len(Ns), num_sims, p, p))
Bhats = np.zeros((len(Ns), num_sims, p, m))
Chats = np.zeros((len(Ns), num_sims, q, p))
Dhats = np.zeros((len(Ns), num_sims, q, m))

# True derived variables
true_A_eigs = np.sort(np.linalg.eig(A)[0])

# Run simulations
for ndx, N in enumerate(Ns):
    print('N: ', N)
    for kdx, k in enumerate(ks):
        print('\tK: ', k)
        for sim in range(num_sims):
            print('\t\t sim #: ', sim)
        
            # Get initial diag z for unitizing
            u = np.random.multivariate_normal(muu,Qu,size=N)
            y, x, u, z, _ = simulate_driven_bernoulli_lds(N,x0,Q0,A,B,Q,C,D,d,muu,Qu,inputs=u)#,R=gamma)

            z_reshaped = future_past_Hankel_order_stream(z, k, q, flip=True)
            sig_z = np.cov(z_reshaped)[: q, q : 2*q]
            diag_z = np.diag(sig_z)
        
            # Resimulate with unitizing data
            y, x, u, z, C_new = simulate_driven_bernoulli_lds(N,x0,Q0,A,B,Q,C,D,d,muu,Qu,diag_z=diag_z,inputs=u)#,R=gamma)

            # Store
            ys[ndx, sim] = y
            xs[ndx, sim] = x
            us[ndx, sim] = u
            zs[ndx, sim] = z

            t0 = time.time()
            
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

            estimator_time = time.time() - t0

            # Save time to run bestLDS estimator
            run_times[ndx, sim, kdx] = estimator_time
            
            # Store
            # Ahats[ndx, sim, :, :] = Ahat
            # Bhats[ndx, sim, :, :] = Bhat
            # Chats[ndx, sim, :, :] = Chat
            # Dhats[ndx, sim, :, :] = Dhat

            # # Get error of A
            # est_A_eigs = np.sort(np.linalg.eig(Ahat)[0])
            # Aangles[ndx, sim, kdx] = np.mean(np.abs(true_A_eigs - est_A_eigs))

            # # Get error of C
            # Cangles[ndx,sim,kdx] = subspace_angles(C_new,Chat)[0] # subspace angle

            # # Get error of D
            # Dangles[ndx,sim,kdx] = np.mean(np.abs(D - Dhat))

            # # Get gain error
            # true_gain = C_new @ np.linalg.inv(np.eye(p) - A) @ B + D
            # est_gain = Chat @ np.linalg.inv(np.eye(p) - Ahat) @ Bhat + Dhat
            # Gangles[ndx, sim, kdx] = np.mean(np.abs(est_gain - true_gain))
            
            # # Save C_new
            # C_news[ndx, sim, kdx, :, :] = C_new
            
        # # Intermediate storage in case everything crashes woop woop
        # np.savez('./err_metrics/fig2_small_Ns.npz', A=A, B=B, C=C, C_news=C_news, D=D, 
        #          Q=Q, Q0=Q0, x0=x0,muu=muu, Qu=Qu, 
        #          ys=ys, zs=zs, us=us, xs=xs, Ahats=Ahats, Bhats=Bhats, Chats=Chats, Dhats=Dhats,
        #          Aangles=Aangles, Cangles=Cangles, Dangles=Dangles, Gangles=Gangles)

        np.save('../data/error_metrics/datasetA_estimator_run_times.npy', run_times)

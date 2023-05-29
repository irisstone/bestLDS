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

### Load data
data = np.load('../data/em-inits/datasetG_q1/bestLDS-smallC-2.npz', allow_pickle=True)
y = data['y']
u = data['u']
z = data['z']

# load real data
# y = np.load('../data/real-data/d2mice_outputs.npy')
# y = 1-np.nonzero(y)[1][np.newaxis].T
# inputs  = np.load('../data/real-data/d2mice_inputs.npy')
# u = np.hstack((inputs[:,0:2],inputs[:,-2:]))

q = y.shape[1]
p = z.shape[1]
m = u.shape[1]
k = p

print(q)
print(m)


# tic
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

# toc
estimator_time = time.time() - t0

print(estimator_time)

import numpy as np
from scipy.stats import norm, multivariate_normal, t

def generate_dynamics_matrix(p, eig_high=0.9, eig_low=0.5):
  '''
  Generate random p x p dynamics matrix with eigenvalues eig_low <= lambda <= eig_high
  '''
  lambd = np.diag(np.linspace(eig_high,eig_low,num=p))
  Q = np.random.randn(p,p)

  return Q @ lambd @ np.linalg.inv(Q)
  
def generate_input_matrix(p, m, shigh=0.9, slow=0.5):
  '''
  Generates p x m input dynamics matrix (B) with singular values bounded slow <= SVs <= shigh
  '''
  sings = np.linspace(shigh,slow,num=min(p,m))
  U = np.random.randn(p,p)
  V = np.random.randn(m,m)
  S = np.zeros((p,m))
  np.fill_diagonal(S, sings)

  return U @ S @ V

def simulate_driven_mvn(N, mu_y, sigma_y, mu_u, sigma_u, sigma_uz):
  '''
  Simulate output of probit GLM with underlying data drawn
  from multivariate normal distribution with given parameters.
  Returns y = N x len(mu_y), u = N x len(mu_u) 
  '''
  joint_mean = np.concatenate((mu_u, mu_y))
  joint_cov = np.hstack((sigma_u, sigma_uz))
  joint_cov = np.vstack((joint_cov, np.hstack((sigma_uz.T, sigma_y))))
  
  data = np.random.multivariate_normal(joint_mean, joint_cov, size=N)
  m = len(mu_u)
  us = data[:, :m]
  zs = data[:, m:]
  ys = (zs >= 0).astype(float)

  return ys, us

def simulate_undriven_mvn(N, mu, sigma):
  '''
  Simulate output of probit GLM with underlying data drawn
  from multivariate normal distribution with given parameters.
  Returns y = N x len(mu)
  '''  
  return np.random.multivariate_normal(mu, sigma, size=N) >= 0
  
def simulate_driven_noiseless_bernoulli_lds(N,x0,A,B,C,D,d,muu,Qu,diag_z=None,inputs=None):
  ''' 
  Simulate from a noiseless probit-bernoulli LDS with inputs
  Returns y = N x q, u = N x m
  Parameters:
    N : number of time steps
    x0 : prior mean
    A  : dynamics matrix
    B  : input dynamics matrix
    C  : loading matrix
    D  : input emissions matrix
    d  : loading translation
    u  : inputs
  '''
  outputs = np.zeros((N,len(d)))
  curr_x = np.zeros((N,A.shape[0]))
  curr_z = np.zeros((N,C.shape[0]))
  curr_x[0,:] = x0

  if inputs is None:
    u = np.random.multivariate_normal(muu,Qu,size=N)
  else:
    u = inputs

  # Renormalize C for unit covariance on z  
  if diag_z is not None:
    C_new = np.zeros_like(C)
    for j in range(A.shape[0]):
      inv_diag = 1/np.sqrt(diag_z)
      C_new[:,j] = C[:,j] * inv_diag
  else:
    C_new = C

  # Simulate
  for i in range(N):
    if len(d) == 1:
        curr_z[i] = np.squeeze(C@curr_x[i,:] + D@u[i,:] + d)
    else:
        curr_z[i] = C@curr_x[i,:] + D@u[i,:] + d
        
    outputs[i,:] = curr_z[i] >= 0
    
    # Avoid issues with endpoints
    if i < N - 1:
        curr_x[i+1,:] = A@curr_x[i,:] + B@u[i,:]

  return outputs, curr_x, u, curr_z, C_new

def simulate_driven_bernoulli_lds(N,x0,Q0,A,B,Q,C,d,muu,Qu,diag_z=None,inputs=None):
  ''' 
  Simulate from a probit-bernoulli LDS with inputs
  Returns y = N x q, u = N x m
  Parameters:
  	N : number of time steps
	x0 : prior mean
	Q0 : prior covariance
	A  : dynamics matrix
	B  : input dynamics matrix
	Q  : dynamics covariance matrix
	C  : loading matrix
	d  : loading translation
	muu : input mean
	Qu  : input covariance
	diag_z : vector to scale C such that the z variances are unitary
			 can be computed by first simulating with diag_z = None
			 and then resimulating with diag_z = cov(z)
  '''
  outputs = np.zeros((N,len(d)))
  curr_x = np.random.multivariate_normal(x0,Q0)
  curr_z = np.zeros((N, len(d)))

  if inputs is None:
    u = np.random.multivariate_normal(muu,Qu,size=N)
  else:
    u = inputs
  
  # Renormalize C for unit covariance on z  
  if diag_z is not None:
    C_new = np.zeros_like(C)
    for j in range(A.shape[0]):
      inv_diag = 1/np.sqrt(diag_z)
      C_new[:,j] = C[:,j] * inv_diag
  else:
    C_new = C
  
  # Sample  
  for i in range(N):
    if len(d) == 1:
        curr_z[i, :] = np.squeeze(C_new@curr_x + d)
    else:
        curr_z[i, :] = C_new@curr_x + d
        
    outputs[i,:] = curr_z[i, :] >= 0
    curr_x = np.random.multivariate_normal(A@curr_x + B@u[i,:],Q)

  return outputs, u, curr_z, C_new

def simulate_driven_bernoulli_lds_trials(N,T,x0,Q0,A,B,Q,C,D,d,muu,Qu,diag_z=None,inputs=None,noise=None):
  ''' 
  Simulate from a probit-bernoulli LDS with inputs
  Returns y = N x q, u = N x m
  Parameters:
    N : number of time steps per trial
    T : number of trials
    x0 : prior mean
    Q0 : prior covariance
    A  : dynamics matrix
    B  : input dynamics matrix
    Q  : dynamics covariance matrix
    C  : loading matrix
    d  : loading translation
    muu : input mean
    Qu  : input covariance
    diag_z : vector to scale C such that the z variances are unitary
        can be computed by first simulating with diag_z = None
        and then resimulating with diag_z = cov(z)
  '''
  outputs = np.zeros((N,len(d),T))
  curr_x = np.random.multivariate_normal(x0,Q0)
  curr_z = np.zeros((N,len(d),T))

  if inputs is None:
    u = np.random.multivariate_normal(muu,Qu,size=(N*T))
  else:
    u = inputs
  # Renormalize C for unit covariance on z  
  if diag_z is not None:
    C_new = np.zeros_like(C)
    for j in range(A.shape[0]):
      inv_diag = 1/np.sqrt(diag_z)
      C_new[:,j] = C[:,j] * inv_diag
  else:
    C_new = C
  
  # Sample  
  for i in range(T):
    curr_x = np.random.multivariate_normal(x0,Q0)
    for j in range(N):
      if noise is None:
          if len(d) == 1:
              curr_z[j, :, i] = np.squeeze(C_new@curr_x + D@u[i,:])
          else:
              curr_z[j, :, i] = C_new@curr_x + D@u[i,:]
      else:
          if len(d) == 1:
              curr_z[j, :, i] = np.squeeze(np.random.multivariate_normal(C_new@curr_x + D@u[i,:] + d,noise))
          else:
              curr_z[j, :, i] = np.random.multivariate_normal(C_new@curr_x + D@u[i,:] + d,noise)
          
      outputs[j,:, i] = curr_z[j, :, i] >= 0
      curr_x = np.random.multivariate_normal(A@curr_x + B@u[(i*N)+j,:],Q)

  # stack shit
  stacked_outputs = outputs[:, :, 0]
  stacked_z = curr_z[:, :, 0]
  for i in range(1, T):
    stacked_outputs = np.concatenate((stacked_outputs, outputs[:, :, i]))
    stacked_z = np.concatenate((stacked_z, curr_z[:, :, i]))

  return stacked_outputs, u, stacked_z, C_new


def simulate_studentt_bernoulli_lds(N,x0,Q0,A,B,Q,C,d,diag_z=None,inputs=None):
  ''' 
  Simulate from a probit-bernoulli LDS with student-t distrubted inputs
  Returns y = N x q, u = N x m
  Parameters:
    N : number of time steps
  x0 : prior mean
  Q0 : prior covariance
  A  : dynamics matrix
  B  : input dynamics matrix
  Q  : dynamics covariance matrix
  C  : loading matrix
  d  : loading translation
  diag_z : vector to scale C such that the z variances are unitary
       can be computed by first simulating with diag_z = None
       and then resimulating with diag_z = cov(z)
  '''
  outputs = np.zeros((N,len(d)))
  curr_x = np.random.multivariate_normal(x0,Q0)
  curr_z = np.zeros((N, len(d)))
  if inputs is None:
    u = t.rvs(df=3, loc=0, scale=0.1, size=(N, B.shape[1]))
  else:
    u = inputs

  if diag_z is not None:
    C_new = np.zeros_like(C)
    for j in range(A.shape[0]):
      inv_diag = 1/np.sqrt(diag_z)
      C_new[:,j] = C[:,j] * inv_diag
  else:
    C_new = C
    
  for i in range(N):
    if len(d) == 1:
        curr_z[i, :] = np.squeeze(C_new@curr_x + d)
    else:
        curr_z[i, :] = C_new@curr_x + d
    outputs[i,:] = curr_z[i, :] >= 0
    curr_x = np.random.multivariate_normal(A@curr_x + B@u[i,:],Q)
  return outputs, u, curr_z, C_new

def simulate_undriven_bernoulli_lds(N,x0,Q0,A,Q,C,d,diag_z=None):
  ''' 
  Simulate from a probit-bernoulli LDS with no inputs
  Returns y = N x q
  Parameters:
  	N : number of time steps
	x0 : prior mean
	Q0 : prior covariance
	A  : dynamics matrix
	B  : input dynamics matrix
	Q  : dynamics covariance matrix
	C  : loading matrix
	d  : loading translation
	diag_z : vector to scale C such that the z variances are unitary
			 can be computed by first simulating with diag_z = None
			 and then resimulating with diag_z = cov(z)
  '''
  outputs = np.zeros((N,len(d)))
  curr_x = np.random.multivariate_normal(x0,Q0)
  curr_zs = np.zeros((N,len(d)))

  # Renormalize C for unit covariance on z 
  if diag_z is not None:
    inv_diag = 1 / np.sqrt(diag_z)
    C_new = C * inv_diag.reshape(-1, 1)
  else:
    C_new = C

  # Simulate
  for i in range(N):
    curr_zs[i,:] = np.squeeze(C_new@curr_x + d)
    outputs[i,:] = curr_zs[i,:] >= 0
    curr_x = np.random.multivariate_normal(A@curr_x,Q)

  return outputs, curr_zs, C_new

def simulate_driven_bernoulli_gauss_lds(N,x0,Q0,A,B,Q,C,d,R,muu,Qu,diag_z=None):
  ''' 
  Simulate outputs from Gaussian and probit-Bernoulli LDSs operating 
  on the same latent variables.
  Returns y_bern = N x q, y_gauss = N x q, u = N x m
  Parameters:
  	N : number of time steps
	x0 : prior mean
	Q0 : prior covariance
	A  : dynamics matrix
	B  : input dynamics matrix
	Q  : dynamics covariance matrix
	C  : loading matrix
	d  : loading translation
	R  : Gaussian output covariance
	muu : input mean
	Qu  : input covariance
	diag_z : vector to scale C such that the z variances are unitary
			 can be computed by first simulating with diag_z = None
			 and then resimulating with diag_z = cov(z)
  '''
  bern_outputs = np.zeros((N,len(d)))
  gauss_outputs = np.zeros((N,len(d)))
  curr_x = np.random.multivariate_normal(x0,Q0)
  curr_z = np.zeros((N, len(d)))

  u = np.random.multivariate_normal(muu,Qu,size=N)
  
  # Renormalize C for unit covariance on z  
  if diag_z is not None:
    C_new = np.zeros_like(C)
    for j in range(A.shape[0]):
      inv_diag = 1/np.sqrt(diag_z)
      C_new[:,j] = C[:,j] * inv_diag
  else:
    C_new = C
  
  # Sample  
  for i in range(N):
    if len(d) == 1:
        curr_z[i, :] = np.squeeze(C_new@curr_x + d)
        gauss_outputs[i, :] = np.random.normal(curr_z[i, :], R)
    else:
        curr_z[i, :] = C_new@curr_x + d
        gauss_outputs[i, :] = np.random.multivariate_normal(curr_z[i, :], R)
        
    bern_outputs[i, :] = curr_z[i, :] >= 0
    curr_x = np.random.multivariate_normal(A@curr_x + B@u[i,:],Q)

  return outputs, u, curr_z, C_new

def simulate_driven_noiseless_gauss_lds(N,x0,A,B,C,D,d,u):
  ''' 
  Simulate outputs from Gaussian LDS without noise.
  Parameters:
    N : number of time steps
    x0 : prior mean
    A  : dynamics matrix
    B  : input dynamics matrix
    C  : loading matrix
    D  : input emissions matrix
    d  : loading translation
    u  : inputs
  '''
  curr_y = np.zeros((N,len(d)))
  curr_x = np.zeros((N,A.shape[0]))
  curr_x[0,:] = x0
  
  # Sample  
  for i in range(N):
    if len(d) == 1:
        curr_y[i, :] = np.squeeze(C@curr_x[i, :] + D@u[i,:] + d)
        #outputs[i, :] = np.random.normal(curr_z[i, :], R)
    else:
        curr_y[i, :] = C@curr_x[i, :] + D@u[i,:] + d
        #outputs[i, :] = np.random.multivariate_normal(curr_z[i, :], R)

    if i < N - 1:  
      curr_x[i+1, :] = A@curr_x[i, :] + B@u[i, :]

  return curr_y, curr_x


def get_good_cov(p):
  '''
  Return random pxp covariance matrix with unit diagonal
  '''
  A = np.random.randn(p,p)
  cov_init = A @ A.T
  for i in range(p):
      var = cov_init[i, i]
      cov_init[i, :] /= np.sqrt(var)
      cov_init[:, i] /= np.sqrt(var)
  
  return cov_init

def get_pie(A,Q,p):
  '''
  Return stationary marignal covariance of LDS
  Print 'P not converged' if stationary marginal covariance has not converged
   after 1000 steps
  '''
  P = np.eye(p)
  for i in range(1000):
    P = A@P@A.T + Q
  Pnext = A@P@A.T + Q
  if np.max(abs(P-Pnext)) < 1e-5:
    return P
  else: 
    print('P not converged')
    return -1
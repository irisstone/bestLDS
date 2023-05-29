import numpy as np
import statsmodels as sm
import statsmodels.api as sma

def get_R(sigma_w, normalize=True):
  '''
  Compute R, the Cholesky factor of sigma_w.
  Occasionally need to normalize due to floating point errors,
  so add slightly more than the smallest eigenvalue to ensure
  sigma_w is positive definite.
  '''
  if normalize and np.min(np.linalg.eig(sigma_w)[0]) <= 0:
    diag_indices = np.diag_indices(sigma_w.shape[0])
    min_eig = np.min(np.linalg.eig(sigma_w)[0])
    sigma_w[diag_indices] += np.abs(min_eig) + 0.0001

  return np.linalg.cholesky(sigma_w)

def driven_n4sid(R,k,m,n,q):
  '''
  Run the N4SID algorithm given R and the various matrix sizes.
  Implements the version of this algorithm from subid.m by Overschee
  Returns the system dynamics/loading matrices + noise covariances
  '''

  # **************************************
  #               STEP 1 
  # **************************************
  mk2  = 2*m*k

  # Set up some matrices
  Rf = R[(2*m+q)*k:(2*(m+q)*k),:]	# Future outputs
  Rp = np.vstack((R[0:(m*k),:],R[2*m*k:((2*m+q)*k),:])) # Past (inputs and) outputs
  Ru  = R[m*k:mk2,0:mk2] 	# Future inputs

  # Perpendicular Future outputs 
  sol = np.linalg.lstsq(Ru.T, Rf[:, 0:mk2].T, rcond=None)[0].T
  Rfp = np.hstack((Rf[:,0:mk2] - sol@Ru , Rf[:,mk2:(2*(m+q)*k)]))

  # Perpendicular Past
  sol = np.linalg.lstsq(Ru.T, Rp[:, 0:mk2].T, rcond=None)[0].T
  Rpp = np.hstack((Rp[:,0:mk2] - sol@Ru, Rp[:,mk2:(2*(m+q)*k)]))

  # The oblique projection:
  # Computed as 6.1 on page 166
  # obl/Ufp = Yf/Ufp * pinv(Wp/Ufp) * (Wp/Ufp)
  # The extra projection on Ufp (Uf perpendicular) tends to give better
  # numerical conditioning (see algo page 131)

  # Funny rank check (SVD takes too long)
  # This check is needed to avoid rank deficiency warnings
  if np.linalg.norm(Rpp[:,((2*m+q)*k-2*q)-1:(2*m+q)*k],ord='fro') < 1e-10: 
    Ob  = (Rfp@np.linalg.pinv(Rpp.T).T)@Rp 	# Oblique projection
  else:
    sol = np.linalg.lstsq(Rpp.T, Rfp.T, rcond=None)[0].T
    Ob = sol @ Rp

  # **************************************
  #               STEP 2 
  # **************************************

  # Compute the SVD
  # Compute the matrix WOW we want to take an SVD of
  # Extra projection of Ob on Uf perpendicular
  sol = np.linalg.lstsq(Ru.T, Ob[:,0:mk2].T, rcond=None)[0].T
  WOW = np.hstack((Ob[:,0:mk2] - sol@Ru, Ob[:,mk2:2*(m+q)*k]))
  U,S,V = np.linalg.svd(WOW)
  ss = S 

  # **************************************
  #               STEP 3 
  # **************************************
  U1 = U[:,0:n] 				# Determine U1
  
  # **************************************
  #               STEP 4 
  # **************************************
  # Determine gam and gamm
  gam  = U1@np.diag(np.sqrt(ss[0:n]))
  gamm = gam[0:q*(k-1),:]

  # The pseudo inverses
  gam_inv  = np.linalg.pinv(gam)
  gamm_inv = np.linalg.pinv(gamm)

  # **************************************
  #               STEP 5 
  # **************************************

  # Determine the matrices A and C
  mat1 = np.hstack((gam_inv@R[(2*m+q)*k:2*(m+q)*k,0:(2*m+q)*k], np.zeros((n,q))))
  mat2 = R[m*k:2*m*k,0:(2*m+q)*k+q]
  Rhs = np.vstack((mat1,mat2))
  Lhs = np.vstack((gamm_inv@R[(2*m+q)*k+q:2*(m+q)*k,0:(2*m+q)*k+q],R[(2*m+q)*k:(2*m+q)*k+q,0:(2*m+q)*k+q]))

  # Solve least squares
  sol = np.linalg.lstsq(Rhs.T,Lhs.T, rcond=None)[0].T

  # Extract the system matrices A and C
  A = sol[0:n,0:n]
  C = sol[n:n+q,0:n]
  res = Lhs - sol@Rhs 			# Residuals

  ######################################
  ###   Recompute gamma from A and C
  ######################################
  gam = np.vstack((C, np.zeros((k * q - C.shape[0], C.shape[1]))))
  for j in range(2,k+1):
    gam[(j-1)*q:j*q,:] = gam[(j-2)*q:(j-1)*q,:]@A 

  gamm = gam[0:q*(k-1),:]     
  gam_inv = np.linalg.pinv(gam)
  gamm_inv = np.linalg.pinv(gamm)

  #############################################
  ###   Recompute the states with the new gamma
  #############################################

  Rhs = np.vstack((np.hstack((gam_inv@R[(2*m+q)*k:2*(m+q)*k,0:(2*m+q)*k],np.zeros((n,q)))),R[m*k:2*m*k,0:(2*m+q)*k+q]))
  Lhs = np.vstack((gamm_inv@R[(2*m+q)*k+q:2*(m+q)*k,0:(2*m+q)*k+q],R[(2*m+q)*k:(2*m+q)*k+q,0:(2*m+q)*k+q]))

  # **************************************
  #               STEP 6 
  # **************************************

  # P and Q as on page 125
  P = Lhs - np.vstack((A,C))@Rhs[0:n,:]
  P = P[:,0:2*m*k]
  Q = R[m*k:2*m*k,0:2*m*k] 		# Future inputs

  # L1, L2, M as on page 119
  L1 = A @ gam_inv
  L2 = C @ gam_inv
  M  = np.hstack((np.zeros((n,q)),gamm_inv))
  X  = np.vstack((np.hstack((np.eye(q), np.zeros((q,n)))),np.hstack((np.zeros((q*(k-1),q)),gamm))))

  totm = 0
  for j in range(1, k + 1): # double dipping, updated from range(k)
    # Calculate N and the Kronecker products (page 126)
    N = np.vstack((np.hstack(( M[:,(j-1)*q:q*k] - L1[:,(j-1)*q:q*k], np.zeros( (n,(j-1)*q)) )),np.hstack((-L2[:,(j-1)*q:q*k], np.zeros((q,(j-1)*q))))))
    
    if j == 1:
      N[n:n+q,0:q] = np.eye(q) + N[n:n+q,0:q]

    N = N@X
    totm = totm + np.kron(Q[(j-1)*m:j*m,:].T,N)
     
  # Solve Least Squares
  P = P.T.reshape(-1, 1)
  sol = np.linalg.lstsq(totm, P,rcond=None)[0]

  # Find B and D
  sol_bd = np.reshape(sol, (n+q,m), order='F')
  D = sol_bd[0:q,:]
  B = sol_bd[q:q+n,:]


  # **************************************
  #               STEP 7 
  # **************************************

  if (np.linalg.norm(res) > 1e-10): 
    # Determine QSR from the residuals
    # Determine the residuals
    cov_mat = res@res.T			# Covariance
    Qs = cov_mat[0:n,0:n]
    Ss = cov_mat[0:n,n:n+q]
    Rs = cov_mat[n:n+q,n:n+q]
    
  #   sig = scipy.linalg.solve_discrete_lyapunov(A,Qs)
  #   G = A@sig@C.T + Ss
  #   L0 = C@sig@C.T + Rs

  #   # Determine K and Ro
  #   K,Ro = scipy.linalg.solve_discrete_are(A,G,C,L0)
  # else: 
  #   Ro = []
  #   K = []
  
  return A,B,C,D,Qs,Rs,Ss,ss

def undriven_n4sid(R,k,n,q):

  # **************************************
  #               STEP 1 
  # **************************************

  # First compute the orthogonal projections Ob and Obm
  Ob = R[q*k:2*q*k,0:q*k]

  # **************************************
  #               STEP 2 
  # **************************************

  U,S,V = np.linalg.svd(Ob)
  ss = S

  # **************************************
  #               STEP 3 
  # **************************************

  U1 = U[:,0:n]       # Determine U1

  # **************************************
  #               STEP 4 
  # **************************************

  # Determine gam and gamm
  gam  = U1@np.diag(np.sqrt(ss[0:n]))
  gamm = U1[0:q*(k-1),:]@np.diag(np.sqrt(ss[0:n]))
  # And their pseudo inverses
  gam_inv  = np.linalg.pinv(gam)
  gamm_inv = np.linalg.pinv(gamm)

  # **************************************
  #               STEP 5 
  # **************************************

  # Determine the states Xi and Xip
  Xi  = gam_inv  @ Ob
  Xip = gamm_inv @ R[q*(k+1):2*q*k,0:q*(k+1)]

  # **************************************
  #               STEP 2a 
  # **************************************

  # Determine the state matrices A and C
  Rhs = np.hstack((Xi , np.zeros((n,q))) )  # Right hand side
  Lhs = np.vstack((Xip   ,  R[q*k:q*(k+1),0:q*(k+1)])) # Left hand side

  # Solve least squares
  sol = np.linalg.lstsq(Rhs.T,Lhs.T, rcond=None)[0].T

  # Extract the system matrices
  A = sol[0:n,0:n]
  C = sol[n:n+q,0:n]


  # **************************************
  #               STEP 3a 
  # **************************************
  # Determine the residuals
  res = Lhs - sol@Rhs   # Residuals
  cov = res@res.T       # Covariance
  Qs = cov[0:n,0:n]
  Ss = cov[0:n,n:n+q]
  Rs = cov[n:n+q,n:n+q] 

  # **************************************
  #               STEP 4b 
  # **************************************

  # sig = dlyap(A,Qs);
  # G = A*sig*C' + Ss;
  # L0 = C*sig*C' + Rs;

  # # Determine K and Ro
  # [K,Ro] = gl2kr(A,G,C,L0);

  return A, C, Qs, Rs, Ss, ss

def latent_regression(x, y, u):
    '''
        Given access to the latents, use regression to extract the system parameters.
    '''
    # Extract relevant variables
    p = x.shape[1]
    q = y.shape[1]
    m = u.shape[1]
    
    # Extract the dynamics matrices by least squares on the latents, predicting
    # the next time-step.
    Y = x[1:, :]
    X = np.hstack((x[:-1, :], u[:-1, :]))
    
    w_hat = np.linalg.inv(X.T @ X) @ X.T @ Y
    A = w_hat[:p, :].T
    B = w_hat[p:, :].T
    
    # Extract the emissions matrices by probit regression on the latents,
    # predicting the output.
    
    # Atm, only implementation of probit regression requires 1d outputs, so dumb solution
    # is to just loop over the columns of the target
    C = np.zeros((q, p))
    D = np.zeros((q, m))
    pred = np.hstack((x, u))
    for i in range(q):
        targ = y[:, i]
        fit = sm.discrete.discrete_model.Probit(targ, pred).fit(disp=0)
        coefs = fit.params
        
        C[i, :] = coefs[:p]
        D[i, :] = coefs[p:]
    
    return A, B, C, D
        
    
    
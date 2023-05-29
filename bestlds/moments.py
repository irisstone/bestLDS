import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import root
from scipy.special import erf

def get_sigmaw_driven(sigma_u, sigma_z, sigma_uz):
  '''
  Permute the rows and columns of sigma_u, sigma_z, sigma_uz such that
  they look like the ws from N4sid. i.e. [u(past to future) z(past to future)]
  '''

  # Permute uu chunk
  perm_sigma_u = np.zeros_like(sigma_u)
  mk2 = sigma_u.shape[0]
  mk = int(mk2 / 2)
  perm_sigma_u[:mk, :mk] = np.flip(np.flip(sigma_u[mk:, mk:], axis=0), axis=1)
  perm_sigma_u[mk:, mk:] = sigma_u[:mk, :mk]
  perm_sigma_u[:mk, mk:] = np.flip(sigma_u[mk:, :mk], axis=0)
  perm_sigma_u[mk:, :mk] = np.flip(sigma_u[:mk, mk:], axis=1)

  # Permute zz chunk
  perm_sigma_z = np.zeros_like(sigma_z)
  qk2 = sigma_z.shape[0]
  qk = int(qk2 / 2)
  perm_sigma_z[:qk, :qk] = np.flip(np.flip(sigma_z[qk:, qk:], axis=0), axis=1)
  perm_sigma_z[qk:, qk:] = sigma_z[:qk, :qk]
  perm_sigma_z[:qk, qk:] = np.flip(sigma_z[qk:, :qk], axis=0)
  perm_sigma_z[qk:, :qk] = np.flip(sigma_z[:qk, qk:], axis=1)

  # Permute uz chunk
  perm_sigma_uz = np.zeros_like(sigma_uz)
  perm_sigma_uz[:mk, :qk] = np.flip(np.flip(sigma_uz[mk:, qk:], axis=0), axis=1)
  perm_sigma_uz[mk:, qk:] = sigma_uz[:mk, :qk]
  perm_sigma_uz[:mk, qk:] = np.flip(sigma_uz[mk:, :qk], axis=0)
  perm_sigma_uz[mk:, :qk] = np.flip(sigma_uz[:mk, qk:], axis=1)

  # Build them all together in a glorious heap
  step_one = np.hstack((perm_sigma_u, perm_sigma_uz))
  step_two = np.hstack((perm_sigma_uz.T, perm_sigma_z))
  sigma_w = np.vstack((step_one, step_two))

  return sigma_w

def get_sigmaw_undriven(sigma_z):
  '''
  Permute the rows and columns of  sigma_z such that
  they look like the ws from N4sid NO INPUTS. i.e. [z(past to future)]
  '''

  # Permute zz chunk
  perm_sigma_z = np.zeros_like(sigma_z)
  qk2 = sigma_z.shape[0]
  qk = int(qk2 / 2)
  perm_sigma_z[:qk, :qk] = np.flip(np.flip(sigma_z[qk:, qk:], axis=0), axis=1)
  perm_sigma_z[qk:, qk:] = sigma_z[:qk, :qk]
  perm_sigma_z[:qk, qk:] = np.flip(sigma_z[qk:, :qk], axis=0)
  perm_sigma_z[qk:, :qk] = np.flip(sigma_z[:qk, qk:], axis=1)

  return perm_sigma_z

def get_u_moments(u):
  '''
  Returns the moments of the inputs, u
  '''
  # MEAN
  muu_hat = np.mean(u, axis=0)

  # COV
  Qu_hat = np.cov(u.T)

  return muu_hat, Qu_hat

def analytical_trunc_norm_mean(mu, sigma):
  '''
  Compute mean of truncated normal distribution from 0 to infinity
  Solution from page 49 of: https://intra.ece.ucr.edu/~Korotkov/papers/Korotkov-book-integrals.pdf

  Note this is the truncated (normal distribution) not the (truncated normal) distribution
  i.e., this distribution does not integrate to 1
  '''
  a = np.sqrt(0.5 / sigma)
  beta = -mu * np.sqrt(0.5 / sigma)

  coeff = 1 / np.sqrt(2 * np.pi * sigma)
  num = np.sqrt(np.pi) * beta * (erf(beta) - 1) + np.exp(-1 * (beta ** 2))
  denom = 2 * (a ** 2)

  return coeff * (num / denom)

def yu_cov(mu_u, mu_z, sigma_z, sigma_uz):
  '''
  Compute E[y_i, u_j] given their means and covariance
  See main text for derivation
  '''
  first_term = mu_u - (sigma_uz * (1 / sigma_z) * mu_z)
  first_term *= (1 - norm.cdf(0, loc=mu_z, scale=np.sqrt(sigma_z)))

  second_term = sigma_uz / sigma_z
  second_term *= analytical_trunc_norm_mean(mu_z, sigma_z)

  return first_term + second_term

def bounded_tanh(x, bound = 1):
  '''
  Scaled tanh, transforms real numbers to range [-bound, bound]
  '''
  return np.tanh(x) * bound * 0.9999

def alpha(mu_i, sigma_ii=1):
  '''
  Compute alpha(params) = E[y_i] = E[y_i^2] = beta(params)
  '''
  return norm.cdf(mu_i)

def gamma(mu_i, mu_j, sigma_ij, sigma_ii=1, sigma_jj=1):
  '''
  Compute gamma(params) = E[y_i, y_j]
  '''
  cov = np.diag([sigma_ii, sigma_jj]).astype(float)
  cov[1,0] = sigma_ij
  cov[0,1] = sigma_ij

  return multivariate_normal.cdf(np.zeros(2), -1*np.array([mu_i,mu_j]), cov)

def second_moment_target(sigma_ij, mu_i, mu_j, second_moment):

  return second_moment - gamma(mu_i, mu_j, bounded_tanh(sigma_ij))

def input_latent_target(sigma_uz, mu_u, mu_z, cross_moment, sigma_z=1, sigma_u=1):
  bound = max(abs(sigma_z), abs(sigma_u))
  return cross_moment - yu_cov(mu_u, mu_z, sigma_z, bounded_tanh(sigma_uz, bound=bound))

def get_y_moments(y, only_lower_triag=True):
  '''
  Get the first and second moments of the outputs y (y of size N x k)
  Parameters:
    y : the data
    only_lower_triag : if True, will only return a flattened version of the lower triangle of the second moment
  '''
  twokq = y.shape[1]
  y1 = np.mean(y,axis=0)
  y2 = np.diag(y1).astype(float)

  for i in range(twokq):
    for j in range(i):
      y2[i,j] = np.mean(np.multiply(y[:, i], y[:, j]))
      y2[j,i] = y2[i, j]
  
  if only_lower_triag:
    y2_lower_triag = y2[np.tril_indices(y2.shape[0], -1)].flatten()
    return y1, y2_lower_triag
  else:
    return y1, y2

def get_cross_moments(y, u):
  '''
  Get the cross moments of the outputs y and inputs u, i.e., E[yu]
  '''
  kq2 = y.shape[1]
  km2 = u.shape[1]
  cross_moment = np.zeros((km2, kq2))
  for i in range(km2):
    for j in range(kq2):
      cross_moment[i, j] = np.mean(np.multiply(y[:, j], u[:, i]))
  
  return cross_moment

def fit_mu_sigma_bernoulli_driven(y, u):
  '''
  Given empirical first and second moments for the data, determine closest
  (mu, sigma) that induce those moments by root-finding. Whole system
  is underdetermined, so fix sigma_zzs_ii = 1 WLOG.

  Throughout, we transform the covariance values using a bounded tanh
  to avoid PSD issues with the root-finder trying values for the off-diagonal
  covariances that are not <= the variances
  '''
  ## Compute first/second moments/cross moment
  y1, y2 = get_y_moments(y)
  cross_moment = get_cross_moments(y, u)

  ## Get mu_zs and sigma_zzs
  kq2 = y.shape[1]
  km2 = u.shape[1]

  # Get mu_zs  
  mu_zs = norm.ppf(y1)

  # Get sigma_zzs
  size_lower_triangle = int((kq2 ** 2 - kq2) / 2)
  initial_guess = -1 

  params_idx = 0
  sigma_zzs = np.zeros(size_lower_triangle)
  for i in range(kq2):
    for j in range(i):
      if i == j:
        continue

      result = root(second_moment_target, x0=initial_guess, method='hybr', options={'xtol': 0.001}, args=(mu_zs[i], mu_zs[j], y2[params_idx]))

      sigma_zzs[params_idx] = bounded_tanh(result.x)
      params_idx += 1

  
  ## Get mu_us and sigma_uus
  muu, Qu = get_u_moments(u)

  ## Get sigma_zus
  sigma_zus = np.zeros((km2, kq2))
  for i in range(km2):
    for j in range(kq2):
      result = root(input_latent_target, x0=initial_guess, method='hybr', options={'xtol': 0.001}, args=(muu[i], mu_zs[j], cross_moment[i,j], 1, Qu[i, i]))
      bound = max(abs(Qu[i,i]), 1)
      sigma_zus[i,j] = bounded_tanh(result.x, bound=bound)
      
  return mu_zs, muu, sigma_zzs, Qu, sigma_zus

def fit_mu_sigma_bernoulli_undriven(first_moment, second_moment):
  '''
  Given empirical first and second moments for the data, determine closest
  (mu, sigma) that induce those moments by root-finding. Whole system
  is underdetermined, so fix \sigma_ii = 1 WLOG.

  Unlike driven version, this one returns the full covariance matrix, rather than its
  lower triangle.
  '''
  kq2 = len(first_moment)

  # Get mu_is
  mu_is = norm.ppf(first_moment)

  # Get sigma_ijs
  initial_guess = -1 
  sigma_ijs = np.eye(kq2)

  for i in range(kq2):
    for j in range(i):
      if i == j:
        continue
      
      result = root(second_moment_target, x0=initial_guess, method='hybr', options={'xtol': 0.001}, args=(mu_is[i], mu_is[j], second_moment[i, j]))
      sigma_ijs[i,j] = bounded_tanh(result.x, bound=max(sigma_ijs[i,i], sigma_ijs[j,j]))
      sigma_ijs[j,i] = sigma_ijs[i,j]

  return mu_is, sigma_ijs

def fit_mu_sigma_inputs_driven(y, u):
  '''
  Given empirical first and second moments for the data, determine closest
  (mu, sigma) that induce those moments by root-finding. Whole system
  is underdetermined, so fix sigma_zzs_ii = 1 WLOG.

  Throughout, we transform the covariance values using a bounded tanh
  to avoid PSD issues with the root-finder trying values for the off-diagonal
  covariances that are not <= the variances
  '''

  ## Get mu_zs and sigma_zzs
  kq2 = y.shape[1]
  km2 = u.shape[1]
    
  # get first/second moments
  y1, sigma_zs = get_y_moments(y)
  mu_zs = y1
    
  # get cross moments
  cross_moment = get_cross_moments(y, u)

  ## Get mu_us and sigma_uus
  muu, Qu = get_u_moments(u)

  ## Get sigma_zus
  initial_guess = -1
  sigma_zus = np.zeros((km2, kq2))
  for i in range(km2):
    for j in range(kq2):
      result = root(input_latent_target, x0=initial_guess, method='hybr', options={'xtol': 0.001}, args=(muu[i], mu_zs[j],             cross_moment[i,j], 1, Qu[i, i]))
      bound = max(abs(Qu[i,i]), 1)
      sigma_zus[i,j] = bounded_tanh(result.x, bound=bound)
      
  return Qu, sigma_zus, sigma_zs

def future_past_Hankel_order_stream(y, k, q, flip=True):
  '''
  Reshape data stream y into past and future vectors stacked
  horizontally  with overlapping chunks with step size 1

  Assume: y is (N x q)
  '''
  N = y.shape[0]
  num_chunks = N - (2 * k - 1)
  mat = np.zeros((2 * k * q, num_chunks))
  for i in range(N - (2*k - 1)):
    y_chunk = y[i : i + 2*k, :]
    y_plus = y_chunk[k:, :]
    y_minus = y_chunk[:k, :]

    if flip:
      y_pm = np.concatenate((y_plus.flatten(), np.flip(y_minus.flatten(), axis=0)))

    else:
      y_pm = np.concatenate((y_plus.flatten(), y_minus.flatten()))
    
    mat[:, i] = y_pm

  return mat

def future_past_Hankel_order_chunk(y, k, q, flip=True):
  '''
  Reshape data stream y into past and future vectors stacked
  horizontally  without overlapping chunks

  Assume: y is (N x q)
  '''

  N = y.shape[0]
  num_chunks = int(N / (2 * k))
  mat = np.zeros((2 * k * q, num_chunks))
  for i in range(num_chunks):
    y_chunk = y[i * (2 * k) : (i + 1) * (2 * k), :]
    y_plus = y_chunk[k:, :]
    y_minus = y_chunk[:k, :]

    if flip:
      y_pm = np.concatenate((y_plus.flatten(), np.flip(y_minus.flatten())))

    else:
      y_pm = np.concatenate((y_plus.flatten(), y_minus.flatten()))
    
    mat[:, i] = y_pm

  return mat

def tril_to_full(tril, N):
  '''
  Get full covariance matrix from its lower triangle.

  Diagonal is 1 by assumption.
  '''
  S = np.eye(N, dtype=float)

  tril_idx = 0
  for i in range(N):
    for j in range(i):
      S[i, j] = tril[tril_idx]
      S[j, i] = S[i, j]

      tril_idx += 1

  return S
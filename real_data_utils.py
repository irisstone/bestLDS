import numpy as np
from ssid import *
from moment_conversion import *
from simulate import *

def get_unique_neurons(clusters):
	return np.unique(clusters)

def build_spike_train(spike_times, spike_clusters, target_clusters, session_time, dt=0.1):
	num_neurons = len(target_clusters)

	start, end = session_time
	bins = np.arange(start, end + dt, dt)
	N = len(bins)
	
	spikes = np.zeros((N, num_neurons))
	for spike_idx in range(len(spike_times)):

		# Ignore all clusters not in our set of interest
		if spike_clusters[spike_idx] not in target_clusters:
			continue

		spike_time = spike_times[spike_idx]
		if spike_time >= end:
			break

		stdx = np.digitize([spike_time], bins)[0]
		clusterdx = np.argwhere(target_clusters == spike_clusters[spike_idx])
		if spikes[stdx, clusterdx] == 1:
			print('Already put a spike for this neuron in this bin. Consider reducing dt.')
		else:
			spikes[stdx, clusterdx] = 1

	return spikes

def build_stim_train(L_contrast, R_contrast, stim_on, stim_off, session_time, dt=0.1):
	start, end = session_time
	bins = np.arange(start, end + dt, dt)
	N = len(bins)

	no_nan_L = np.nan_to_num(L_contrast)
	no_nan_R = np.nan_to_num(R_contrast)

	inputs = np.zeros((N, 1))
	for sdx in range(len(stim_on)):
		onset = stim_on[sdx]
		offset = stim_off[sdx]

		if onset < start or offset >= end:
			continue

		net_contrast = no_nan_L[sdx] - no_nan_R[sdx]

		onset_bin = np.digitize([onset], bins)[0]
		offset_bin = np.digitize([offset], bins)[0]
		inputs[onset_bin : offset_bin + 1] = net_contrast

	return inputs

def build_reward_train(feedback_times, feedback_type, session_time, dt=0.1):
	start, end = session_time
	bins = np.arange(start, end + dt, dt)
	N = len(bins)

	inputs = np.zeros((N, 1))
	for sdx in range(len(feedback_times)):
		onset = feedback_times[sdx]
		offset = onset + 0.1

		if onset < start or offset >= end:
			continue

		reward = feedback_type[sdx]

		onset_bin = np.digitize([onset], bins)[0]
		offset_bin = np.digitize([offset], bins)[0]
		inputs[onset_bin : offset_bin + 1] = reward

	return inputs

def best_lds(y, u, k, p=None):
	q = y.shape[1]
	m = u.shape[1]
	
	y_reshaped = future_past_Hankel_order_stream(y, k, q).T
	u_reshaped = future_past_Hankel_order_stream(u, k, m).T

	mu_zs, mu_us, sigma_zz, sigma_uu, sigma_zu = fit_mu_sigma_bernoulli_driven(y_reshaped, u_reshaped)

	# rearrange sigma, get estimate of covariance w 
	sigma_zz_full = tril_to_full(sigma_zz, 2 * k * q)
	sigma_what = get_sigmaw_driven(sigma_uu, sigma_zz_full, sigma_zu)

	# cholesky decompose R
	R = get_R(sigma_what)

	# run n4sid
	if p:
		Ahat,Bhat,Chat,Dhat,_,_,_,ss = driven_n4sid(R,k,m,p,q)
	else: 
		Ahat,Bhat,Chat,Dhat,_,_,_,ss = driven_n4sid_nop(R,k,m,q)
		return ss

	return Ahat, Bhat, Chat, Dhat

def crossval_split(x,y,sessions,mouseIDs,test_size=0.2, seeds=None):

    '''
    Splits data into train and test sets for cross validation by partitioning entire sessions and balancing 
    the number of animals in each test set. 
    Parameters
    ----------
    x : N x m design matrix
    y : length N vector of observations
    sessions : vector containing the starting indices of each session
    mouseIDs : vector of length N indicating which animal each trial is associated with
    test_size : optional, the percentage of sessions to put in each test set (default is 0.2)
    seeds : optional, list of random seeds that determines how train and test sets are split (length = N/test_size)
    
    Returns
    -------
    x_train : training sets for the design matrix
    x_test : test sets for the design matrix
    y_train : training sets for the observations
    y_test : test sets for the observations
    sessions_train : starting indices of the sessions in each training set
    sessions_test : starting indices of the sessions in each test set
    testIx : indices of the trials alloted to each test set
    seeds : list of random seeds that determines how train and test sets are split (length = N/test_size)
    '''

    # if seeds not specified, choose randomly
    if seeds is None:
        seeds = np.random.randint(0,high=500,size=int(N/0.2))

    # initialize as lists since not every test/train set will be exactly the same size
    x_train, x_test, y_train, y_test, sessions_train, sessions_test, testIx = [],[],[],[],[],[],[]

    # split the data
    for seed in seeds:
        train_ix, sessionsTrain, test_ix, sessionsTest = splitData(sessions,mouseIDs,testSize=test_size,seed=seed)
        x_train.append(x[train_ix,:])
        x_test.append(x[test_ix,:])
        y_train.append(y[train_ix])
        y_test.append(y[test_ix])
        sessions_train.append(sessionsTrain)
        sessions_test.append(sessionsTest)
        testIx.append(test_ix)

    return x_train, x_test, y_train, y_test, sessions_train, sessions_test, testIx, seeds

if __name__ == '__main__':
	# Choice coding: -1 = right
	#                 1 = left
	# super cursed

	## Get data from open-source IBL DB
	print('Loading data', flush=True)
	from one.api import ONE
	one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', silent=True)

	eids,info = one.search(dataset=['probes.description'],details=True)
	eid = eids[3]

	# print(one.list_datasets(eid))
	data, junk = one.load_datasets(eid, ['_ibl_trials.stimOn_times', '_ibl_trials.stimOff_times',
										 '_ibl_trials.feedbackType', '_ibl_trials.contrastLeft', '_ibl_trials.contrastRight',
										 '_ibl_trials.choice',
										 'spikes.times', 'spikes.clusters', 'clusters.brainLocationAcronyms_ccf_2017.npy'])

	stimon, stimoff, fbtype, cleft, cright, choice, spike_times, spike_clusters, cluster_locs = data



	## Preprocess raw data
	# Pick desired clusters
	print('Preprocessing data', flush=True)
	num_desired_neurons = 10
	all_neurons = get_unique_neurons(spike_clusters)

	targ_neurons = []
	for cluster_id in all_neurons:
		if cluster_locs[cluster_id] == 'CP':
			targ_neurons.append(cluster_id)

	targ_neurons = targ_neurons[:num_desired_neurons]

	# Convert spike times into spike trains
	dt = 0.0001
	start = 0
	end = 600 # 10 minutes
	spikes = build_spike_train(spike_times, spike_clusters, targ_neurons, [start, end], dt=dt)

	# Convert stim on and stim off times into inputs
	inputs = build_input_train(cleft, cright, stimon, stimoff, [start, end], dt=dt)

	# Run BEST-LDS
	print('Fitting data', flush=True)
	q = num_desired_neurons
	k = num_desired_neurons
	m = 1

	ss = best_lds(spikes, inputs, k, plot_svs=True)
	plt.figure()
	plt.plot(ss)
	plt.show()
	# A, B, C, D = best_lds(spikes, inputs, k)

	# print(A)
	# print(B)
	# print(C)
	# print(D)

	# np.savez('IBL_proc_data.npz', A=A, B=B, C=C, D=D, spikes=spikes, inputs=inputs)
	# np.savez('IBL_raw_data.npz', data=data)




	

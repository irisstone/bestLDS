import ssm
import numpy as np
import scipy.io as sio

def fit_lp_EM(y, u, num_inits, num_iters,
              init_params=None, num_substeps=2):
    '''
        Given outputs y and inputs u, run num_inits different 
        initializations of EM, for num_iters * num_substeps steps each.
        The system parameters are saved for each init every num_substeps steps of
        EM. Users may optionally specify initial settings for the
        LDS system parameters (A, B, C, D) by passing a dict with the desired
        matrices.

        y: N x q
        u: N x m
        num_inits: int
        num_iters: int
        init_params: dictionary mapping strings to matrices
        num_substeps: int
    '''
    num_inits = 3
    num_iters = 50
    elbos = np.zeros((num_inits,num_iters * 2))
    LDS = np.empty(num_inits, dtype=object)

    As = np.zeros((num_inits,num_iters,p,p))
    Bs = np.zeros((num_inits,num_iters,p,m))
    Cs = np.zeros((num_inits,num_iters,q,p))
    Ds = np.zeros((num_inits,num_iters,q,m))

    for i in range(num_inits):
        lds = ssm.LDS(q, p, M=m, emissions="bernoulli")
        
        if init_params:
            if 'A' in init_params:
                lds.dynamics.As[0] = A
            if 'B' in init_params:
                lds.dynamics.Vs[0] = B
            if 'C' in init_params:
                lds.emissions.Cs[0] = C
            if 'D' in init_params:
                lds.emissions.Fs[0] = D

        for j in range(num_iters):
            print(i, j)
            try:
                # Bizarre substep/iter construct needed to circumvent technical issues with extracting step-by-step 
                # system parameters.
                inst_elbo, _ = lds.fit(data, inputs=u, method="laplace_em", initialize=False, num_iters=num_substeps)
                elbos[i,2*j : 2*(j+1)] = inst_elbo[1:]

            except:
                lds = None
                break
                
            As[i,j,:,:] = lds.dynamics.As[0]
            Bs[i,j,:,:] = lds.dynamics.Vs[0]
            Cs[i,j,:,:] = lds.emissions.Cs[0]
            Ds[i,j,:,:] = lds.emissions.Fs[0]
        
        LDS[i] = lds

    return elbos, LDS, As, Bs, Cs, Ds

def convert_mat_em_fits_to_npz(matlab_files, num_iters, q, p, m, simulated_data=False):

    '''
    Takes in a .mat file of EM fit information and converts the saved values into the proper format
    to save them as a .npz file. Needs to be done before running the check_convergence() function and 
    running fig4.ipynb

    PARAMETERS
    ----------
    matlab_files : a list of strings containing the file paths for each .mat file that needs converting
    num_iters : the number of iterations of EM specified when fitting
    q : the dimensionality of the observations for the data 
    p : the dimensionality of the latents for the data
    m : the dimensionality of the inputs for the data
    simulated_data : boolean, True if fits come from simulated data (meaning file contains info about true parameters)
    
    RETURNS
    -------
    None (the result of running this function is that an .npz file is saved in the same location as the )
    
    '''

    # number of different files to convert and consolidate into single file
    num_inits = len(matlab_files)

    # create empty arrays for storing values
    elbos = np.zeros((num_inits, num_iters))
    As = np.zeros((num_inits, num_iters, p, p))
    Bs = np.zeros((num_inits, num_iters, p, m))
    Cs = np.zeros((num_inits, num_iters, q, p))
    Ds = np.zeros((num_inits, num_iters, q, m))
    time_per_iteration = np.zeros((num_inits, num_iters))

    for init, matlab_file in enumerate(matlab_files):

        # load .mat file
        matlab_fits = sio.loadmat(matlab_file)

        # load and convert LLs to proper format
        elbos[init] = matlab_fits['results']['logev'][0][0].T

        # get number of iterations
        num_iters = int(elbos.shape[1])

        # load and convert As to proper format
        A = np.array([matlab_fits['results']['params'][0][0][0]['As'][0][:,:,i] for i in range(num_iters)])
        As[init] = A[np.newaxis,:]

        # load and convert Bs to proper format
        B = np.array([matlab_fits['results']['params'][0][0][0]['Bs'][0][:,:,i] for i in range(num_iters)])
        Bs[init] = B[np.newaxis,:]

        # load and convert Cs to proper format
        C = np.array([matlab_fits['results']['params'][0][0][0]['Cs'][0][:,:,i] for i in range(num_iters)])
        Cs[init] = C[np.newaxis,:]

        # load and convert Ds to proper format
        D = np.array([matlab_fits['results']['params'][0][0][0]['Ds'][0][:,:,i] for i in range(num_iters)])
        Ds[init] = D[np.newaxis,:]

        # the .mat files have some extra information, too, so let's store those as well just in case
        # note that we don't need to store this multiple times, even with multiple inits, because the true
        # values for the same dataset will always be the same for every init
        if simulated_data:
            logev_true = matlab_fits['results']['logev_true'][0][0][0][0]
            A_true = matlab_fits['results']['params_true'][0][0]['A'][0][0]
            B_true = matlab_fits['results']['params_true'][0][0]['B'][0][0]
            C_true = matlab_fits['results']['params_true'][0][0]['C'][0][0]
            D_true = matlab_fits['results']['params_true'][0][0]['D'][0][0]
        time_per_iteration[init] = matlab_fits['results']['time'][0][0].T

    # save information in numpy file format
    if num_inits == 1: 
        numpy_file = matlab_file[0:-4] + '.npz'
    else: 
        numpy_file = matlab_file[0:-7] + '.npz'

    np.savez(numpy_file, elbos=elbos, As=As, Bs=Bs, Cs=Cs, Ds=Ds, logev_true=logev_true, 
             A_true=A_true, B_true=B_true, C_true=C_true, D_true=D_true, time_per_iteration=time_per_iteration)

    return None

def check_convergence(inits,tol=0.005,method='As_diff',min_steps=20, half_num_iters=True):
    
    num_inits = inits['elbos'].shape[0]

    # for the q>1 datasets using ssm, we only store the parameter values for every other iteration, 
    # thus half_num_iters=True. for the q=1 datasets using matlab, there are parameter values for 
    # every iteration and thus half_num_iters=False. 
    if half_num_iters:
        num_iters = int(inits['elbos'].shape[1]/2)
    else:
        num_iters = int(inits['elbos'].shape[1])
    
    steps_to_convergence = np.zeros((num_inits))
    elbo_diffs = np.zeros((num_inits))
    
    As = inits['As']
    Bs = inits['Bs']
    Cs = inits['Cs']
    Ds = inits['Ds']
    elbos = inits['elbos']
    
    if half_num_iters:
        min_steps = int(min_steps/2)
    
    
    for j in range(num_inits):
        start = min_steps - 1
        A_prev = As[j,start,:,:]
        Gain_prev = Cs[j,start,:,:] @ np.linalg.inv(np.eye(A_prev.shape[0])-A_prev) @ Bs[j,start,:,:] - Ds[j,start,:,:]
        for i in range(min_steps,num_iters):

            if method == 'As_diff':
                # get error using average difference between values
                A_i = As[j,i,:,:]
                diff = np.mean(abs(A_i-A_prev))
                A_prev = A_i

            elif method == 'As_eigs':
                # get error using summed difference between eigenvalues
                A_i = As[j,i,:,:]
                est_eigs_A_i = np.sort(np.linalg.eig(A_i)[0])
                est_eigs_A_prev = np.sort(np.linalg.eig(A_prev)[0])
                diff = np.sum(np.abs(est_eigs_A_i - est_eigs_A_prev))
                A_prev = A_i
            
            elif method == 'Gain':
                # get error using difference in Gain
                Gain_i = Cs[j,i,:,:] @ np.linalg.inv(np.eye(A_prev.shape[0])-As[j,i,:,:]) @ Bs[j,i,:,:] - Ds[j,i,:,:]
                diff = np.mean(abs(Gain_i - Gain_prev))
                Gain_prev = Gain_i

            elif method == 'ELBO':
                if half_num_iters:
                    diff = elbos[j,i*2]-elbos[j,i*2-1]
                else:
                    diff = elbos[j,i]-elbos[j,i-1]

            if diff <= tol: 
                if half_num_iters:
                    elbo_diff = elbos[j,i*2]-elbos[j,i*2-1]
                    steps_to_convergence[j] = i*2
                else:
                    elbo_diff = elbos[j,i]-elbos[j,i-1]
                    steps_to_convergence[j] = i

                print('initialization: %s, steps to convergence: %s, elbo diff: %.f' %(j, steps_to_convergence[j], abs(elbo_diff)))
                
                elbo_diffs[j] = abs(elbo_diff)
                break 
            
            elif i==int(num_iters)-1:
                if half_num_iters:
                    elbo_diff = elbos[j,i*2]-elbos[j,i*2-1]
                    steps_to_convergence[j] = (num_iters * 2) - 1
                else:
                    elbo_diff = elbos[j,i]-elbos[j,i-1]
                    steps_to_convergence[j] = num_iters - 1
                print('initialization: %s, steps to convergence: %s, elbo diff: %.f' %(j, steps_to_convergence[j]+1, abs(elbo_diff)))

    return elbos, steps_to_convergence

def plot_elbos(elbos,steps,suppress_ticklabels=True):
    fig,axes=plt.subplots(4,5)
    fig.set_size_inches(20, 10)
    for i in range(4):
        for j in range(5):
            axes[i,j].plot(elbos[(i*5)+j])
            axes[i,j].plot(steps[(i*5)+j],elbos[(i*5)+j,int(steps[(i*5)+j])],'k*')
            axes[i,j].set_title('init %s' %((i*5)+j+1))
            if suppress_ticklabels:
                axes[i,j].set_yticklabels('')
                axes[i,j].set_xticklabels('')
                
            
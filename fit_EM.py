import ssm
import numpy as np

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

def check_convergence(inits,tol=0.005,method='As_diff',min_steps=20):
    
    num_inits = inits['elbos'].shape[0]
    num_iters = int(inits['elbos'].shape[1]/2)
    
    steps_to_convergence = np.zeros((num_inits))
    elbo_diffs = np.zeros((num_inits))
    
    As = inits['As']
    Bs = inits['Bs']
    Cs = inits['Cs']
    Ds = inits['Ds']
    elbos = inits['elbos']
    
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

            if diff <= tol: 
                elbo_diff = elbos[j,i*2]-elbos[j,i*2-1]
                print('initialization: %s, steps to convergence: %s, elbo diff: %.f' %(j, i*2, elbo_diff))
                steps_to_convergence[j] = i*2
                elbo_diffs[j] = abs(elbo_diff)
                break 
            
            elif i==int(num_iters)-1:
                steps_to_convergence[j] = (num_iters * 2) - 1
                print('initialization: %s, steps to convergence: %s, elbo diff: %s' %(j, num_iters * 2, 'N/A'))

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
                
            
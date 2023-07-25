# Import sys and add folder one level up to path
import sys
sys.path.insert(0, '..')

import numpy as np
from glmhmm import glm
from glmhmm.analysis import reshape_obs

def perseverative_model(y_tests, prob_stay=0.70, num_sims=50):
    '''
    Parameters:
	-----------
	y_tests: list of test sets containing observations
    prob_stay: float, probability that the animal repeats the same choice on the previous trial ('perseveration factor')
    num_sims: how many simulations of the model to run (averages over the results)

	Returns:
	-----------
    percent_match_persev: array with the fraction correct for each test set
    '''

    percent_match_persev = np.zeros(5)
    for j in range(len(y_tests)):
        percent_match = np.zeros(num_sims)
        probs = [prob_stay,1-prob_stay]

        for sim in range(num_sims):
            ## simulate choices
            yt = np.squeeze(y_tests[j][0]) # observation at t=1
            y_perseverative = np.zeros_like(y_tests[j])
            y_perseverative[0] = yt
            for i in range(1,len(y_tests[j])):
                y_perseverative[i] = np.random.choice([yt,1-yt],p=probs)
                yt = y_perseverative[i][0]
                
            ## compare to real data
            percent_match[sim] = np.sum(np.squeeze(y_tests[j]) == np.squeeze(y_perseverative)) / len(y_tests[j])

        percent_match_persev[j] = np.mean(percent_match)

    return percent_match_persev

def glm_prediction_accuracy(y_trains, u_trains, y_tests, u_tests):


    percent_match_glm = np.zeros(len(y_trains))
    for i in range(len(y_trains)):
        # instantiate model
        m = u_trains[i].shape[1]
        c = y_trains[i].shape[1]
        model = glm.GLM(y_trains[i].shape[0],m,c,observations="multinomial")

        # initialize weights
        w_init = model.init_weights()

        # fit model to training data
        w, phi = model.fit(u_trains[i],w_init,y_trains[i],compHess=False)
        
        # compute prediction accuracy on test set
        phi = model.compObs(u_tests[i],w) # observation probabilities
        y_glm = np.argmax(phi, axis=1) # predict choices from observation probabilities
        y_test_vectorized = np.argmax(y_tests[i], axis=1) # convert real data from one-hot encoded array to vector of integers
        percent_match_glm[i] = np.sum(y_glm == y_test_vectorized)/len(y_test_vectorized)

        return percent_match_glm, y_glm, y_test_vectorized, phi


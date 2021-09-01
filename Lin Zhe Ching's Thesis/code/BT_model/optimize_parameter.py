import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import LabelEncoder

def optimize_parameter(
    data, 
    start_variance= 20 ** 2, 
    beta_square = 3 ** 2,
    verbose = True,
    tolerance = 1e-2,
):
    '''Fits the parameters in B-T model.
    Args:
    winners: The array of winners for each of N contests, of shape [N, 2].
    losers: The array of losers for each of N contests, of shape [N, 2].
    surface: The array of surfaces for each of N contest, of shape [N, 3].
    beta: The uncontral variances of the game
    start_variance_1: The initial variance of the surface 1
    verbose: Whether or not to print the progress of the optimisation.
    tolerance: The tolerance required for the optimisation to successfully terminate.
    '''
    def fun_to_minimize(theta):
        # Constrain
        variance = theta 
        _, discrepancy = calculate_ratings(data = data,start_mean = 1500,start_var = variance ,beta_square = beta_square,k = 0.0001,gamma_q =1/2)
        
        if verbose:
            print('variance: {} ; discrepancy: {}'.format(variance[0],discrepancy[0]))
        return discrepancy
    
    opt_result = minimize(fun_to_minimize,
                          np.array([start_variance],dtype='float'),
                          method='Nelder-Mead',
                          tol=tolerance,)
    return (opt_result.success, {'variance':opt_result.x[0]})

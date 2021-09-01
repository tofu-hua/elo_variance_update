import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
pd.set_option('display.max_columns', None)

def optimize_parameter(
    winners, 
    losers, 
    surface,
    beta = 3 ** 2,
    start_variance_1 = 10 ** 2, 
    start_variance_2 = 10 ** 2, 
    start_variance_3 = 10 ** 2, 
    start_rho_12 = 0.5, 
    start_rho_13 = 0.5, 
    start_rho_23 = 0.5,
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
    start_variance_2: The initial variance of the surface 2
    start_variance_3: The initial variance of the surface 3
    rho_12: The initial correlation coefficient between surface 1 and surface 2
    rho_13: The initial correlation coefficient between surface 1 and surface 3
    rho_23: The initial correlation coefficient between surface 2 and surface 3
    verbose: Whether or not to print the progress of the optimisation.
    tolerance: The tolerance required for the optimisation to successfully terminate.
    '''
    def fun_to_minimize(theta):
        # Constrain
        variance_1, variance_2, variance_3, rho_12, rho_13, rho_23  = theta 
        _, discrepancy = calculate_ratings(winners, losers, surface, beta, variance_1, variance_2, variance_3, rho_12, rho_13, rho_23)
        
        if verbose:
            print(
                f'variance_1: {variance_1:.2f}; '
                f'variance_2: {variance_2:.2f}; '
                f'variance_3: {variance_3:.2f}; '
                f'rho_12: {rho_12:.4f}; '
                f'rho_13: {rho_13:.4f}; '
                f'rho_23: {rho_23:.4f}; '
                f'discrepancy: {discrepancy:.3f}'
            )
        return discrepancy
    
    opt_result = minimize(fun_to_minimize,
                          np.array([start_variance_1, start_variance_2, start_variance_3, start_rho_12, start_rho_13, start_rho_23]),
                          method='Nelder-Mead',
                          tol=tolerance,)
    return (opt_result.success, {'variance_1':opt_result.x[0],
                                 'variance_2':opt_result.x[1],
                                 'variance_3':opt_result.x[2],
                                 'rho_12':opt_result.x[3],
                                 'rho_13':opt_result.x[4],
                                 'rho_23':opt_result.x[5]})
import numpy as np
def calculate_win_prob(mu_ik, mu_jk, c_ij):
    '''
    Calculate the win probability of p_ijk
    '''
    return np.array(mp.exp(mu_ik / c_ij)/(mp.exp(mu_ik / c_ij) + mp.exp(mu_jk / c_ij)), dtype = 'float64')

def calculate_discrepancy(s_ijk, mu_ik, mu_jk, c_ij):
    '''
    Calculate the discrepency between predicted and actual outcome.
    '''
    p_ijk = calculate_win_prob(mu_ik, mu_jk, c_ij)
    return -s_ijk * np.log(p_ijk) - (1 - s_ijk) * np.log(1-p_ijk)

def calculate_ratings(winners, losers, surface, beta, variance_1, variance_2, variance_3, rho_12, rho_13, rho_23):
    '''
    Calculate B-T model ratings.
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
    '''
    prior_ratings = defaultdict(lambda :(np.array([[1500],[1500],[1500]],dtype = 'float64')))
    
    surface_variance = np.array([[variance_1, rho_12 * np.sqrt(variance_1*variance_2), rho_13 * np.sqrt(variance_1*variance_3)],
                                [rho_12 * np.sqrt(variance_1*variance_2), variance_2, rho_23 * np.sqrt(variance_2*variance_3)],
                                [rho_13 * np.sqrt(variance_1*variance_3), rho_23 * np.sqrt(variance_2*variance_3), variance_3]])
    
    rating_history = list()
    total_discrepancy = 0
    for game in np.arange(winners.shape[0]):
        # current winners & losers
        cur_winner_1, cur_winner_2 = winners['winner1_name'][game], winners['winner2_name'][game]
        cur_loser_1, cur_loser_2 = losers['loser1_name'][game], losers['loser2_name'][game]
        # current surface [1,3]
        cur_surface = surface[game]
        # current player
        current_player = [cur_winner_1, cur_winner_2, cur_loser_1, cur_loser_2]
        # check whether the player play
        for player in current_player:
            if player not in prior_ratings:
                prior_ratings[player]
        # save the prior rating
        rating_history.append(copy.deepcopy(prior_ratings))
        
        new_rating = prior_ratings.copy()
        # winners & loser rating
        winner_1, winner_2 = new_rating[cur_winner_1], new_rating[cur_winner_2]
        loser_1, loser_2 = new_rating[cur_loser_1], new_rating[cur_loser_2]
        # team rating
        winner = winner_1 + winner_2
        loser = loser_1 + loser_2
        # calculate sigma_k
        sigma_k = cur_surface.dot(surface_variance).dot(cur_surface)        
        # calculate c_ij        
        c_ij = np.sqrt(sigma_k + sigma_k + 2 * beta **2)

        # calculate p(x)
        mu_ik = cur_surface.dot(winner)[0] # float
        mu_jk = cur_surface.dot(loser)[0] #float

        p_s = calculate_win_prob(mu_ik = mu_ik , mu_jk= mu_jk , c_ij= c_ij) # float
 
        # calculate discrepancy
        discrepancy = calculate_discrepancy(s_ijk= 1 , mu_ik = mu_ik, mu_jk= mu_jk, c_ij= c_ij)
        # calculate total discrepancy
        total_discrepancy += discrepancy
        # calculate jacobian 6*1
        jacobian = np.zeros((6,1))
        jacobian[:3,:] = cur_surface.reshape((3,1)) - p_s * cur_surface.reshape((3,1))
        jacobian[3:,:] = -(cur_surface.reshape((3,1)) - p_s * cur_surface.reshape((3,1)))
        jacobian = 1/c_ij * jacobian

        # calculate hessian matrix 6*6
        Sigma = block_diag(surface_variance,surface_variance)
        hes = np.zeros((6,6))
        discre = p_s*(1-p_s) * cur_surface.reshape((3,1)).dot(cur_surface.reshape((1,3)))
        hes[:3,:3] = - discre
        hes[3:,:3] =  discre
        hes[:3,3:] =  discre
        hes[3:,3:] = - discre
        hes = 1/(c_ij**2) * hes 

        hessian = - inv(Sigma) + hes

        # update the team rating
        
        #winner -= inv(hessian).dot(jacobian)[:3,:]
        #loser -= inv(hessian).dot(jacobian)[3:,:]
        # update the winner player & loser player rating
        new_rating[cur_winner_1] -= inv(hessian).dot(jacobian)[:3,:]/2
        new_rating[cur_winner_2] -= inv(hessian).dot(jacobian)[:3,:]/2
        
        new_rating[cur_loser_1] -= inv(hessian).dot(jacobian)[3:,:]/2
        new_rating[cur_loser_2] -= inv(hessian).dot(jacobian)[3:,:]/2
        
        # update prior rating
        prior_ratings = new_rating
        
    return rating_history, total_discrepancy
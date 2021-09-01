import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
pd.set_option('display.max_columns', None)

def calculate_ratings(data,start_mean = 1500,start_var = (100)**2,beta_square = 3**2,k = 0.0001,gamma_q =1/2):
    '''
    Args:
        data: The atp matches double game
        start_mean: Each of the player start with performance mean
        start_var: Each of the player start with performance variance
        beta_square: the bias of the uncertainty about the actual performance X
        k: the small positive number to prevent variance from being negative
        gamma_q: update rules
    Return: 
        player_rate: each player skill
    '''
    winner_double = list(zip(data['winner1_name'],data['winner2_name']))
    loser_double = list(zip(data['loser1_name'],data['loser2_name']))
    player_double = list(zip(data['winner1_name'],data['winner2_name'],data['loser1_name'],data['loser2_name']))
    prior_rating = defaultdict(lambda: (start_mean, start_var))
    player_rating = list()
    total_discrepancy = 0
    for game in np.arange(data.shape[0]):
        # check whether the player has play before
        current_players = player_double[game]
        for player in current_players:
            if player not in prior_rating:
                prior_rating[player]
        # collect prior rating in player_rating        
        player_rating.append(prior_rating.copy())
        
        new_rating = prior_rating.copy()
        # each player mean & variance
        winner1_mean,winner1_var = new_rating[current_players[0]]
        winner2_mean,winner2_var = new_rating[current_players[1]]
        loser1_mean,loser1_var = new_rating[current_players[2]]
        loser2_mean,loser2_var = new_rating[current_players[3]]
        # calculate team meam & variance
        winner_mean = sum([winner1_mean,winner2_mean])
        winner_var = sum([winner1_var,winner2_var])
        loser_mean = sum([loser1_mean,loser2_mean])
        loser_var = sum([loser1_var,loser2_var])
        # calculate c_iq
        c_iq = np.sqrt(winner_var + loser_var + 2 * beta_square)
        # calculate p_iq ,p_qi   p_iq是指預期比賽結果
        p_iq = np.exp(winner_mean/c_iq)/(np.exp(winner_mean/c_iq) + np.exp(loser_mean/c_iq))
        p_qi = np.exp(loser_mean/c_iq)/(np.exp(winner_mean/c_iq) + np.exp(loser_mean/c_iq))
        # sum discrepancy
        discrepancy = -np.log(p_iq)
        total_discrepancy += discrepancy
        # define s 比賽結果
        winner_s, loser_s = 1, 0
        # calculate delta_q, delta_i
        delta_q = winner_var/c_iq*(winner_s-p_iq)
        delta_i = loser_var/c_iq*(loser_s-p_qi)
        # calculate eta_q, eta_i
        eta_q = gamma_q * (np.sqrt(winner_var)/c_iq)**2 * p_iq * p_qi
        eta_i = gamma_q * (np.sqrt(loser_var)/c_iq)**2 * p_iq * p_qi
        # calculate omega_i, omega_q
        omega_i = delta_q
        omega_q = delta_i
        # calculate Delta_i, Delta_q
        Delta_i = eta_q
        Delta_q = eta_i
        # Indiviual skill update
        # update winner
        new_rating[current_players[0]] = (winner1_mean + winner1_var/winner_var*omega_i, winner1_var*np.max([1-winner1_var/winner_var*Delta_i,k]))
        new_rating[current_players[1]] = (winner2_mean + winner2_var/winner_var*omega_i, winner2_var*np.max([1-winner2_var/winner_var*Delta_i,k]))
        # update loser
        new_rating[current_players[2]] = (loser1_mean + loser1_var/loser_var*omega_q, loser1_var*np.max([1-loser1_var/loser_var*Delta_q,k]))
        new_rating[current_players[3]] = (loser2_mean + loser2_var/loser_var*omega_q, loser2_var*np.max([1-loser2_var/loser_var*Delta_q,k]))
        # update prior rating
        prior_rating = new_rating
    return player_rating, total_discrepancy
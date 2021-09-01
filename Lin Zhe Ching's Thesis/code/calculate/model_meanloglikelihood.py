import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
pd.set_option('display.max_columns', None)

def model_meanloglikelihood(Train, to_use, variance,beta_square):
    '''
    Args:
        player_rating: 根據之前訓練出的每個時段各選手的評分
        data: 資料集
    Return:
        loglikelihood : the loglikelihood of the model
    '''
    loglikelihood = 0
    players_rating = calculate_ratings(to_use,start_mean = 1500,start_var = variance ,beta_square = beta_square,k = 0.0001,gamma_q =1/2)
    for i in range(Train.shape[0],to_use.shape[0]):
        # get the current winners & losers name
        data_i = to_use.iloc[i,:]
        current_winner1, current_winner2 = data_i['winner1_name'], data_i['winner2_name']
        current_loser1, current_loser2 = data_i['loser1_name'], data_i['loser2_name']
        # get the current player the prior rating 
        winner1_rating, winner2_rating = players_rating[0][i][current_winner1][0],players_rating[0][i][current_winner2][0]
        loser1_rating, loser2_rating = players_rating[0][i][current_loser1][0],players_rating[0][i][current_loser2][0]
        # get the current player the prior variance 
        winner1_var, winner2_var = players_rating[0][i][current_winner1][1],players_rating[0][i][current_winner2][1]
        loser1_var, loser2_var = players_rating[0][i][current_loser1][1],players_rating[0][i][current_loser2][1]
        # get the team rating
        winner_mean = sum([winner1_rating, winner2_rating])
        winner_var = sum([winner1_var, winner2_var])
        loser_mean = sum([loser1_rating, loser2_rating])
        loser_var = sum([loser1_var, loser2_var])
        # get the c_iq
        c_iq = np.sqrt(winner_var + loser_var + 2 * beta_square)
        # Likelihood
        mp.dps = 30
        data_i_loglikelihood = mp.log((mp.exp(winner_mean/c_iq)/(mp.exp(winner_mean/c_iq) + mp.exp(loser_mean/c_iq))))
        # calculate loglikelihood
        loglikelihood +=data_i_loglikelihood
    return loglikelihood/(to_use.shape[0]-Train.shape[0])
def surface_meanloglikelihood(train, to_use, beta, variance_1, variance_2, variance_3, rho_12, rho_13, rho_23):
    '''
    Args:
    train: training dataset
    to_use: total dataset
    surface: The array of surfaces for each of N contest, of shape [N, 3]
    beta: The uncontral variances of the game
    variance_1: The initial variance of surface 1
    variance_2: The initial variance of surface 2
    variance_3: The initial variance of surface 3
    rho_12: The initial correlation coefficient between surface 1 and surface 2
    rho_13: The initial correlation coefficient between surface 1 and surface 3
    rho_23: The initial correlation coefficient between surface 2 and surface 3
    Return:
    mean_loglikelihood: the loglikelihood
    '''
    to_use_winners = to_use[['winner1_name','winner2_name']]
    to_use_losers = to_use[['loser1_name','loser2_name']]
    to_use_surface = encode_marks(to_use['surface'])[0]
    # 1. calculate rating 
    to_use_player_rating = calculate_ratings(winners= to_use_winners,
                                             losers= to_use_losers,
                                             surface= to_use_surface,
                                             beta= beta,
                                             variance_1=variance_1,
                                             variance_2= variance_2,
                                             variance_3=variance_3,
                                             rho_12= rho_12,
                                             rho_13= rho_13,
                                             rho_23= rho_23)
    surface_variance = np.array([[variance_1, rho_12 * np.sqrt(variance_1*variance_2), rho_13 * np.sqrt(variance_1*variance_3)],
                                [rho_12 * np.sqrt(variance_1*variance_2), variance_2, rho_23 * np.sqrt(variance_2*variance_3)],
                                [rho_13 * np.sqrt(variance_1*variance_3), rho_23 * np.sqrt(variance_2*variance_3), variance_3]])
    
    loglikelihood = 0 
    total = 0
    # 2. calculate the test loglikelihood 
    for game in np.arange(train.shape[0], to_use.shape[0]):
        # get the current winners & losers
        cur_winner_1 , cur_winner_2 = to_use_winners['winner1_name'][game], to_use_winners['winner2_name'][game]
        cur_loser_1, cur_loser_2 = to_use_losers['loser1_name'][game], to_use_losers['loser2_name'][game]
        # get the current surface
        cur_surface = to_use_surface[game]
        # get the winner & loser rating
        cur_player_rating = to_use_player_rating[0][game]
        cur_winner_1_rating, cur_winner_2_rating = cur_player_rating[cur_winner_1], cur_player_rating[cur_winner_2]
        cur_loser_1_rating, cur_loser_2_rating = cur_player_rating[cur_loser_1], cur_player_rating[cur_loser_2]
        # get the team rating
        winner_team = cur_winner_1_rating + cur_winner_2_rating
        loser_team = cur_loser_1_rating + cur_loser_2_rating
        # get the surface
        winner_surface = cur_surface.reshape((1,3)).dot(winner_team)[0][0]
        loser_surface = cur_surface.reshape((1,3)).dot(loser_team)[0][0]
        # c_ijk
        sigma_k = cur_surface.reshape((1,3)).dot(surface_variance).dot(cur_surface.reshape((3,1)))[0]
        c_ijk = np.sqrt(2*beta + sigma_k*2)[0]
        # loglikelihood
        data_i_likelihood = mp.log(mp.exp(winner_surface/c_ijk)/(mp.exp(winner_surface/c_ijk)+mp.exp(loser_surface/c_ijk)))
        loglikelihood += data_i_likelihood
        total+= 1
    meanloglikelihood = loglikelihood / total
    return meanloglikelihood
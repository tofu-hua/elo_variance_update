import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
pd.set_option('display.max_columns', None)

def rating_surface(train, to_use, beta, variance_1, variance_2, variance_3, rho_12, rho_13, rho_23):
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
    surface_rating: For each surface,winner's team rating & loser's team rating
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
    clay = []
    grass = []
    hard = []
    # 2. calculate the test accuracy 
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
        winner_surface = cur_surface.reshape((1,3)).dot(winner_team).item()
        loser_surface = cur_surface.reshape((1,3)).dot(loser_team).item()
        # check current surface
        if cur_surface.tolist() ==[1,0,0]:
            clay.append([winner_surface,loser_surface])
        elif cur_surface.tolist() ==[0,1,0]:
            grass.append([winner_surface,loser_surface])
        else:
            hard.append([winner_surface,loser_surface])
    surface_rating = [clay,grass,hard]
    return surface_rating
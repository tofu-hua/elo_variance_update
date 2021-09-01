import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
pd.set_option('display.max_columns', None)import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
pd.set_option('display.max_columns', None)
from BT_model import calculate_ratings
def accuracy(Train, to_use, variance):
    '''
    Args:
        player_rating: 根據之前訓練出的每個時段各選手的評分
        data: 資料集
    Return:
        accurate : the accuracy of the model
    '''
    count = 0
    total = 0
    players_rating = calculate_ratings(to_use,start_mean = 1500,start_var = variance ,beta_square = 3**2,k = 0.0001,gamma_q =1/2)
    for i in range(Train.shape[0],to_use.shape[0]):
        # get the current winners & losers name
        data_i = to_use.iloc[i,:]
        current_winner1, current_winner2 = data_i['winner1_name'], data_i['winner2_name']
        current_loser1, current_loser2 = data_i['loser1_name'], data_i['loser2_name']
        # get the current player the prior rating 
        winner1_rating, winner2_rating = players_rating[0][i][current_winner1][0],players_rating[0][i][current_winner2][0]
        loser1_rating, loser2_rating = players_rating[0][i][current_loser1][0],players_rating[0][i][current_loser2][0]
        # get the team rating
        winner_team = sum([winner1_rating, winner2_rating])
        loser_team = sum([loser1_rating, loser2_rating])
        # count the correct prediction
        if winner_team > loser_team:
            count +=1
        total +=1
    accurate = count / total
    return accurate
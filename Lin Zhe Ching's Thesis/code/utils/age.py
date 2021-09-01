import numpy as np
def age(Train,to_use):
    '''
    Return:
    winner_average_age
    loser_average_age
    '''
    winner_average_age = []
    loser_average_age = []
    for i in range(Train.shape[0],to_use.shape[0]):
        game_i = to_use.iloc[i,:]
        winner_1_age, winner_2_age = game_i['winner1_age'],game_i['winner2_age']
        loser_1_age, loser_2_age = game_i['loser1_age'],game_i['loser2_age']
        winner_team_age = (winner_1_age + winner_2_age)/2
        loser_team_age = (loser_1_age + loser_2_age)/2
        winner_average_age.append(winner_team_age)
        loser_average_age.append(loser_team_age)
    return winner_average_age, loser_average_age
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Rand_Walk as RW
import TD_Functions as TDF

def RMSE(True_Value,Learned_Values):
    diff = Learned_Values - True_Value
    diff_sq = diff**2
    avg = np.sum(diff_sq)/5
    return avg**.5

def history_to_trajectory(history,outcome):
    num_transitions = len(history) -1
    trajectory = np.zeros((num_transitions,3))
    for i in range(0,len(history)-1):
        trajectory[i,0] = history[i].index(1)
        trajectory[i,2] = history[i+1].index(1)
    if outcome[0] == 1:
        trajectory[len(trajectory)-1,1] =1.0
    return trajectory

True_Values = np.array([0.0, (1.0/6.0), (2.0/6.0), (3.0/6.0), (4.0/6.0), (5.0/6.0), 0.0])


lambdas_range = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
a_vals_range = np.arange(0,.66,step=.01)
best_alpha_values = []
lowest_errors = []

for l in lambdas_range:    
    a_lambda_list = []    
    for als in a_vals_range:
        avg_list = []
        for i in range(0,100):
            np.random.seed(i)
            game_tup = env.play_n_episodes(num_episodes=10)
            Learned_Values = TDF.TD_Lambda_Inc(game_tuples=game_tup,gamma=1.0,lambda_value=l,alpha=als)
            result = RMSE(True_Values,Learned_Values)
            avg_list.append(result)
        a_lambda_list.append(np.mean(avg_list))
    best_alpha_values.append(np.argmin(a_lambda_list))
    lowest_errors.append(np.min(a_lambda_list))
    
plt.figure(figsize=(10,7))
plt.plot(lambdas_range,lowest_errors,marker='.',linewidth=3,markersize=15)
plt.title('Average Training Set RMSE for Varying Lambda Using Optimal Alpha Hyper-Parameter',fontsize=16)
plt.xlabel('Lambda Value',fontsize=16)
plt.ylabel('Average Training Set RMSE',fontsize=16)
plt.show()        



optimal_alpha = []
for i in best_alpha_values:
    optimal_alpha.append(a_vals_range[i])
plt.figure(figsize=(10,7))
plt.plot(lambdas_range,optimal_alpha,marker='.',linewidth=3,markersize=15)
plt.title('Optimal Alpha for Varying Lambda',fontsize=16)
plt.xlabel('Lambda Value',fontsize=16)
plt.ylabel('Optimal Alpha Value',fontsize=16)
plt.show()
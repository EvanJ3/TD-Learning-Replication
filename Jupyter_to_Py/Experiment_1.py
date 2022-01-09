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

env = RW.Random_Walk_Env(.5)
test_lambda_values = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
output_dict = dict()
se_dict = dict()
for ls in test_lambda_values:
    rms_list = []
    for i in range(1,101):
        np.random.seed(i)
        game_tup = env.play_n_episodes(num_episodes=10)
        Learned_Values = TDF.TD_Lambda_Batch(game_tuples=game_tup,gamma=1.0,lambda_value=ls,alpha=0.01)
        result = RMSE(True_Values,Learned_Values)
        rms_list.append(result)
        
    rms_avg = np.mean(rms_list)
    rms_se = np.std(rms_list)/(len(rms_list)**.5)
    output_dict[ls] = rms_avg
    se_dict[ls] = rms_se
    
lambs = []
rms = []
for a,b in output_dict.items():
    lambs.append(a)
    rms.append(b)
    
plt.figure(figsize=(10,7))    
plt.plot(lambs,rms,marker='.',lw=3,markersize=15)
plt.title('Average RMSE of TD(lambda) on Training Set Under Repeated Presentations ',fontsize=16)
plt.xlabel('Lambda Values',fontsize=16)
plt.ylabel('Average RMSE',fontsize=16)
plt.show()

print('Standard Error = %s'%round(np.mean(list(se_dict.values())),4))

env = RW.Random_Walk_Env(.5)
np.random.seed(1)
game_tup = env.play_n_episodes(num_episodes=10)
vt, vtb = TDF.TD_Lambda_Batch_Weights(game_tuples=game_tup,gamma=1.0,lambda_value=0.0,alpha=.01)
vtb_array = np.array(vtb).T
vtb_array = vtb_array[1:,:]
vtb_array = vtb_array[:-1,:]
df = pd.DataFrame(vtb_array).T
df.columns = ['V(B)','V(C)','V(D)','V(E)','V(F)']
df.plot(figsize=(10,7))
plt.title('Weight Convergence of TD(0) Over Repeated Presentations',fontsize=16)
plt.axhline(y=5/6,color='black',linestyle='--')
plt.axhline(y=4/6,color='black',linestyle='--')
plt.axhline(y=3/6,color='black',linestyle='--')
plt.axhline(y=2/6,color='black',linestyle='--')
plt.axhline(y=1/6,color='black',linestyle='--')
plt.ylabel('State Value',fontsize=16)
plt.legend(fontsize=16)
plt.ylim((0,1))
plt.xlabel('Number of Repeated Presentations',fontsize=16)
plt.show()
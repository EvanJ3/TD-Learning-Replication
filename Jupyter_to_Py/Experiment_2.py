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


#TD Lambda
env = RW.Random_Walk_Env(.5)
test_lambda_values2 = [0.0,0.3,0.8,1.0]
test_alpha_values = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,.55,.6,]
output_lister = []
for ls in test_lambda_values2:
    rms_list = []
    for a in test_alpha_values:
        alpha_list = []
        for i in range(0,100):
            np.random.seed(i)
            game_tup = env.play_n_episodes(num_episodes=10)
            Learned_Values = TDF.TD_Lambda_Inc(game_tuples=game_tup,gamma=1.0,lambda_value=ls,alpha=a)
            result = RMSE(True_Values,Learned_Values)
            alpha_list.append(result)
        
        rms_avg = np.mean(alpha_list)
        rms_list.append(rms_avg)
    
    output_lister.append((ls,rms_list))
     
lambda_va = []
re_la = []
for i in output_lister:
    lambda_ = i[0]
    results = i[1]
    lambda_va.append(lambda_)
    re_la.append(results)
lambda_va = np.array(lambda_va)
re_la = np.array(re_la)

df = pd.DataFrame(re_la,lambda_va,columns=test_alpha_values).T
df.columns = ['Lambda Value = 0.0','Lambda Value = .3','Lambda Value = .8','Lambda Value = 1.0']  
wh = list(df['Lambda Value = 1.0'])
    
windrow_als = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,.4,.45,.5]
plt.figure(figsize=(10,7))
plt.plot(df.index,'Lambda Value = 0.0', data=df,marker='.', color='blue',label='Lambda = 0')
plt.plot(df.index,'Lambda Value = .3',data=df, marker='.', color='red',label='Lambda = .3')
plt.plot(df.index,'Lambda Value = .8',data=df, marker='.', color='green',label='Lambda = .8')
plt.plot(windrow_als,wh[0:11], marker='.', color='orange',label='Lambda = 1')
plt.legend(fontsize=16)
plt.xlabel('Alpha Value',fontsize=16)
plt.ylabel('Average Training Set RMSE',fontsize=16)
plt.ylim((.0,.7))
plt.title('Average RMSE After 10 Sequences for Varying Lambda and Alpha Values',fontsize=16)
plt.show()



env = RW.Random_Walk_Env(.5)
np.random.seed(1)
game_tup = env.play_n_episodes(num_episodes=100)
vt1, vtb1 = TDF.TD_Lambda_Inc_Weights(game_tuples=game_tup,gamma=1.0,lambda_value=1,alpha=.05)
vt2, vtb2 = TDF.TD_Lambda_Inc_Weights(game_tuples=game_tup,gamma=1.0,lambda_value=0,alpha=.05)
vtb_array1 = np.array(vtb1).T
vtb_array1 = vtb_array1[1:,:]
vtb_array1 = vtb_array1[:-1,:]
vtb_array2 = np.array(vtb2).T
vtb_array2 = vtb_array2[1:,:]
vtb_array2 = vtb_array2[:-1,:]

df1 = pd.DataFrame(vtb_array1).T
df2 = pd.DataFrame(vtb_array2).T


df1.columns = ['V(B)','V(C)','V(D)','V(E)','V(F)']
df2.columns = ['V(B)','V(C)','V(D)','V(E)','V(F)']

fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(14,7))
ax1.plot(df1)
ax1.set_title('State Value Estimates of TD(1)', fontsize=16)
ax2.plot(df2)
ax2.set_title('State Value Estimates of TD(0)', fontsize=16)
ax1.axhline(y=5/6,color='black',linestyle='--')
ax1.axhline(y=4/6,color='black',linestyle='--')
ax1.axhline(y=3/6,color='black',linestyle='--')
ax1.axhline(y=2/6,color='black',linestyle='--')
ax1.axhline(y=1/6,color='black',linestyle='--')
ax2.axhline(y=5/6,color='black',linestyle='--')
ax2.axhline(y=4/6,color='black',linestyle='--')
ax2.axhline(y=3/6,color='black',linestyle='--')
ax2.axhline(y=2/6,color='black',linestyle='--')
ax2.axhline(y=1/6,color='black',linestyle='--')
ax2.set_xlabel('Number of Episodes', fontsize=16)
#ax2.set_ylabel('State Values', fontsize=18)
ax1.set_xlabel('Number of Episodes', fontsize=16)
ax1.set_ylabel('State Values', fontsize=16)
ax1.legend(df1.columns)
ax2.legend(df2.columns)
plt.show()
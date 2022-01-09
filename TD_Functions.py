#Straight up implementation of TD Zero Used implementation of Figure 4 TD(lambda = 0) updates applied at the sequence level
#This implementation does not rely on convergence and is only intended to test experiement 2 where past episodes are only seen 
#The first time and we do not experience a repeated presentations paradigm
import numpy as np
import Rand_Walk as RW

def TD_Zero(gamma,max_episodes,alpha,seed):
    env = RW.Random_Walk_Env(.5)
    vt = [0.0,0.5,0.5,0.5,0.5,0.5,0.0]
    vt = np.array(vt)
    episodes = 1
    for i in range(0,max_episodes):
        #np.random.seed(seed)
        history,outcome = env.play_n_episodes(num_episodes=1)
        array_history = history_to_trajectory(history,outcome)
        weight_Vec = np.zeros((7))
        for j in range(0,array_history.shape[0]):
            state = int(array_history[j,0])
            reward = int(array_history[j,1])
            new_state = int(array_history[j,2])
            TD_Target = (reward + gamma * vt[new_state]) 
            TD_Error = TD_Target - vt[state]
            weight = alpha/(array_history.shape[0]**1/episodes) * TD_Error
            weight_Vec[state] += weight
        vt = vt + weight_Vec
        episodes+=1
    return vt




def TD_Lambda_Batch(game_tuples,gamma,lambda_value,alpha):
    episodes = game_tuples[0]
    outcomes = game_tuples[1]
    v_t = [0.0,0.5,0.5,0.5,0.5,0.5,0.0]
    v_t = np.array(v_t)
    alpha = alpha
    repeats = 1
    delta = np.inf
    while delta > 1e-4:
        episode_updates = []
        episode_count = 1
        for i in range(0,len(episodes)):
            history = episodes[i]
            outcome = [outcomes[i]]
            array_history = history_to_trajectory(history,outcome)
            el_trace = np.zeros((7))
            seq_weights_list = []
            step = 1
            for j in range(0,array_history.shape[0]):
                TD_tuple = array_history[j,:]
                el_trace[int(TD_tuple[0])] = el_trace[int(TD_tuple[0])] + 1.0
                State_Current_Value = v_t[int(TD_tuple[0])]
                Next_state_reward = TD_tuple[1]
                Next_state_Current_Value = v_t[int(TD_tuple[2])]
                TD_target = Next_state_reward + gamma * Next_state_Current_Value   
                TD_error = TD_target - State_Current_Value
                seq_weight = alpha * TD_error * el_trace
                episode_updates.append(seq_weight)
                el_trace = el_trace * gamma * lambda_value
                step+=1
            episode_count +=1
        epi_up_arr = np.sum(np.vstack(episode_updates),axis=0)
        v_t_new = v_t + epi_up_arr
        delta = (np.sum(np.square(v_t_new-v_t))/5)**.5
        v_t = v_t_new
        repeats +=1
    return v_t


def TD_Lambda_Inc(game_tuples,gamma,lambda_value,alpha):
    episodes = game_tuples[0]
    outcomes = game_tuples[1]
    v_t = [0.0,0.5,0.5,0.5,0.5,0.5,0.0]
    v_t = np.array(v_t)
    alpha = alpha
    episode_count = 1
    
    for i in range(0,len(episodes)):
        history = episodes[i]
        outcome = [outcomes[i]]
        array_history = history_to_trajectory(history,outcome)
        el_trace = np.zeros((7))
        weight_cum = []
        step_count = 1
        for j in range(0,array_history.shape[0]):
            TD_tuple = array_history[j,:]
            State_Current_Value = v_t[int(TD_tuple[0])]
            el_trace[int(TD_tuple[0])] = el_trace[int(TD_tuple[0])]+ 1
            Next_state_reward = TD_tuple[1]
            Next_state_Current_Value = v_t[int(TD_tuple[2])]
            TD_target = Next_state_reward + gamma * Next_state_Current_Value   
            TD_error = TD_target - State_Current_Value
            weight =  alpha/(array_history.shape[0]**1/episode_count) * TD_error * el_trace
            weight_cum.append(weight)
            el_trace = el_trace * gamma * lambda_value
            
        update_arr = np.sum(np.vstack(weight_cum),axis=0)
        v_t = v_t + update_arr
            
            
        episode_count +=1
    return v_t


def history_to_trajectory(history,outcome):
    num_transitions = len(history) -1
    trajectory = np.zeros((num_transitions,3))
    for i in range(0,len(history)-1):
        trajectory[i,0] = history[i].index(1)
        trajectory[i,2] = history[i+1].index(1)
    if outcome[0] == 1:
        trajectory[len(trajectory)-1,1] =1.0
    return trajectory


def TD_Lambda_Inc_Weights(game_tuples,gamma,lambda_value,alpha):
    episodes = game_tuples[0]
    outcomes = game_tuples[1]
    #v_t = [0.0,0.5,0.5,0.5,0.5,0.5,0.0]
    v_t = np.zeros((7))
    v_t = np.array(v_t)
    alpha = alpha
    episode_count = 1
    weight_buffer = []
    weight_buffer.append(v_t)
    for i in range(0,len(episodes)):
        history = episodes[i]
        outcome = [outcomes[i]]
        array_history = history_to_trajectory(history,outcome)
        el_trace = np.zeros((7))
        weight_cum = []
        step_count = 1
        for j in range(0,array_history.shape[0]):
            TD_tuple = array_history[j,:]
            State_Current_Value = v_t[int(TD_tuple[0])]
            el_trace[int(TD_tuple[0])] = el_trace[int(TD_tuple[0])]+ 1
            Next_state_reward = TD_tuple[1]
            Next_state_Current_Value = v_t[int(TD_tuple[2])]
            TD_target = Next_state_reward + gamma * Next_state_Current_Value   
            TD_error = TD_target - State_Current_Value
            weight =  alpha * TD_error * el_trace
            weight_cum.append(weight)
            el_trace = el_trace * gamma * lambda_value
            
        update_arr = np.sum(np.vstack(weight_cum),axis=0)
        v_t = v_t + update_arr
        weight_buffer.append(v_t)
            
        episode_count +=1
    return v_t, weight_buffer

def TD_Lambda_Batch_Weights(game_tuples,gamma,lambda_value,alpha):
    episodes = game_tuples[0]
    outcomes = game_tuples[1]
    #v_t = [0.0,0.5,0.5,0.5,0.5,0.5,0.0]
    #v_t = np.random.rand(7)
    v_t = np.zeros((7))
    #v_t = np.array(v_t)
    alpha = alpha
    repeats = 1
    delta = np.inf
    v_t_buffer = []
    v_t_buffer.append(v_t)
    while delta > 1e-4:
        episode_updates = []
        episode_count = 1
        for i in range(0,len(episodes)):
            history = episodes[i]
            outcome = [outcomes[i]]
            array_history = history_to_trajectory(history,outcome)
            el_trace = np.zeros((7))
            seq_weights_list = []
            step = 1
            for j in range(0,array_history.shape[0]):
                TD_tuple = array_history[j,:]
                el_trace[int(TD_tuple[0])] = el_trace[int(TD_tuple[0])] + 1.0
                State_Current_Value = v_t[int(TD_tuple[0])]
                Next_state_reward = TD_tuple[1]
                Next_state_Current_Value = v_t[int(TD_tuple[2])]
                TD_target = Next_state_reward + gamma * Next_state_Current_Value   
                TD_error = TD_target - State_Current_Value
                seq_weight = alpha * TD_error * el_trace
                episode_updates.append(seq_weight)
                el_trace = el_trace * gamma * lambda_value
                step+=1
            episode_count +=1
        epi_up_arr = np.sum(np.vstack(episode_updates),axis=0)
        v_t_new = v_t + epi_up_arr
        v_t_buffer.append(v_t_new)
        delta = (np.sum(np.square(v_t_new-v_t))/5)**.5
        v_t = v_t_new
        repeats +=1
    return v_t,v_t_buffer
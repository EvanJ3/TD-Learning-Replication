import numpy as np
class Random_Walk_Env():
    
    def __init__(self,move_right_probability):
        self.move_right_probability = move_right_probability
        self.move_left_probability = 1.0 - move_right_probability
        self.current_position = 3
        self.terminal_positions = [0,6]
        self.state_array = np.array([0,0,0,1,0,0,0])
        self.steps_taken = 0
        self.is_terminal = False
        self.move_history = [[0,0,0,1,0,0,0]]
        
    def reset_state(self):
        self.state_array = np.array([0,0,0,1,0,0,0])
        self.steps_taken = 0
        self.current_position = 3
        self.is_terminal = False
        self.move_history = [[0,0,0,1,0,0,0]]
    
    def set_state(self,new_state,new_steps_taken):
        assert new_state.shape() == self.state_array.shape()
        assert np.sum(new_state) == 1
        self.state_array = new_state
        self.steps_taken = new_steps_taken
    
    def get_current_position(self,state_array):
        
        return np.where(self.state_array==1)[0]
    
    def check_terminal(self):
        if self.current_position in self.terminal_positions:
            self.is_terminal = True
        else:
            pass
    
    def step(self):
        
        self.steps_taken +=1
        self.state_array[self.current_position] = 0
        random_number = np.random.rand()
        
        if random_number >= self.move_right_probability:
            self.current_position += 1
            self.state_array[self.current_position] = 1
            
        else:
            self.current_position -= 1
            self.state_array[self.current_position] = 1
        
        self.move_history.append(list(self.state_array))
        self.check_terminal()
        
    def play_episode(self):
        while not(self.is_terminal):
            self.step()
            
    def play_n_episodes(self,num_episodes):
        replay_buffer = []
        outcomes = []
        for i in range(0,num_episodes):
            self.play_episode()
            replay_buffer.append(self.move_history)
            if self.current_position == 0:
                outcomes.append(0)
            else:
                outcomes.append(1)
            self.reset_state()
        if num_episodes == 1:
            return replay_buffer[0],outcomes
        else:
            return replay_buffer,outcomes

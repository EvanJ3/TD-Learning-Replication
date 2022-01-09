# TD-Learning-Replication

<h1><center>CS 7642 Spring 2021 Project 1: Replicating Sutton 1988</center></h1>
<h2><center>Evan Jones</center></h2>
<h3><center>February 16, 2021</center></h3>

---

## Importing Standard and User Created Libraries:
We begin by import the modules necessary for the implementation of Sutton's code. Since we are using python we will import NumPy to handle array operations and construction, matplotlib.pyplot as graphing library, and pandas to easily convert arrays into a more manipulatable datatype for further input into matplotlib. In addition to these three-standard python libraries you will also notice I import two other libraries each of which is simply a ".py" file I created to house the separate parts of the implementation. Rand_Walk (RW) contains my implementation of the Random Walk environment as a class identical in function to the specified environment used in Sutton 1988. The main object of the Random_Walk class is the random walk environment object specified as Random_Walk(p=.5). Methods used in this library will also control all the "engine" behind the environment separate from our TD functions. The most used methods within this class are the step() and play_n_episodes() methods. Environmental resets and terminations are handled inside the environment itself see docstrings for further information on the functionality of this class. The libaray TD_Functions (TDF) is another ".py" file that contains my custom implementations of the TD($\lambda$) algorithm for the different use cases required under experiments 1,2, & 3. TD_Lambda_Batch() will be used in replicating experiment 1 and TD_Lambda_Inc() will be used in replicating experiments 2 and 3. Additionally, included in the TDF module is my implementation of TD(0) in a standalone fashion. This implementation was included to verify the results of TD($\lambda$). 

---

### Specification of Additional Helper Functions:

Next, I define two functions RMSE() and history_to_trajectory() which will both be used throughout the notebook to calculate RMSE between true and predicted state values and the trajectory function to better format the output of our random walk environment for use in incremental learning procedures by converting experience tuples to an array representation. We also declare the variable True_Values which is a vector representing the true state values recognized by Sutton for use in comparing the accuracy our of learning model outputs.

---

## Experiment 1: 

Below we implement Sutton's first experiment where we compare the batch TD($\lambda$) performance as measure by RMSE over varying values of lambda. For each value of lambda in the plot below we run the algorithm repeatedly on a training set of 10 episodes until convergence. We repeat this process one hundred times for each value of lambda so runtime may take a while. All 100 training sets are randomly initialized choosing a seed that best compares to Sutton, and the same training sets were used across all values of lambda to keep results directly comparable. Weight vectors are randomly initialized although due to repeated presentations this does not have any impact on the final results. Alpha value was set to be sufficiently small at so as to guarantee convergence at .01. The results below are as follows. 


### Results of Experiment 1 Repication:

![Experiment_1_Figure_1](/assets/Experiment_1_Figure_1.png)


### Compare to Sutton 1988 Figure 3:


![Sutton_1988_Figure_3](/assets/Sutton_1988_Figure_3.png)

## Batch Weight Updates Converge Under TD(0)

![Convergence_Batch_TD_Zero](/assets/Convergence_Batch_TD_Zero.png)


---

## Experiment 2:

Below I implemented Sutton's second experiment where we compute episodically updated TD($\lambda$) for lambda values of 0.0, 0.3, 0.8, & 1.0. Each lambda value is applied to 100 training sets of 10 episodes each for a variety of alpha values. The average RMSE of each respective hyper-parameter combination is plotted on the y-axis. As per Sutton 1988 weight vectors are initialized for each episode as .5 so as to void right or left side termination bias. The results are plotted below:

### Results of Experiment 2 Repication:

![Experiment_2_Figure_1](/assets/Experiment_2_Figure_1.png)

### Compare to Sutton 1988 Figure 4:

![Sutton_1988_Figure_4](/assets/Sutton_1988_Figure_4.png)

### TD(0) vs. TD(1) Episodic Weight Updates

![TD(0) vs. TD(1) Episodic](/assets/TD(0)_vs_TD(1).png)

---

## Experiment 3:

Below I implemented Sutton's third and final experiment where we compute episodically updated TD($\lambda$) for lambda values ranging from 0-1 by increments of .1. Each lambda value is applied to 100 training sets of 10 episodes each for optimal alpha values that minimized RMSE for a given lambda value. The average RMSE of each respective hyper-parameter combination is plotted on the y-axis. As per Sutton 1988 weight vectors are initialized for each episode as .5 to void right or left side termination bias. The results are plotted below:

### Results of Experiment 3 Repication:

![Experiment_3_Figure_1](/assets/Experiment_3_Figure_1.png)

### Compare to Sutton 1988 Figure 5:

![Sutton_1988_Figure_5](/assets/Sutton_1988_Figure_5.png)

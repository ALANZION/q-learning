# Q Learning Algorithm

## AIM
To develop a Python program to find the optimal policy for the given Reinforcement Learning environment using Q-Learning and comparing the state values with the First Visit Monte Carlo method.
## PROBLEM STATEMENT
For the given frozen lake environment, find the optimal policy applying the Q-Learning algorithm and compare the value functions obtained with that of First Visit Monte Carlo method. Plot graphs to analyse the difference visually.
## Q LEARNING ALGORITHM
# Step 1:
Store the number of states and actions in a variable, initialize arrays to store policy and action value function for each episode. Initialize an array to store the action value function.
# Step 2: 
Define function to choose action based on epsilon value which decides if exploration or exploitation is chosen.
# Step 3:
Create multiple learning rates and epsilon values.
# Step 4: 
Run loop for each episode, compute the action value function but in Q-Learning the maximum action value function is chosen instead of choosing the next state and next action's value. 
# Step 5:
Return the computed action value function and policy. Plot graph and compare with Monte Carlo results.

## Q LEARNING FUNCTION
### Name: ALAN ZION H
### Register Number: 212223240004
```
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:

        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```
## OUTPUT:
<img width="913" height="172" alt="image" src="https://github.com/user-attachments/assets/9c187cb3-bb46-4ff5-8323-72b167b876a4" />

<img width="1260" height="720" alt="image" src="https://github.com/user-attachments/assets/2be68c0d-5c35-49f0-9bae-4f07ad852a06" />

<img width="1600" height="656" alt="image" src="https://github.com/user-attachments/assets/cfd55c72-fe04-4804-9b82-8d4f165c4109" />

<img width="1600" height="659" alt="image" src="https://github.com/user-attachments/assets/55999afa-cba6-48d3-8cff-4b712793b813" />

## RESULT:
Thus the Reinforcement Learning environment using Q-Learning and comparing the state values with the First Visit Monte Carlo method hs been executed successfully.

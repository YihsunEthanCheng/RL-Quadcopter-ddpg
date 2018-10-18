from agents.ounoise import OUNoise
from agents.replayBuffer import ReplayBuffer
from agents.actor import Actor
from agents.critic import Critic
import numpy as np
import os, sys
import pandas as pd
import matplotlib.pyplot as plt

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, basename):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # learning rates 
        self.actor_learning_rate = 0.0001 
        self.critic_learning_rate = 0.001
        
        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, 
                                 self.action_low, self.action_high, 
                                 self.actor_learning_rate)
        self.actor_target = Actor(self.state_size, self.action_size,
                                  self.action_low, self.action_high, 
                                  self.actor_learning_rate)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size,
                                   self.critic_learning_rate)
        self.critic_target = Critic(self.state_size, self.action_size, 
                                    self.critic_learning_rate)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, 
                             self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 1000000
        self.batch_size = 128
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters
        
        # keep track of the best run
        self.nEpisode = 0
        self.bestEpisode = []
        self.bestEpisodeAt = -1
        
        # logging business
        self.state_labels = self.task.get_state_labels()
        self.action_labels = [ 'ac{}'.format(i) for i in range(self.action_size) ]
        self.df_columns = ['t'] + self.state_labels.tolist() + self.action_labels + ['R']
        self.basename = os.path.join('log',basename)
        self.currentEpisode = []
        self.bestCumReward = -np.inf

    def reset_episode(self):
        self.noise.reset()
        self.last_state = self.task.reset()
        self.currentEpisode = []
        return self.last_state
          
    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]
                          ).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]
                          ).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients(
            [states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights),\
            "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def step(self, action):
        last_state_variables = self.task.get_state_variables()
        last_t = self.task.sim.get_time()
        
        # call the model for state transition
        next_state, reward, done = self.task.step(action)
        
        # logging the current episode
        self.currentEpisode += [ np.hstack([ last_t, last_state_variables, action,  reward]) ]
        
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

        if done:
            # log the episode
            df = pd.DataFrame(self.currentEpisode, columns = self.df_columns)
            fn_i = '{}_{}'.format(self.basename, self.nEpisode)
            df.to_csv(fn_i + '.csv')
            cumR = df.R.sum()        
            if len(df) > len(self.bestEpisode) or \
            (len(df) == len(self.bestEpisode) and cumR > self.bestCumReward):
                self.bestCumReward = cumR
                self.bestEpisode = df
                self.bestEpisodeAt = self.nEpisode
                self.plot_episode(df, self.nEpisode, fn_i)
            sys.stdout.write("\rEp#{:4d} dur_{} cumR_{:5.3f} best@{} dur_{} cumR_{:5.3f} "
                .format(self.nEpisode,len(self.bestEpisode), cumR, self.bestEpisodeAt, 
                len(self.bestEpisode), self.bestCumReward))
            self.nEpisode += 1
        return next_state, done

    def train(self, num_episodes = 1):
        for ep_i in range(num_episodes):  
            state, done = self.reset_episode(), False
            while not done:
                action = self.act(state) 
                state, done = self.step(action)
    
    def plot_episode(self, df, episNo, filename = ''):
        fig = plt.figure(1)
        fig.clf()
        ax2 = fig.add_subplot(313)
        ax1 = fig.add_subplot(312, sharex = ax2)
        ax0 = fig.add_subplot(311, sharex = ax2)
        # plot selected state variables        
        ax0.set_title('Ep#{} dur={:5.2f} sec'.format(episNo,df.t.iloc[-1]))
        df.plot(x = 't', y = self.state_labels[:6], ax = ax0,style = '.:')
        df.plot(x = 't', y = self.state_labels[6:], ax = ax1, style = '.:')
        df.plot(x = 't', y = self.action_labels, ax = ax2, style = '.:')        
        df.plot(x = 't', y = 'R', ax=ax2, secondary_y=True)
        plt.ylabel('Reward')
        plt.show()
        if len(filename) > 0:
            fig.savefig(filename)
            
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV

import random
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
First, we need to define buffer allowing us
to sample data form the environment with an iid assumption.
We will store all the samples in a training set with a distribution
that is close to the distribution of the policy we are trying to learn, 
and independently sample from this training set to train the model.

We will use the ReplayBuffer class introduced in the notebook RL4 (FIFO mechanism).
'''

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    

'''
Our model will build an approximate value iteration function using neural networks.
For this, we implemented a very simple Deep Q-Network (DQN) model described in 'Playing Atari with Deep Reinforcement Learning' by Mnih et al. (2013).
The model takes epsilon-greedy actions, with a decreasing epsilon over time, stores the samples in a replay buffer, 
and at each interaction compute the target values of a drawn mini-batch to take a gradient step.

The model was selected through a hyperparameter search, starting from the base model given in the RL4 notebook
We increased epsilon decay period to 1000 to increase exploration at the beginning of the training, (wait 1000 steps before decay)
and did the same with the epsilon delay to 100 (20 was too fast). 

The best tested results we got with a simple neural network with 5 hidden layers, 
starting at 256 neurons and increasing by a factor of 2 at each layer, the last layers being of size 1024
(the number of neurons are chosen as multiples of 2 to allow for better parallelization on GPUs).
The batch sizes were increased to 512 to allow for better generalization (as the problem is more complicated than cartpole), 
and the number of gradient steps was increased to 3 to allow for better convergence
The training was done on 200 episodes, but the best models were obtained after 100 episodes in general.
'''
    

class ProjectAgent:
    def act(self, observation, use_random=False):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        cpu_device = torch.device("cpu")
        DQN = self.get_DQN()
        self.model = DQN.to(cpu_device)
        self.model.load_state_dict(torch.load(self.path, map_location=cpu_device, weights_only=True))
        self.model.eval()
        return

    def get_config(self):
      # Setting the configuration dictionnary for the model
      self.config = {'n_action': env.action_space.n,
          'state_dim': env.observation_space.shape[0],
          'learning_rate': 0.001,
          'gamma': 0.99, 
          'buffer_size': 100000, # We get a smaller buffer size memory to not overload the computer
          'epsilon_min': 0.01,
          'epsilon_max': 1., # Pure random exploration at the beginning
          'epsilon_decay_period': 10000,
          'epsilon_delay_decay': 100,
          'batch_size': 512,
          'gradient_steps': 3, 
          'update_target_freq': 200,
          'criterion': torch.nn.SmoothL1Loss()}
      return

    def get_DQN(self):
        # Defining an instance of the DQN model tested
        state_dim = self.config['state_dim']
        n_action = self.nb_actions
        nb_neurons = 256
        DQN = torch.nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons*2),
            nn.ReLU(),
            nn.Linear(nb_neurons*2, nb_neurons*4),
            nn.ReLU(),
            nn.Linear(nb_neurons*4, nb_neurons*8),
            nn.ReLU(),
            nn.Linear(nb_neurons*8, n_action)
        ).to(DEVICE)
        return DQN

    def __init__(self):
        # Simple initialization of the agent with the configuration
        self.get_config()
        self.path = "src/model.pt"
        self.nb_actions = self.config['n_action']
        self.model = self.get_DQN()
        self.gamma = self.config['gamma']
        self.batch_size = self.config['batch_size']
        buffer_size = self.config['buffer_size']
        self.memory = ReplayBuffer(buffer_size,DEVICE)
        self.epsilon_max = self.config['epsilon_max']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_stop = self.config['epsilon_decay_period']
        self.epsilon_delay = self.config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.target_model = deepcopy(self.model).to(DEVICE)
        self.criterion = self.config['criterion']
        lr = self.config['learning_rate']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) # Standard optimizer
        self.nb_gradient_steps = self.config['gradient_steps']
        # Using the replace strategy
        # replacing the target network every  200 steps by the current model
        self.update_target_freq = self.config['update_target_freq']

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            # run through the target model
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def gradient_step_double(self):
        '''
        We test the Double DQN model from  Deep Reinforcement Learning with Double Q-Learning by Hasselt et al. (2015)
        The difference lies in how the target values are computed for the update
        Instead of using argmax Q_target for action selection, we use the current model to select the highest Q value action
        The action is then evaluated using the target model like before

        The results were not as good as the standard DQN model, so we kept the standard model for the final training.
        '''
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            next_actions = self.model(Y).argmax(1).detach()
            QYmax = self.target_model(Y).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, max_episode=200):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        # Keep the score on the chosen patient
        # to get the model with the best result
        score = 0
        while episode < max_episode:
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act(state)

            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            if step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            step += 1
            if done or trunc:
                episode += 1
                val_score = evaluate_HIV(agent=self, nb_episode=1)
                print("Episode ", '{:3d}'.format(episode),
                      ", epsilon ", '{:6.2f}'.format(epsilon),
                      ", batch size ", '{:5d}'.format(len(self.memory)),
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      ", Evaluation score  ", '{:2e}'.format(val_score),
                      sep='')
                state, _ = env.reset()
                if val_score > score:
                  score = val_score
                  self.best_model = deepcopy(self.model).to(DEVICE)
                  self.save(self.path)
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return

'''
# Code for the training loop (training done on google collab):

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

agent = ProjectAgent()
episode_return = agent.train(env)
'''
import os
import random
import torch
import argparse

import numpy as np
import torch.nn as nn

from copy import deepcopy
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm

from utils import load_yaml_into_namespace
from env_hiv import HIVPatient
from evaluate import evaluate_HIV
from models import DQN

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.data = []
        self.index = 0

    def append(self, state, action, reward, next_state, done):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return [torch.tensor(np.array(x), device=self.device) for x in zip(*batch)]

    def __len__(self):
        return len(self.data)
    

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(
        self, 
        env, 
        gamma, 
        batch_size, 
        epsilon_max, 
        epsilon_min, 
        epsilon_decay_period, 
        update_target_freq, 
        learning_rate,
        buffer_size,
        model_hidden_dim=256,
        model_num_layers=3,
        model_dropout_prob=0.2
    ):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon_max - epsilon_min) / epsilon_decay_period
        self.update_target_freq = update_target_freq
        self.model_hidden_dim = model_hidden_dim
        self.model_num_layers = model_num_layers
        self.model_dropout_prob = model_dropout_prob
        self.memory = ReplayBuffer(buffer_size, self.device)

        self.model = DQN(self.state_dim, self.action_dim, self.model_hidden_dim, self.model_num_layers, self.model_dropout_prob).to(self.device)
        self.target_model = deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

    def act(self, state, greedy=False):
        if not greedy and np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state_tensor)).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def train(self, max_episodes):
        state, _ = self.env.reset()
        step = 0
        best_score = -float('inf')
        for episode in tqdm(range(max_episodes)):
            episode_reward = 0
            while True:
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                episode_reward += reward
                self.gradient_step()

                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                if done:
                    validation_score = evaluate_HIV(self, nb_episode=1)
                    print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Validation Score: {validation_score:.2f}")
                    if validation_score > best_score:
                        best_score = validation_score
                        self.save('best_model.pth')
                    break
                state = next_state
                step += 1
            state, _ = self.env.reset()

    def gradient_step(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Compute target Q-values
        with torch.no_grad():
            target_q_values = rewards + (~dones) * self.gamma * self.target_model(next_states).max(1)[0]

        # Compute current Q-values
        current_q_values = self.model(states).gather(1, actions.long().unsqueeze(1)).squeeze()

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to the config file",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    args = load_yaml_into_namespace(args.config, args)

    # Initialize and train the agent
    agent = ProjectAgent(
        env=env, 
        gamma=args.gamma, 
        batch_size=args.batch_size, 
        epsilon_max=args.epsilon_max, 
        epsilon_min=args.epsilon_min, 
        epsilon_decay_period=args.epsilon_decay_period, 
        update_target_freq=args.update_target_freq, 
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        model_hidden_dim=args.model_hidden_dim,
        model_num_layers=args.model_num_layers,
        model_dropout_prob=args.model_dropout_prob,
    )
    agent.train(max_episodes=args.max_train_episodes)

import gym
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import collections
from torch.optim.lr_scheduler import StepLR
import contextlib
from typing import Tuple, Optional, Any, Dict, Sequence, List
import random
import logging
from metagamer.environments.tictactoe import TicTacToeEnv
from metagamer.agents.qagent import Agent, QtableWrapper

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("tictrainer")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """
    The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
    """

    def __init__(self, action_dim, state_dim, hidden_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1


class Memory:
    """
    memory to save the state, action, reward sequence from the current episode.
    """

    def __init__(self, len):
        self.rewards = collections.deque(maxlen=len)
        self.state = collections.deque(maxlen=len)
        self.action = collections.deque(maxlen=len)
        self.is_done = collections.deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.is_done)
        idx = random.sample(range(0, n - 1), batch_size)

        return (
            torch.Tensor(self.state)[idx].to(device),
            torch.LongTensor(self.action)[idx].to(device),
            torch.Tensor(self.state)[1 + np.array(idx)].to(device),
            torch.Tensor(self.rewards)[idx].to(device),
            torch.Tensor(self.is_done)[idx].to(device),
        )

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()


class DQTicTacTable(Agent):
    def wrapper(self, env):
        return QtableWrapper(env)

    def __init__(self, hidden_dim):
        super().__init__()
        env = TicTacToeEnv()
        Q_1 = QNetwork(
            action_dim=env.action_space.n,
            state_dim=env.observation_space.shape[0],
            hidden_dim=hidden_dim,
        ).to(device)
        Q_2 = QNetwork(
            action_dim=env.action_space.n,
            state_dim=env.observation_space.shape[0],
            hidden_dim=hidden_dim,
        ).to(device)

    def get_action(self, model, state):
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = model(state)

        # select a random action wih probability eps
        if random.random() <= self.epsilon:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(values.cpu().numpy())

        return action

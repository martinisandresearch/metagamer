import gym
import torch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import collections
from torch.optim.lr_scheduler import StepLR
import contextlib
from typing import Tuple, Optional, Any, Dict, Sequence, List
import random
import logging
from metagamer.environments.tictactoe import TicTacToeEnv
from metagamer.agents.qagent import Agent, ModWrapper, QtableWrapper

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

    def update(self, state, action, reward):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(False)

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


def evaluate(Qmodel, env, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()
    perform = 0
    for _ in range(repeats):
        state = env.reset()
        done = False
        while not done:
            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                values = Qmodel(state)
            action = np.argmax(values.cpu().numpy())
            state, reward, done, _ = env.step(action)
            perform += reward
    Qmodel.train()
    return perform / repeats


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def train_network(batch_size, current, target, optim, memory, gamma):
    optim.zero_grad()

    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(
        1, torch.max(next_q_values, 1)[1].unsqueeze(1)
    ).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    loss.backward()
    optim.step()


class DQWrapper(ModWrapper):
    def observation(self, observation: np.array) -> torch.Tensor:
        flatboard = observation.reshape(-1).astype(np.float32)
        my_tensor = torch.from_numpy(flatboard)
        return my_tensor

    def action(self, action: int) -> Tuple[int, int]:
        new_action = tuple(np.unravel_index(action, self.unwrapped.board.shape))
        return new_action

    def reverse_action(self, action: Tuple[int, int]) -> int:
        return self.unwrapped.board.shape[1] * action[0] + action[1]


class DQTicTacTable(Agent):
    def wrapper(self, env):
        return DQWrapper(env)

    def __init__(
        self, hidden_dim=3, lr=0.2, lr_gamma=0.9, lr_step=10, max_memory_size=50000
    ):
        super().__init__()
        env = TicTacToeEnv()
        self.Q_1 = QNetwork(
            # Should improve these to acquire information from environment better
            action_dim=9,
            state_dim=9,
            hidden_dim=hidden_dim,
        ).to(device)
        self.Q_2 = QNetwork(
            action_dim=9,
            state_dim=9,
            hidden_dim=hidden_dim,
        ).to(device)
        update_parameters(self.Q_1, self.Q_2)
        # we only train Q_1
        for param in self.Q_2.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.Q_1.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=lr_step, gamma=lr_gamma)
        self.gamma = 0.99
        self.memory = Memory(max_memory_size)
        self.performance = []

    def get_action(self, state, valid_actions, **kwargs):

        # Unpack the model, fed in as kwarg.  This is clunky, should be reworked, as model is required
        model = kwargs["model"]

        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = model(state)

        # If we're just evaluating, not training, just get the best available
        if self.eval:
            return np.argmax(values.cpu().numpy())

        # select a random action wih probability epsilon
        if random.random() <= self.epsilon:
            action = random.choice(valid_actions)
        else:
            action = np.argmax(values.cpu().numpy())
        return action

    def set_reward(self, state: str, action: int, value: float):
        """
        Only allowed to set a state, value pair.
        Update is as per alpha value
        """
        if self.eval:
            # no learning in eval mode
            return
        self.memory.update(state, action, value)

    def get_max(self, state, **kwargs):
        """Find the maximum reward that the Deep Q network thinks is possible from this state"""

        # Unpack the model, fed in as kwarg.  This is clunky, should be reworked, as model is required
        model = kwargs["model"]

        state = torch.Tensor(state).to(device)

        with torch.no_grad():
            values = model(state)

        return np.max(values.cpu().numpy())

    def get_discounted_max(self, state, **kwargs):
        """Here the discount can be provided for q value adaptations"""

        return self.discount * self.get_max(state, **kwargs)


class DTicTacToeRunner:
    def __init__(self, agent1: DQTicTacTable, agent2: DQTicTacTable):
        self.env = TicTacToeEnv()
        # player order
        self.agent1 = agent1
        self.agent2 = agent2

        # interact with the same environment, but through their preferred representation
        self.agent1_env = self.agent1.wrapper(self.env)
        # player 2 wrapper multiplies reward by -1
        self.agent2_env = self.agent2.wrapper(self.env)

        self.measure_step = 50
        self.measure_repeats = 100
        self.min_episodes = 100
        self.update_step = 5
        self.update_repeats = 10
        self.batch_size = 1

    def train(self, num_episodes):
        """"""

        results = np.zeros(num_episodes)
        game_len = np.zeros(num_episodes)

        for i in range(num_episodes):
            self.env.reset()
            state = self.env.reset()
            self.agent1.memory.state.append(state)
            self.agent2.memory.state.append(state)
            # run first two moves

            p1_state = self.agent1_env.get_observation()
            p1_action = self.agent1.get_action(
                p1_state, self.agent1_env.valid_actions, model=self.agent1.Q_2
            )
            _, p1_reward, done, info = self.agent1_env.step(p1_action, 1)
            logger.debug(f"Training: p1_action: {p1_action}")

            while True:
                p2_state = self.agent2_env.get_observation()
                p2_action = self.agent2.get_action(
                    p2_state, self.agent2_env.valid_actions, model=self.agent2.Q_2
                )
                logger.debug(f"Training: p2_action: {p2_action}")

                if p2_action in self.agent2_env.valid_actions:
                    _, reward, done, info = self.agent2_env.step(p2_action, -1)
                else:
                    reward = 1
                    done = True
                    info = {}

                if not done:
                    p1_qscore = reward + self.agent1.get_discounted_max(
                        self.agent1_env.get_observation(), model=self.agent1.Q_2
                    )
                    self.agent1.set_reward(p1_state, p1_action, p1_qscore)

                else:
                    self.agent1.set_reward(p1_state, p1_action, reward)
                    self.agent1.memory.is_done.append(True)
                    self.agent2.set_reward(p2_state, p2_action, reward * -1)
                    self.agent2.memory.is_done.append(True)
                    break

                p1_state = self.agent1_env.get_observation()
                p1_action = self.agent1.get_action(
                    p1_state, self.agent1_env.valid_actions, model=self.agent1.Q_2
                )
                logger.debug(f"Training: p1_action: {p1_action}")

                if p1_action in self.agent1_env.valid_actions:
                    _, reward, done, info = self.agent1_env.step(p1_action, 1)
                else:
                    reward = -1
                    done = True
                    info = {}

                if not done:
                    p2_qscore = reward * -1 + self.agent2.get_discounted_max(
                        self.agent2_env.get_observation(), model=self.agent2.Q_2
                    )
                    self.agent2.set_reward(p2_state, p2_action, p2_qscore)
                else:
                    self.agent1.set_reward(p1_state, p1_action, reward)
                    self.agent1.memory.is_done.append(True)
                    self.agent2.set_reward(p2_state, p2_action, reward * -1)
                    self.agent2.memory.is_done.append(True)
                    break

            game_len[i] = self.agent1_env.unwrapped.turns_played
            results[i] = reward
            logger.debug("Result is: %s", reward)
            if i >= self.min_episodes and i % self.update_step == 0:
                for _ in range(self.update_repeats):
                    train_network(
                        self.batch_size,
                        self.agent1.Q_1,
                        self.agent1.Q_2,
                        self.agent1.optimizer,
                        self.agent1.memory,
                        self.agent1.gamma,
                    )
                    train_network(
                        self.batch_size,
                        self.agent2.Q_1,
                        self.agent2.Q_2,
                        self.agent2.optimizer,
                        self.agent2.memory,
                        self.agent2.gamma,
                    )

                # transfer new parameter from Q_1 to Q_2
                update_parameters(self.agent1.Q_1, self.agent1.Q_2)
                update_parameters(self.agent2.Q_1, self.agent2.Q_2)

                # update learning rate and eps
            self.agent1.scheduler.step()
            self.agent2.scheduler.step()
        print(
            f"games {num_episodes}"
            f" p1: {np.sum(results == 1):.2f}"
            f" p2: {np.sum(results == -1):.2f}"
            f" draws: {np.sum(results == 0):.2f}"
            f" num_turns: {np.average(game_len):.2f}"
        )

        return results

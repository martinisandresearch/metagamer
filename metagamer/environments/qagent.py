import contextlib
from typing import Tuple, Optional, Any, Dict, Sequence, List

import random
import collections
import logging

import gym
import numpy as np

from metagamer.environments import tictactoe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tictrainer")


def exploit(epsilon):
    """exploitation increases as epsilon decreases"""
    return random.random() > epsilon


class ModWrapper(gym.Wrapper):
    """A convenience wrapper that allows modifying the state and action """

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action, *args, **kwargs):
        observation, reward, done, info = self.env.step(
            self.action(action), *args, **kwargs
        )
        return self.observation(observation), reward, done, info

    def get_observation(self):
        return self.observation(self.env.get_observation())

    @property
    def valid_actions(self) -> List[Any]:
        return [self.reverse_action(a) for a in self.env.valid_actions]

    # to be overwritten

    def action(self, action):
        return action

    def reverse_action(self, action):
        """necessary for valid action"""
        return action

    def observation(self, observation):
        return observation


class QtableWrapper(ModWrapper):
    """
    An TicTacTo subclass environment adapted to Q learning
    Agents define their own wrappers which define the interfacr we expect


    State is represented as a string
    Action is represented as an index between 0-8
    """

    def observation(self, observation: np.array) -> str:
        return "".join([self.unwrapped.SYMBOLS[i] for i in observation.flatten()])

    def action(self, action: int) -> Tuple[int, int]:
        new_action = tuple(np.unravel_index(action, self.unwrapped.board.shape))
        return new_action

    def reverse_action(self, action: Tuple[int, int]) -> int:
        return self.unwrapped.board.shape[1] * action[0] + action[1]


class Agent:
    """
    A Q learning focussed Agent interface
    The intention is to have both deterministic and trained agents interact via this API
    It has in built epsilon-greedy, and the discounting and updating of the q values
    is done via the api as well as a train/eval mode

    This may be insufficient for policy gradients etc, but we can cross that bridge then.
    """

    def __init__(self):
        self.train = True
        self.epsilon = 0.9
        self.discount = 0.9

    def wrapper(self, env):
        """provide a function that wraps a given environment into it's prefferred representation"""
        return env

    @property
    def eval(self) -> bool:
        return not self.train

    @eval.setter
    def eval(self, val):
        self.train = not val

    @contextlib.contextmanager
    def exploit_mode(self):
        curr_train = self.train
        try:
            self.train = False
            yield self
        finally:
            self.train = curr_train

    def get_action(self, state, valid_actions):
        """Epsilon greedy when in training mode. Override for determinstic agents"""
        if self.eval or exploit(self.epsilon):
            return self.get_arg_max(state, valid_actions)
        return random.choice(valid_actions)

    def get_arg_max(self, state, valid_actions):
        """used to play, override this or get_action"""
        raise NotImplementedError

    def get_max(self, state, valid_actions=None):
        """used to train and update rewards - set to max of environment for determenistic environments
        as this won't be used"""
        return 1

    def get_discounted_max(self, state, valid_actions=None):
        """Here the discount can be provided for q value adaptations"""
        return self.discount * self.get_max(state, valid_actions)

    def set_reward(self, state, action, reward):
        """update when training - unused by determenistic environments"""
        return


class Random(Agent):
    def get_action(self, state, valid_actions):
        return random.choice(valid_actions)


class PageLines(Agent):
    def get_action(self, state, valid_actions):
        """
        return a possible position that hasn't yet been played.
        This policy tries simply to fill the board from the top left.
        Args:
            state: a 3*3 numpy array whose elements are 0, 1, -1 representing empty, player1 and player 2,
            valid_actions: either 1 or -1, representing player 1 and player 2


        Returns: a tuple representing an unfilled coordinate that's selected.

        """
        for row in range(state.shape[0]):
            for col in range(state.shape[1]):
                if state[row, col] == 0:
                    return row, col


class EpsilonPageLines(Agent):
    def get_arg_max(self, state, valid_actions):
        """
        return a possible position that hasn't yet been played.
        This policy tries simply to fill the board from the top left.
        Args:
            board: a 3*3 numpy array whose elements are 0, 1, -1 representing empty, player1 and player 2,
            player: either 1 or -1, representing player 1 and player 2


        Returns: a tuple representing an unfilled coordinate that's selected.

        """
        for row in range(state.shape[0]):
            for col in range(state.shape[1]):
                if state[row, col] == 0:
                    return row, col

    # qtable
    # random
    # dqn
    # policy gradient
    # left to right


class QTicTacTable(Agent):
    def wrapper(self, env):
        return QtableWrapper(env)

    def __init__(self, alpha=0.1, gamma=0.9):
        """
        this will contain the state
        qdict[state][action] = qvalue
        where state is a string representing the board and action is an integer
        """
        # choice to prune or build
        # we choose prune here since it's less likely to go wrong
        super().__init__()
        self.qdict: Dict[str, Dict[int, float]] = collections.defaultdict(
            lambda: {i: 0 for i in range(9)}
        )
        # update decay - smaller alphas have longer memory, larger alphas shorter
        self.alpha = alpha
        # discount factor when updating reward - high gamma is more greedy, low gamma is more patient
        self.discount = gamma

    def set_reward(self, state: str, action: int, value: float):
        """
        Only allowed to set a state, value pair.
        Update is as per alpha value
        """
        self.qdict[state][action] = (
            self.qdict[state][action] * (1 - self.alpha) + value * self.alpha
        )

    def prune_qdict(self, state: str, valid_actions: Sequence[int]):
        """Optional function that allows us to reduce space"""
        if set(valid_actions) - self.qdict[state].keys():
            raise ValueError("We have found valid actions not in our qtable for state")
        elif self.qdict[state].keys() - set(valid_actions):
            # we can prune down since we don't need the other moves
            self.qdict[state] = {k: self.qdict[state][k] for k in valid_actions}

    def get_max(self, state: str, valid_actions: Optional[Sequence[int]] = None):
        """get maximum q score for a state over actions"""
        if valid_actions:
            self.prune_qdict(state, valid_actions)
        return max(self.qdict[state].values())

    def get_arg_max(self, state, valid_actions: Optional[Sequence[int]] = None):
        """
        which action gives max q score. If identical max q scrores exist,
           this'll choose somehing at random
        """
        mx = self.get_max(state, valid_actions)
        # get all moves that return optimal outcome
        all_arg_maxes = [k for k, v in self.qdict[state].items() if v == mx]
        return random.choice(all_arg_maxes)


class TicTacToeRunner:
    def __init__(self, agent1: Agent, agent2: Agent):
        self.env = tictactoe.TicTacToeEnv()
        # player order
        self.agent1 = agent1
        self.agent2 = agent2

        # interact with the same environment, but through their preferred representation
        self.agent1_env = self.agent1.wrapper(self.env)
        # player 2 wrapper multiplies reward by -1
        self.agent2_env = self.agent2.wrapper(self.env)

    def train(self, num_epsiodes):
        """"""

        results = np.zeros(num_epsiodes)
        eps_vals = np.zeros(num_epsiodes)

        for i in range(num_epsiodes):
            self.env.reset()

            # run first two moves

            p1_state = self.agent1_env.get_observation()
            p1_action = self.agent1.get_action(p1_state, self.agent1_env.valid_actions)
            _, p1_reward, done, info = self.agent1_env.step(p1_action, 1)

            while True:

                p2_state = self.agent2_env.get_observation()
                p2_action = self.agent2.get_action(
                    p2_state, self.agent2_env.valid_actions
                )
                _, reward, done, info = self.agent2_env.step(p2_action, -1)

                if not done:
                    p1_qscore = reward + self.agent1.get_max(
                        self.agent1_env.get_observation()
                    )
                    self.agent1.set_reward(p1_state, p1_action, p1_qscore)

                else:
                    self.agent1.set_reward(p1_state, p1_action, reward)
                    self.agent2.set_reward(p2_state, p2_action, reward * -1)
                    break

                p1_state = self.agent1_env.get_observation()
                p1_action = self.agent1.get_action(
                    p1_state, self.agent1_env.valid_actions
                )
                _, reward, done, info = self.agent1_env.step(p1_action, 1)

                if not done:
                    p2_qscore = reward * -1 + self.agent2.get_max(
                        self.agent2_env.get_observation()
                    )
                    self.agent2.set_reward(p2_state, p2_action, p2_qscore)
                else:
                    self.agent1.set_reward(p1_state, p1_action, reward)
                    self.agent2.set_reward(p2_state, p2_action, reward * -1)
                    break

            results[i] = reward

        print(
            f"{num_epsiodes} games crosses: {np.sum(results == 1):.2f}"
            f" naughts: {np.sum(results == -1):.2f}"
            f" draws: {np.sum(results == 0):.2f}"
        )

        return results, eps_vals

    def run(self):
        done = False
        with self.agent1.exploit_mode(), self.agent2.exploit_mode():
            self.env.reset()
            while not done:
                p1_state = self.agent1_env.get_observation()
                logger.info(
                    "p1_state: '%s', valid_actions: %s",
                    p1_state,
                    self.agent1_env.valid_actions,
                )
                p1_action = self.agent1.get_action(
                    p1_state, self.agent1_env.valid_actions
                )
                logger.info("p1_action: %s", p1_action)

                _, reward, done, _ = self.agent1_env.step(p1_action, 1)
                logger.info("p1 reward: %s, done: %s", reward, done)

                self.env.render("human")

                if done:
                    break

                p2_state = self.agent2_env.get_observation()
                p2_action = self.agent2.get_action(
                    p2_state, self.agent2_env.valid_actions
                )
                _, reward, done, _ = self.agent2_env.step(p2_action, -1)

                self.env.render("human")
                if done:
                    break
            print(f"Result was : {reward}")

        self.agent1.train = True
        self.agent2.train = True


if __name__ == "__main__":
    from pprint import pprint

    agent = QTicTacTable(0.1)
    determined = EpsilonPageLines()
    battledome = TicTacToeRunner(determined, agent)
    battledome.run()
    for i in range(10):
        agent.epsilon = (9 - i) / 10
        print("Agent epsilon ", agent.epsilon)
        battledome.train(100)
    battledome.run()
    # pprint(agent.qdict)

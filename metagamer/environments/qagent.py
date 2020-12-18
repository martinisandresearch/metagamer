from typing import Tuple, Optional, Any, Dict

GAMMA = 0.9
import random
import numpy as np
from metagamer.environments import tictactoe
from metagamer.environments import qtable


def exploit(epsilon):
    """exploitation increases as epsilon decreases"""
    return random.random() > epsilon


class TicTacToeTableEnv(tictactoe.TicTacToeEnv):
    """An TicTacTo subclass environment adapted to Q learning"""

    SYMBOLS = {1: "X", -1: "O", 0: " "}

    def _get_obs(self) -> str:
        return "".join([self.SYMBOLS[i] for i in self.board.flatten()])

    def action_to_2d(self, action: int):
        """Turns the integer position (in array of board) into 2D coordinate in form of tuple """
        new_action = tuple(
            np.unravel_index(action, (self.board.shape[0], self.board.shape[1]))
        )
        return new_action

    def action_to_1d(self, action: Tuple[int, int]) -> int:
        """Take a 2D coordinate on the board, and turn it into an integer position on the board"""
        return self.board.shape[1] * action[0] + action[1]


class Agent:
    def __init__(self, player_number):
        self.player_number = player_number
        self.train = True
        self.epsilon = 0.9

    def get_action(self, state, valid_actions):
        if exploit(self.epsilon) and not self.train:
            return self.get_arg_max(state, self.valid_actions())
        return random.choice(valid_actions)

    def get_arg_max(self, state, valid_actions):
        """used to play"""
        pass

    def get_max(self, state, valid_actions):
        """used to train"""
        pass

    def __setitem__(self, key, value):
        pass


class Random(Agent):
    def to_action(self, action):
        return

    def get_arg_max(self, state, valid_actions):
        return tictactoe.random_policy(state, self.player_number)

    def get_max(self, state, valid_actions):
        return 1

    # qtable
    # random
    # dqn
    # policy gradient
    # left to right


class TicTacToeRunner:
    def __init__(self, agent1: Agent, agent2: Agent, gamma: float = GAMMA):
        self.env = TicTacToeTableEnv()
        self.gamma = gamma
        # player order
        self.agent1 = Random(1)
        self.agent2 = Random(-1)

    def train(self, num_epsiodes, initeps=1, finaleps=0.05):
        """
        Simple linear drop for epsilon.
        """
        epsdecay = (initeps - finaleps) / num_epsiodes
        epsilon = initeps

        test_window = int(num_epsiodes / 20)

        results = np.zeros(num_epsiodes)
        eps_vals = np.zeros(num_epsiodes)

        for i in range(num_epsiodes):
            p1_state = self.env.reset()
            steps = 0
            epsilon -= epsdecay

            done = False

            while not done:
                p1_action = self.agent1.get_action(p1_state, self.env.valid_actions)

                p2_state, p2_reward, done, info = self.env.step(p1_action, 1)
                if i == num_epsiodes - 1:
                    self.env.render()

                if done:
                    # player 1 has won and we need send rewards
                    self.agent1[p1_state, p1_action] = p2_reward

                    # if game is over we also need to send state back to player2
                    self.agent2[p2_state, p2_action] = p2_reward * -1
                    break

                # Get the other player to take their turn, and update state
                p2_action = self.agent2.get_action(p2_state, self.env.valid_actions)

                p1_state, p1_reward, done, info = self.env.step(p2_action, -1)
                if i == num_epsiodes - 1:
                    self.env.render()

                if done:
                    # player 2
                    self.agent2[p2_state, p2_action] = p1_reward * -1
                    self.agent1[p1_state, p1_action] = p1_reward
                    break

                else:
                    # add the future reward * decay if we're still going
                    p1_reward += self.gamma * self.agent1.get_max(
                        p1_state, self.env.valid_actions
                    )
                    steps += 1
                    # self.agent1[p1_state]
                # # to 1 d
                # action = action[0] * 3 + action[1]
                # self.Qstate[state, action] = reward * self.reward_multiple
                # state = new_state

            # if reward == -1 and i > 3 *num_epsiodes//4:
            #     print("Found a loss")
            #     self.env.render()

            results[i] = reward
            eps_vals[i] = epsilon

            if i and i % test_window == 0:
                # every 5%
                upp = i
                low = upp - test_window
                print(
                    f"{i}: eps:{epsilon:.2f},  wins: {np.sum(results[low:upp]==1)}"
                    f" losses: {np.sum(results[low:upp]==-1):.2f}"
                    f" draws: {np.sum(results[low:upp]==0):.2f}"
                )

        return results, eps_vals

    # def run(self):
    #     from gym.wrappers.monitor import Monitor
    #
    #     # env = Monitor(self.env, Agent.VID_DIR, force=True)
    #     done = False
    #     steps = 0
    #     state = self.env.reset()
    #     while not done:
    #         env.render()
    #         action = self.get_action(state, epsilon=0)
    #         print(f"{steps} {state} {action}")
    #         new_state, reward, done, info = env.step(action)
    #         state = new_state
    #         steps += 1
    #
    #     print(f"Numsteps: {steps}")
    #     env.close()


if __name__ == "__main__":
    from pprint import pprint

    agent = Agent()
    agent.train(10000, finaleps=-0.1)
    pprint(agent.Qstate.qdict[" " * 9])
    pprint(len(agent.Qstate.qdict))

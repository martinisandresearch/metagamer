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
    VID_DIR = "./extra/video"

    def __init__(self, gamma: float = GAMMA, player: int = 1):
        self.env = TicTacToeTableEnv()
        self.gamma = gamma
        self.Qstate = qtable.QTicTacTable()
        self.player = player
        if self.player == 1:
            self.other = -1
            self.reward_multiple = 1
        else:
            self.other = 1
            self.reward_multiple = -1

    def valid_actions(self):
        return [self.env.action_to_1d(a) for a in self.env.valid_actions]

    def get_action(self, state, epsilon):
        if exploit(epsilon):
            return self.env.action_to_2d(
                self.Qstate.get_arg_max(state, self.valid_actions())
            )
        else:
            return random.choice(self.env.valid_actions)

    def train(self, num_epsiodes, initeps=1, finaleps=0.05):
        """
        Simple linear drop for epsilon.
        """
        epsdecay = (initeps - finaleps) / num_epsiodes
        epsilon = initeps

        test_window = int(num_epsiodes / 20)

        num_steps = np.zeros(num_epsiodes)
        eps_vals = np.zeros(num_epsiodes)

        for i in range(num_epsiodes):
            state = self.env.reset()
            steps = 0
            epsilon -= epsdecay

            done = False

            while not done:
                action = self.get_action(state, epsilon=epsilon)

                new_state, reward, done, info = self.env.step(action, self.player)
                if i == num_epsiodes - 1:
                    self.env.render()

                # Get the other player to take their turn, and update state
                if not done:
                    new_state, reward, done, info = self.env.step(
                        tictactoe.policy_page_lines(self.env.board, self.other),
                        self.other,
                    )
                    if i == num_epsiodes - 1:
                        self.env.render()

                if not done:
                    reward = reward * self.reward_multiple
                    # add the future reward * decay if we're still going
                    reward += self.gamma * self.Qstate.get_max(
                        new_state, self.valid_actions()
                    )
                    steps += 1
                # to 1 d
                action = action[0] * 3 + action[1]
                self.Qstate[state, action] = reward * self.reward_multiple
                state = new_state

            num_steps[i] = steps
            eps_vals[i] = epsilon

            if i and i % test_window == 0:
                # every 5%
                upp = i
                low = upp - test_window
                print(
                    f"{i}: eps:{epsilon:.2f},  max: {np.max(num_steps[low:upp])}"
                    f" ave: {np.mean(num_steps[low:upp]):.2f}"
                    f" std: {np.std(num_steps[low:upp]):.2f}"
                )

        return num_steps, eps_vals

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
    agent = Agent()
    agent.train(1000, finaleps=0)

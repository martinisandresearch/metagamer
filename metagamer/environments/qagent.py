GAMMA = 0.9
import random
import numpy as np
from metagamer.environments import tictactoe


def exploit(epsilon):
    """exploitation increases as epsilon decreases"""
    return random.random() > epsilon


class Agent:
    VID_DIR = "./extra/video"

    def __init__(self, Qstate, gamma: float = GAMMA, player: int = 1):
        self.env = tictactoe.TicTacToeEnv()
        self.gamma = gamma
        self.Qstate = Qstate(statedim=1, num_actions=self.env.action_space.n)
        self.player = player
        if self.player == 1:
            self.other = -1
            self.reward_multiple = 1
        else:
            self.other = 1
            self.reward_multiple = -1

    def get_action(self, state, epsilon):
        if exploit(epsilon):
            return self.Qstate.get_arg_max(state)
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

                # Get the other player to take their turn, and update state
                if not done:
                    new_state, reward, done, info = self.env.step(
                        tictactoe.policy_page_lines(self.env.board, self.other),
                        self.other,
                    )

                if not done:
                    reward = reward * self.reward_multiple
                    # add the future reward * decay if we're still going
                    reward += self.gamma * self.Qstate.get_max(new_state)
                    steps += 1

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

    def run(self):
        from gym.wrappers.monitor import Monitor

        env = Monitor(self.env, Agent.VID_DIR, force=True)
        done = False
        steps = 0
        state = env.reset()
        while not done:
            env.render(mode="rgb_array")
            action = self.get_action(state, epsilon=0)
            print(f"{steps} {state} {action}")
            new_state, reward, done, info = env.step(action)
            state = new_state
            steps += 1

        print(f"Numsteps: {steps}")
        env.close()

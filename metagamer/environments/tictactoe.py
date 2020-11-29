"""
Original implementation provided by https://github.com/haje01/gym-tictactoe
"""
import logging

import numpy as np
import gym
from gym import spaces

from typing import Tuple

CODE_MARK_MAP = {0: " ", 1: "O", -1: "X"}
NUM_LOC = 9
O_REWARD = 1
X_REWARD = -1
NO_REWARD = 0

LEFT_PAD = "  "

logger = logging.getLogger(__name__)


def tomark(code):
    return CODE_MARK_MAP[code]


def tocode(mark):
    return 1 if mark == "O" else -1


def next_mark(mark):
    return "X" if mark == "O" else "O"


def agent_by_mark(agents, mark):
    for agent in agents:
        if agent.mark == mark:
            return agent


def after_action_state(state, action):
    """Execute an action and returns resulted state.

    Args:
        state (tuple): Board status + mark
        action (int): Action to run

    Returns:
        tuple: New state
    """

    board, mark = state
    nboard = list(board[:])
    nboard[action] = tocode(mark)
    nboard = tuple(nboard)
    return nboard, next_mark(mark)


def check_game_status(board):
    """Return game status by current board status.

    Args:
        board (list): Current board state

    Returns:
        int:
            -1: game in progress
            0: draw game,
            1 or 2 for finished game(winner mark code).
    """
    for t in [1, 2]:
        for j in range(0, 9, 3):
            if [t] * 3 == [board[i] for i in range(j, j + 3)]:
                return t
        for j in range(0, 3):
            if board[j] == t and board[j + 3] == t and board[j + 6] == t:
                return t
        if board[0] == t and board[4] == t and board[8] == t:
            return t
        if board[2] == t and board[4] == t and board[6] == t:
            return t

    for i in range(9):
        if board[i] == 0:
            # still playing
            return -1

    # draw game
    return 0


class TicTacToeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.action_space = spaces.Discrete(NUM_LOC)
        self.observation_space = spaces.Discrete(NUM_LOC)
        self.seed()
        self.reset()

    def reset(self):
        self.board = np.a
        self.mark = "X"
        self.done = False
        return self._get_obs()

    def step(self, action: Tuple[int, int]):
        """Step environment by action.

        Args:
            action (int): Location

        Returns:
            list: Observation
            int: Reward
            bool: Done
            dict: Additional information
        """
        assert self.action_space.contains(action)

        loc = action
        if self.done:
            return self._get_obs(), 0, True, None

        reward = NO_REWARD
        # place
        self.board[loc] = tocode(self.mark)
        status = check_game_status(self.board)
        logger.debug(
            "check_game_status board {} mark '{}'"
            " status {}".format(self.board, self.mark, status)
        )
        if status >= 0:
            self.done = True
            if status in [1, 2]:
                # always called by self
                reward = O_REWARD if self.mark == "O" else X_REWARD

        # switch turn
        self.mark = next_mark(self.mark)
        return self._get_obs(), reward, self.done, None

    def _get_obs(self):
        return tuple(self.board), self.mark

    def render(self, mode="human", close=False):
        if close:
            return
        if mode == "human":
            self._show_board(print)  # NOQA
            print("")
        else:
            self._show_board(logger.info)
            logger.info("")

    def show_episode(self, human, episode):
        self._show_episode(print if human else logger.warning, episode)

    def _show_episode(self, showfn, episode):
        showfn("==== Episode {} ====".format(episode))

    def _show_board(self, showfn):
        """Draw tictactoe board."""
        for j in range(0, 9, 3):

            showfn(LEFT_PAD + "|".join([tomark(self.board[i]) for i in range(j, j + 3)]))
            if j < 6:
                showfn(LEFT_PAD + "-----")

    def show_turn(self, human, mark):
        self._show_turn(print if human else logger.info, mark)

    def _show_turn(self, showfn, mark):
        showfn("{}'s turn.".format(mark))

    def show_result(self, human, mark, reward):
        self._show_result(print if human else logger.info, mark, reward)

    def _show_result(self, showfn, mark, reward):
        status = check_game_status(self.board)
        assert status >= 0
        if status == 0:
            showfn("==== Finished: Draw ====")
        else:
            msg = "Winner is '{}'!".format(tomark(status))
            showfn("==== Finished: {} ====".format(msg))
        showfn("")

    def available_actions(self):
        return [i for i, c in enumerate(self.board) if c == 0]

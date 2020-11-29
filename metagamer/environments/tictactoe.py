"""
Thie file provides a simple

"""
import logging

import numpy as np
import gym
from gym import spaces, error

from typing import Iterable, Any, Tuple, Dict, List, Optional

logger = logging.getLogger(__name__)


def turn_iterator(turn_order: Iterable[Any]):
    while True:
        for turn in turn_order:
            yield turn


def check_win(board: np.array) -> int:
    """
    Given a 3*3 numpy array whose elements are 0, 1, -1
    representing empty, player1 and player 2,
    return the winner or the

    Args:
        board (np.array): 3 x 3 board

    Returns:
        int: 0  -> no winner
             1  -> player 1
             -1 -> player 2

    """
    assert board.shape == (3, 3)
    for axis in [0, 1]:
        axis_sum = board.sum(axis=axis)
        if 3 in axis_sum:
            return 1
        elif -3 in axis_sum:
            return -1

    diag = board.diagonal().sum()
    diag_inv = np.fliplr(board).diagonal().sum()
    for diag_sum in [diag, diag_inv]:
        if diag_sum == 3:
            return 1
        elif diag_sum == -3:
            return -1

    # no winner yet
    return 0


def to_one_hot(board: np.array) -> np.array:
    """
    Convert the representation to one hot encoding
    We assume a 3x3 board input

    Args:
        board (np.array): 3 x 3

    Returns:
        3 x 3 x 3:
            board[:,: 0] = location of empty squares
            board[:, :, 1] = location of player 1
            board[:, :, 2] = location of player 2 moves

    """
    oh = np.stack((board == 0, board == 1, board == -1))
    return oh.shape.astype(int)


def to_human(board: np.array) -> np.array:
    human_board = np.full(board.shape, '')
    human_board[np.where(board == 1)] = 'X'
    human_board[np.where(board == -1)] = 'O'
    return human_board


class TicTacToeEnv(gym.Env):
    """
    TicTacToe

    """

    metadata = {"render.modes": ["human"]}

    TURN_ORDER = [1, -1]

    def __init__(self):
        self.board_shape = 3, 3
        self.symbols = {1: "X", -1: "O"}
        self.action_space = spaces.Tuple([spaces.Discrete(3), spaces.Discrete(3)])

        self.board = None
        self.turn_iterator = None
        self.curr_turn = None
        self.done = None

        self.reset()

    def reset(self):
        self.board = np.zeros(self.board_shape, dtype=int)
        self.turn_iterator = turn_iterator(self.TURN_ORDER)
        self.curr_turn = next(self.turn_iterator)
        self.done = False

    def _get_obs(self):
        return self.board

    def step(self, action: Tuple[int, int], player:Optional[int]=None) -> Tuple[Any, float, bool, Dict]:
        action = tuple(action)
        if self.board[action] != 0:
            raise error.InvalidAction(f"action {action} is not a vaid choice")
        if not self.done:
            error.ResetNeeded("Call reset as game is over")

        self.board[action] = self.curr_turn

        reward = self.check_win()
        if reward:
            self.done = True
            return self._get_obs(), float(reward), self.done, {}

        self.curr_turn = next(self.turn_iterator)

        return self._get_obs(), 0.0, self.done, {}

    def check_win(self) -> int:
        return check_win(self.board)

    @property
    def valid_actions(self) -> List[Tuple[int, int]]:
        return np.argwhere(self.board == 0).tolist()

    @property
    def turns_played(self):
        return np.sum(self.board != 0)

    def render(self, mode='human'):
        from tabulate import tabulate
        human_board = to_human(self.board)
        print("\n")
        print(f"Turn : {self.turns_played}")
        print(tabulate(human_board.tolist(), tablefmt='fancy_grid'))
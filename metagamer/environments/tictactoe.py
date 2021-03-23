"""
This file provides some simple tools to simulate and work with a tic tac toe board.
The functions provided assume the board is represented by
    - np array of shape 3,3
    - player 1 is 1, player 2 is -1 and empty is 0

"""
import logging
import itertools
import random

import numpy as np
import gym
from gym import spaces, error

from typing import Any, Tuple, Dict, List, Optional

logger = logging.getLogger(__name__)


def check_win(board: np.array) -> int:
    """
    Given a 3*3 numpy array whose elements are 0, 1, -1
    representing empty, player1 and player 2,
    return the player who's won if any or 0 if no winner found

    Args:
        board (np.array): 3 x 3 board

    Returns:
        int:  0 -> no winner
              1 -> player 1
             -1 -> player 2

    """
    assert board.shape[0] == board.shape[1]
    compval = board.shape[0]
    for axis in [0, 1]:
        axis_sum = board.sum(axis=axis)
        if compval in axis_sum:
            return 1
        elif -compval in axis_sum:
            return -1

    diag = board.diagonal().sum()
    diag_inv = np.fliplr(board).diagonal().sum()
    for diag_sum in [diag, diag_inv]:
        if diag_sum == compval:
            return 1
        elif diag_sum == -compval:
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
            board[0 :  :] = location of empty squares
            board[1, :, :] = location of player 1
            board[2, :, :] = location of player 2 moves

    """
    oh = np.stack((board == 0, board == 1, board == -1))
    return oh.astype(int)


def to_human(board: np.array, symbols) -> np.array:
    """Convert this into a """
    human_board = np.full(board.shape, " ")
    for value, sym in symbols.items():
        human_board[np.where(board == value)] = sym
    return human_board


class TicTacToeEnv(gym.Env):
    """
    TicTacToe environment in the openai gym style: https://gym.openai.com/docs/

    In addition, we require the definition of two functions
    1. `get_observation` which returns the observation
    2. `valid_actions` - returns a list of valid actions

    The first is necessary for
    """

    # openai gym api - can also have rgb (for things like atari games) or ansi (text)
    metadata = {"render.modes": ["human"]}

    # constants that define the game's implementation
    TURN_ORDER = (1, -1)
    BOARD_SHAPE = 3, 3
    SYMBOLS = {1: "X", -1: "O", 0: " "}

    def __init__(self):
        # open AI Gym API
        # necessary to set these to allow for easy network architecture

        # space of the actions - in this case the coordinate of the board to play
        self.action_space = spaces.Tuple(
            [spaces.Discrete(self.BOARD_SHAPE[0]), spaces.Discrete(self.BOARD_SHAPE[1])]
        )
        # how are the observations represented. Since we return the board, we're returning
        # a discrete 3x3 matrix where each entry is {-1, 0, 1}.
        # this doesn't have a nice gym.spaces representation so we leave it unfilled for now
        self.observation_space = None

        # state representation variables. We define them here and set them in reset
        self.board = None
        self.turn_iterator = None
        self.curr_turn = None
        self.done = None

        # reset does the initalisation
        self.reset()

    def reset(self) -> np.array:
        self.board = np.zeros(self.BOARD_SHAPE, dtype=int)
        self.turn_iterator = itertools.cycle(self.TURN_ORDER)
        self.curr_turn = next(self.turn_iterator)
        self.done = False
        return self.get_observation()

    def step(self, action: Any, player: Optional[int] = None) -> Tuple[Any, float, bool, Dict]:
        """

        Args:
            action: locaton we
            player: In more complex environments, we'll want to ensure we're not playing as the
                the same player twice. This provides a way of checking we're not breaking
                order by mistake

        Returns:
            observation, reward, done, info

        """
        # check the action is valid and the game isn't over
        action = tuple(action)
        logger.debug("Selected action: %s on turn %d", action, self.turns_played + 1)
        if self.board[action] != 0:
            raise error.InvalidAction(
                f"action {action} is not a valid choice - try {self.valid_actions}"
            )
        if self.done:
            raise error.ResetNeeded("Call reset as game is over")
        if player and player != self.curr_turn:
            raise error.InvalidAction(f"Player {self.curr_turn}'s turn. Move request from {player}")

        # set the location on the board to the current player. Since curr_turn
        # and current player use the same indicator, we just use that
        self.board[action] = self.curr_turn

        # check if the game is over. Reward is player that won (1 or -1)
        reward = check_win(self.board)
        if reward:
            self.done = True
            return self.get_observation(), float(reward), self.done, {}

        # check if the game is over (i.e. no more turns). Since we don't have a win
        # it must be a draw
        if self.turns_played == 9:
            self.done = True
            return self.get_observation(), 0.0, self.done, {}

        # otherwise game is still going. Advance turn and return state + no reward
        self.curr_turn = next(self.turn_iterator)
        return self.get_observation(), 0.0, self.done, {}

    def get_observation(self) -> Any:
        """
        Abstracted the observation from the underlying state though in this case they're
        identical. This is a common pattern in most third party gym environments.

        This makes changing the state output as simple as a subclass that overrides this function
        as well as the action_space/observation space as opposed to the more onerous gym wrapper

        Returns:
            np.array of 3x3 representing the board in it's default state

        """
        return self.board

    @property
    def valid_actions(self) -> List[Tuple[int, int]]:
        return [tuple(act) for act in np.argwhere(self.board == 0).tolist()]

    @property
    def turns_played(self) -> int:
        return np.sum(self.board != 0)

    def render(self, mode="human"):
        import tabulate

        tabulate.PRESERVE_WHITESPACE = True

        human_board = to_human(self.board, self.SYMBOLS)
        print("\n")
        print(f"Turn : {self.turns_played}")
        print(tabulate.tabulate(human_board.tolist(), tablefmt="fancy_grid"))

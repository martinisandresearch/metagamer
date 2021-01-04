import pytest
import numpy as np
import random
import gym.error
from metagamer.environments import tictactoe


@pytest.fixture
def tt():
    return tictactoe.TicTacToeEnv()


def test_policy_page_lines_first(tt):
    """
    Just a calling a policy to make sure that it replay from a couple of different boards playing first.
    """
    while not tt.done:
        tt.step(random.choice(tt.valid_actions))
        if not tt.done:
            tt.step(tictactoe.policy_page_lines(tt.get_observation(), 1))


def test_policy_page_lines_second(tt):
    """
    Just a calling a policy to make sure that it replay from a couple of different boards playing second.
    """
    while not tt.done:
        tt.step(tictactoe.policy_page_lines(tt.get_observation(), -1))
        if not tt.done:
            tt.step(random.choice(tt.valid_actions))


def test_simple_game(tt):
    """
    We make a game go where X is going from top left to bottom left
    and O is going from top middle to bottom middle and check that
    the rewards and state are expected each time.

    We expect the game to be not done and have 0 reward until the final move
    at which point we'll see the env.done and reward = 1

    ╒═══╤═══╤═══╕
    │ X │ O │   │
    ├───┼───┼───┤
    │ X │ O │   │
    ├───┼───┼───┤
    │ X │   │   │
    ╘═══╧═══╧═══╛
    """
    _, reward, done, _ = tt.step((0, 0))
    assert not done
    assert reward == 0
    _, reward, done, _ = tt.step((0, 1))
    assert not done
    assert reward == 0
    _, reward, done, _ = tt.step((1, 0))
    assert not done
    assert reward == 0
    _, reward, done, _ = tt.step((1, 1))
    assert not done
    assert reward == 0

    _, reward, done, _ = tt.step((2, 0))
    assert done
    assert reward == 1


@pytest.mark.parametrize("player", (1, -1))
def test_diagonal(player):
    """diagonal logic needed some checking"""
    board = np.zeros((3, 3))
    board[0, 0] = player
    assert tictactoe.check_win(board) == 0
    board[1, 1] = player
    assert tictactoe.check_win(board) == 0
    board[2, 2] = player
    assert tictactoe.check_win(board) == player


@pytest.mark.parametrize("player", (1, -1))
def test_rev_diagonal(player):
    board = np.zeros((3, 3))
    board[0, 2] = player
    assert tictactoe.check_win(board) == 0
    board[1, 1] = player
    assert tictactoe.check_win(board) == 0
    board[2, 0] = player
    assert tictactoe.check_win(board) == player


@pytest.mark.parametrize("seed", range(100))
def test_random(tt, seed):
    """Run a 100 random games and check that our properties are always satisfied"""
    random.seed(seed)
    while not tt.done:
        action = random.choice(tt.valid_actions)

        _, reward, *_ = tt.step(action)
        assert tt.board.sum() in {0, 1}
    assert tt.turns_played >= 5
    assert reward in {-1, 0, 1}


def test_one_hot():
    # set a board up 4 moves played
    board = np.array([[1, 0, 0], [-1, 1, 0], [0, 0, -1]])
    onehot = tictactoe.to_one_hot(board)
    # all squares have a one in at least one place
    assert (onehot.sum(axis=0) == 1).all()
    assert (onehot[1, :, :] == np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])).all()
    assert (onehot[2, :, :] == np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]])).all()
    assert onehot[0, :, :].sum() == 5


def test_cant_occupy_used(tt):
    """can't occupy same place twice"""
    tt.step((0, 0))
    with pytest.raises(gym.error.InvalidAction):
        tt.step((0, 0))


def test_check_player_order(tt):
    """valid move, playing out of order"""
    assert tt.curr_turn == 1

    tt.step((0, 0), 1)
    assert tt.curr_turn == -1

    tt.step((0, 1), -1)

    assert tt.curr_turn == 1

    with pytest.raises(gym.error.InvalidAction):
        tt.step((1, 1), -1)


def test_error_after_game_over(tt):
    """valid move, game over"""

    tt.step((0, 0))
    tt.step((0, 1))
    tt.step((1, 0))
    tt.step((1, 1))
    tt.step((2, 0))

    with pytest.raises(gym.error.ResetNeeded):
        tt.step((2, 1))


def test_render(tt, capsys):
    tt.step((0, 0))
    tt.render()
    captured = capsys.readouterr()
    assert (
        captured.out
        == """

Turn : 1
╒═══╤═══╤═══╕
│ X │   │   │
├───┼───┼───┤
│   │   │   │
├───┼───┼───┤
│   │   │   │
╘═══╧═══╧═══╛
"""
    )

    tt.step((2, 1))
    tt.render()
    captured = capsys.readouterr()
    assert (
        captured.out
        == """

Turn : 2
╒═══╤═══╤═══╕
│ X │   │   │
├───┼───┼───┤
│   │   │   │
├───┼───┼───┤
│   │ O │   │
╘═══╧═══╧═══╛
"""
    )

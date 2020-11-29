import pytest
import numpy as np

from metagamer.environments import tictactoe


@pytest.fixture
def tt():
    return tictactoe.TicTacToeEnv()


def test_simple_game(tt):
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
    board = np.zeros((3, 3))
    board[0, 0] = player
    board[1, 1] = player
    board[2, 2] = player
    assert tictactoe.check_win(board) == player


@pytest.mark.parametrize("player", (1, -1))
def test_rev_diagonal(player):
    board = np.zeros((3, 3))
    board[0, 2] = player
    board[1, 1] = player
    board[2, 0] = player
    assert tictactoe.check_win(board) == player


def test_render(tt, monkeypatch):
    tt.step((0, 0))
    tt.render()

    tt.step((2, 1))
    tt.render()

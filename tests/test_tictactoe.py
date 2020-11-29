import pytest
import numpy as np
import random
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


@pytest.mark.parametrize("seed", range(100))
def test_random(tt, seed):
    random.seed(seed)
    while not tt.done:
        action = random.choice(tt.valid_actions)

        _, reward, *_ = tt.step(action)
        assert tt.board.sum() in {0, 1}
    assert tt.turns_played >= 5
    assert reward in {-1, 0, 1}


def test_render(tt, capsys):
    tt.step((0, 0))
    tt.render()
    captured = capsys.readouterr()
    assert captured.out == """

Turn : 1
╒═══╤═══╤═══╕
│ X │   │   │
├───┼───┼───┤
│   │   │   │
├───┼───┼───┤
│   │   │   │
╘═══╧═══╧═══╛
"""

    tt.step((2, 1))
    tt.render()
    captured = capsys.readouterr()
    assert captured.out == """

Turn : 2
╒═══╤═══╤═══╕
│ X │   │   │
├───┼───┼───┤
│   │   │   │
├───┼───┼───┤
│   │ O │   │
╘═══╧═══╧═══╛
"""


def test_f(tt):
    tt.step((0,0))
    tt.step((2,1))

    tt.render()
    assert False
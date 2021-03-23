from metagamer.agents import dqagent
import torch


def test_pure_self_play():
    p1 = dqagent.DQTicTacTable()
    p2 = dqagent.DQTicTacTable()
    new_dome = dqagent.DTicTacToeRunner(p1, p2)
    nr = 30
    p2.train = True
    p1.train = True
    for i in range(nr):
        p1.epsilon = (nr - 1 - i) / nr
        p2.epsilon = (nr - 1 - i) / nr
        print("epsilon", p1.epsilon, end=" ")
        new_dome.train(100)
    print("\nNo training")
    p1.train = False
    p2.train = False
    new_dome.train(100)
    print(p1.Q_1(torch.zeros(9)))
    print(p2.Q_2(torch.zeros(9)))
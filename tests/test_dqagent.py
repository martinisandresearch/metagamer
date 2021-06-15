from metagamer.agents import dqagent
import torch
import pandas as pd
from matplotlib import pyplot as plt


def test_pure_self_play():
    p1 = dqagent.DQTicTacNetwork(name="P1")
    p2 = dqagent.DQTicTacNetwork(name="P2")
    new_dome = dqagent.DTicTacToeRunner(p1, p2)
    nr = 10
    p2.train = True
    p1.train = True
    for i in range(nr):
        p1.epsilon = (nr - 1 - i) / nr
        p2.epsilon = (nr - 1 - i) / nr
        print(f"epsilon {p1.epsilon:.2f}", end=" ")
        new_dome.train(100)
    loss_df = pd.DataFrame(
        {
            "P1mean": new_dome.agent1meanloss,
            "P1max": new_dome.agent1maxloss,
            "P1min": new_dome.agent1minloss,
            "P2mean": new_dome.agent2meanloss,
            "P2max": new_dome.agent2maxloss,
            "P2min": new_dome.agent2minloss,
        }
    )
    loss_df.plot()
    plt.show()
    print("\nNo training")
    p1.train = False
    p2.train = False
    new_dome.train(100)
    print(p1.Q_1(torch.zeros(9)))
    print(p2.Q_2(torch.zeros(9)))

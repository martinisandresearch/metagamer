from metagamer.agents import dqagent
import torch
import pandas as pd
from matplotlib import pyplot as plt
from metagamer.agents import qagent


def test_pure_self_play():
    p1 = dqagent.DQTicTacNetwork(
        name="P1",
        hidden_dim=27,
        lr=0.005,
        lr_gamma=0.99,
        lr_step=100,
        max_memory_size=7000,
    )
    p1.gamma = 0.6

    p2 = dqagent.DQTicTacNetwork(
        name="P2",
        hidden_dim=27,
        lr=0.005,
        lr_gamma=0.99,
        lr_step=100,
        max_memory_size=7000,
    )
    p2.gamma = 0.6

    new_dome = dqagent.DTicTacToeRunner(p1, p2)
    new_dome.min_episodes = 200
    new_dome.update_step = 10
    new_dome.update_repeats = 20
    new_dome.batch_size = 100
    new_dome.draw_reward = 1.0

    starting_eps = 0.8
    p2.train = True
    p1.train = True
    p1.epsilon = starting_eps
    p2.epsilon = starting_eps
    print(f"epsilon {p1.epsilon:.2f}", end=" ")
    new_dome.train(400)
    p1.epsilon = 0.5
    p2.epsilon = 0.5
    print(f"epsilon {p1.epsilon:.2f}", end=" ")
    new_dome.train(400)
    p1.epsilon = 0.2
    p2.epsilon = 0.2
    print(f"epsilon {p1.epsilon:.2f}", end=" ")
    new_dome.train(800)
    p1.epsilon = 0.05
    p2.epsilon = 0.05
    print(f"epsilon {p1.epsilon:.2f}", end=" ")
    new_dome.batch_size = 128
    new_dome.update_repeats = 30
    new_dome.train(2000)
    p1.epsilon = 0.01
    p2.epsilon = 0.01
    new_dome.batch_size = 130
    print(f"epsilon {p1.epsilon:.2f}", end=" ")
    new_dome.train(1500)
    p1.positive_greedy = True
    p2.positive_greedy = True
    p1.epsilon = 0.05
    p2.epsilon = 0.05
    print(f"epsilon {p1.epsilon:.2f}", end=" ")
    new_dome.train(1000)
    p1.positive_greedy = False
    p2.positive_greedy = False
    p1.epsilon = 0.01
    p2.epsilon = 0.01
    print(f"epsilon {p1.epsilon:.2f}", end=" ")
    new_dome.train(1000)
    p1.gamma = 0.8
    p2.gamma = 0.8
    p1.epsilon = 0.0
    p2.epsilon = 0.0
    new_dome.train(1000)

    loss_df = pd.DataFrame(
        {
            "P1mean": new_dome.agent1meanloss,
            "P2mean": new_dome.agent2meanloss,
            "P1umoves": new_dome.agent1unique_moves_frac,
            "P2umoves": new_dome.agent2unique_moves_frac,
        }
    )
    loss_df.plot()
    plt.show()
    print("\nNo training")
    p1.train = False
    p2.train = False
    new_dome.train(100)
    # print(p1.Q_1(torch.zeros(9)))
    print(p1.Q_2(torch.zeros(9)))
    print(p1.Q_2(torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0])))
    # tensor([0.5591, -0.8117, 0.0979,
    #         -0.5002, -1.1774, -0.1033,
    #         -2.0456, -1.0836, 0.6419])
    # print(p2.Q_1(torch.zeros(9)))
    print(p2.Q_2(torch.zeros(9)))
    print(p2.Q_2(torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0])))
    # tensor([-0.0150, 0.1610, 0.3941,
    #         0.3336, -0.1797, -0.6697,
    #         0.6472, -0.6193, -0.0606])

    new_dome.run()

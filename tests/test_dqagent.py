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
        lr_step=10,
        max_memory_size=4000,
    )
    p1.gamma = 0.6

    p2 = dqagent.DQTicTacNetwork(
        name="P2",
        hidden_dim=27,
        lr=0.005,
        lr_gamma=0.99,
        lr_step=10,
        max_memory_size=4000,
    )
    p2.gamma = 0.6

    new_dome = dqagent.DTicTacToeRunner(p1, p2)
    new_dome.min_episodes = 200
    new_dome.update_step = 10
    new_dome.update_repeats = 20
    new_dome.batch_size = 100
    new_dome.draw_reward = 1.0

    nr = 3
    starting_eps = 0.8
    p2.train = True
    p1.train = True
    # for i in range(int(nr)+1):
    #     p1.epsilon = starting_eps*(nr-i)/nr
    #     p2.epsilon = starting_eps*(nr-i)/nr
    #     print(f"epsilon {p1.epsilon:.2f}", end=" ")
    #     new_dome.train(100)
    p1.epsilon = starting_eps
    p2.epsilon = starting_eps

    print(f"epsilon {p1.epsilon:.2f}", end=" ")
    new_dome.train(800)

    p1.epsilon = 0.5
    p2.epsilon = 0.5
    print(f"epsilon {p1.epsilon:.2f}", end=" ")
    new_dome.train(400)
    # p1.positive_greedy = True
    # p2.positive_greedy = True

    # p1.epsilon = 0.2
    # p2.epsilon = 0.2
    # print(f"epsilon {p1.epsilon:.2f}", end=" ")
    # new_dome.train(150)
    p1.epsilon = 0.2
    p2.epsilon = 0.2
    print(f"epsilon {p1.epsilon:.2f}", end=" ")
    new_dome.train(400)
    # p1.epsilon = 0.1
    # p2.epsilon = 0.1
    # print(f"epsilon {p1.epsilon:.2f}", end=" ")
    # new_dome.train(80)
    p1.epsilon = 0.05
    p2.epsilon = 0.05
    print(f"epsilon {p1.epsilon:.2f}", end=" ")
    new_dome.train(1000)

    loss_df = pd.DataFrame(
        {
            "P1mean": new_dome.agent1meanloss,
            "P2mean": new_dome.agent2meanloss,
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
    # tensor([-6.9808e-01, 1.3359e-03, -1.5508e-01,
    #                -1.2238e+00, -1.5539e+00,  5.8003e-01,
    #                 -5.2739e-01, -1.0781e+00, -7.0096e-01])
    # print(p2.Q_1(torch.zeros(9)))
    print(p2.Q_2(torch.zeros(9)))
    print(p2.Q_2(torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0])))
    # #
    # tensor([-0.5150, -1.0812, 0.1013,
    # -0.2387, 0.1110, -0.6354,
    # -0.6491, -1.1317,-0.5774])

    new_dome.run()

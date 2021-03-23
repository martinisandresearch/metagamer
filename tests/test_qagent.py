from metagamer.agents import qagent


def test_pure_self_play():
    p1 = qagent.QTicTacTable(0.1)
    p2 = qagent.QTicTacTable(0.1)
    new_dome = qagent.TicTacToeRunner(p1, p2)
    nr = 20
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
    new_dome.run()
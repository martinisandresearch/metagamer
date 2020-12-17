from metagamer.environments import qagent


def test_qagent_runs():
    """A most basic test that essentially makes sures it completes training and learns something"""
    n = 1000
    this_agent = qagent.Agent()
    num_steps, eps_vals = this_agent.train(n)
    assert len(num_steps) == n
    assert len(eps_vals) == n
    assert len(this_agent.Qstate.qdict) > 1

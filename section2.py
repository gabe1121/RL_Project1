import numpy as np
from functools import lru_cache
from section1 import Domain, Agent
import sys
sys.setrecursionlimit(10000)


@lru_cache(maxsize=10000)
def function_j(domain, state, agent, N, stocha=False):
    """
    The if else condition is defined in order to be able to treat deterministic and stochastic cases with the same
    function. To do so, one simply has to input the param stocha=True. if not mentioned, the deterministic case is
    considered.
    """
    if N == 0:
        return 0

    if stocha:
        disturbance = [domain.w - 10 ** -3, domain.w + 10 ** -3]
        probabilities = [(1 - domain.w), domain.w]
    else:
        disturbance = [0]
        probabilities = [1]

    j_n = 0
    for w, e in zip(disturbance, probabilities):
        next_state = domain.dynamic(state, agent.chose_action(state), w)
        reward = domain.reward(state, agent.chose_action(state), w)
        j_n += e * (reward + domain.gamma * function_j(domain, (next_state[0], next_state[1]), agent, N - 1, stocha))
    return j_n


def main(stocha):
    d = Domain()
    a = Agent()
    N = 981

    J_n = np.zeros([d.n, d.m])
    for i in range(d.n):
        for j in range(d.m):
            J_n[i, j] = format(function_j(d, (i, j), a, N, stocha),'.2f')

    print("\n", f"Cumulative reward J_n(s), N = {N}\n", J_n, "\n")


if __name__ == "__main__":
    main(True)

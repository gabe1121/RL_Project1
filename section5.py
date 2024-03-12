import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from section1 import Domain
from section2 import function_j
from section3 import function_q, Agent3
from section3 import compute_mean_reward, compute_probability
from section4 import make_trajectory, Agent4


def function_q_ini(domain, agent, trajectory, alpha):
    def function_q(domain, agent, trajectory, state, action, k, alpha):
        pass

    trajectory = [(trajectory[i], trajectory[i + 1], trajectory[i + 2], trajectory[i+3]) for i in range(0, len(trajectory)-4, 3)]

    # Initialisation
    q_dict = {}
    for i in range(domain.n):
        for j in range(domain.m):
            state = (i, j)
            for action in agent.action:
                q_dict[(state, tuple(action))] = 0

    for s_k, a_k, r_k, s_k_1 in trajectory:
        q_k = q_dict.get((s_k, a_k))
        q_k_1_max = -10**-10
        for action in agent.action:
            q_k_i = q_dict.get((s_k_1, tuple(action)))
            if q_k_i > q_k_1_max:
                q_k_1_max = q_k_i

        q_dict[(s_k, a_k)] = (1-alpha)*q_k + alpha * (r_k + domain.gamma * q_k_1_max)

    return q_dict


def subsection1(stocha=False):
    d = Domain()
    a = Agent4()
    h = make_trajectory(d, a, 50000, stocha)

    Q = function_q_ini(d, a, h, 0.05)

    mu = np.zeros([d.n, d.m])
    for i in range(d.m):
        for j in range(d.n):
            state = (i, j)
            q_a = []
            for action in a.action:
                q_a.append(Q.get((state, tuple(action))))
            mu[i, j] = np.argmax(q_a)

    print(mu)
    J_n_hat = np.zeros([d.n, d.m])
    my_a = Agent3(mu)

    for i in range(d.m):
        for j in range(d.n):
            J_n_hat[i, j] = function_j(d, (i, j), my_a, 8, stocha)

    print(J_n_hat)
    return






if __name__ == "__main__":
    stocha = False
    subsection1(stocha)

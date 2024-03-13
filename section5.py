import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from section1 import Domain, Agent
from section2 import function_j
from section3 import function_q, Agent3
from section3 import compute_mean_reward, compute_probability
from section4 import make_trajectory, Agent4


class Agent5:
    def __init__(self, domain, epsilon):
        self.domain = domain
        self.epsilon = epsilon
        self.action = [
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
        ]

    def chose_action(self, state):
        greedy = np.random.random()
        if greedy < (1-self.epsilon):
            reward_list = []
            for action in self.action:
                reward_list.append(self.domain.reward(state, action, 0))
            action_id = np.argmax(reward_list)
        else:
            action_id = np.random.randint(0, 4)
        return self.action[action_id]


def q_dict_ini(domain, agent):
    # Initialisation
    q_dict = {}
    for i in range(domain.n):
        for j in range(domain.m):
            state = (i, j)
            for action in agent.action:
                q_dict[(state, tuple(action))] = 0
    return q_dict


def function_q_5(domain, agent, trajectory, alpha, q_dict, section_2b=False):
    trajectory = [(trajectory[i], trajectory[i + 1], trajectory[i + 2], trajectory[i+3]) for i in range(0, len(trajectory)-4, 3)]

    for s_k, a_k, r_k, s_k_1 in trajectory:
        q_k = q_dict.get((s_k, a_k))
        q_k_1_max = -10**-10
        for action in agent.action:
            q_k_i = q_dict.get((s_k_1, tuple(action)))
            if q_k_i > q_k_1_max:
                q_k_1_max = q_k_i

        temporal_difference = r_k + domain.gamma * q_k_1_max - q_k
        q_dict[(s_k, a_k)] = q_k + alpha * temporal_difference
        if section_2b:
            alpha = 0.8*alpha

    return q_dict


def function_q_5_bis(domain, agent, trajectory, alpha, q_dict):
    trajectory = [(trajectory[i], trajectory[i + 1], trajectory[i + 2], trajectory[i+3]) for i in range(0, len(trajectory)-4, 3)]
    replay_buffer = []

    for one_step in trajectory:
        replay_buffer.append(one_step)
        transitions_id = np.random.choice(len(replay_buffer), min(10, len(replay_buffer)), replace=False)
        transitions = [replay_buffer[i] for i in transitions_id]
        for s_k, a_k, r_k, s_k_1 in transitions:
            q_k = q_dict.get((s_k, a_k))
            q_k_1_max = -10**-10
            for action in agent.action:
                q_k_i = q_dict.get((s_k_1, tuple(action)))
                if q_k_i > q_k_1_max:
                    q_k_1_max = q_k_i

            temporal_difference = r_k + domain.gamma * q_k_1_max - q_k
            q_dict[(s_k, a_k)] = q_k + alpha * temporal_difference

    return q_dict


def sarsa(domain, agent, trajectory, alpha, q_dict):
    trajectory = [(trajectory[i], trajectory[i + 1], trajectory[i + 2], trajectory[i + 3], trajectory[i+4]) for i in
                  range(0, len(trajectory) - 4, 3)]

    for s_k, a_k, r_k, s_k_1, a_k_1 in trajectory:
        q_k = q_dict.get((s_k, a_k))
        q_k_1 = q_dict.get((s_k_1, a_k_1))
        temporal_difference = r_k + domain.gamma * q_k_1 - q_k
        q_dict[(s_k, a_k)] = q_k + alpha * temporal_difference

    return q_dict


def subsection1(stocha=False):
    d = Domain()
    a = Agent4()
    h = make_trajectory(d, a, 50000, stocha)

    Q = function_q_5(d, a, h, 0.05, q_dict_ini(d, a))

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


def subsection2(stocha=False, section_2b=False, section_2c=False, section_3=False):
    nb_episode = 100
    nb_transition = 1000
    alpha = 0.05
    epsilon = 0.5
    N = 7

    # Compute J_n^µ*
    if section_3:
        d = Domain(gamma=0.4)
    else:
        d = Domain()
    a = Agent()

    Q = np.zeros([d.n, d.m])
    mu = np.zeros([d.n, d.m])

    for i in range(d.m):
        for j in range(d.n):
            q_a = []
            for action in a.action:
                q_a.append(function_q(d, a, (i, j), (action[0], action[1]), 8, stocha))

            Q[i, j] = max(q_a)
            mu[i, j] = np.argmax(q_a)

    J_n = np.zeros([d.n, d.m])
    my_a = Agent3(mu)

    for i in range(d.m):
        for j in range(d.n):
            J_n[i, j] = function_j(d, (i, j), my_a, N, stocha)

    # Compute Q^hat and J_n^hat
    Q_hat = {}
    norm_q = []
    norm_j = []
    norm_j_2 = []

    for episode in range(nb_episode):
        if section_3:
            d = Domain(gamma=0.4)
        else:
            d = Domain()

        a = Agent5(d, epsilon)
        h = make_trajectory(d, a, nb_transition, stocha)
        # print(h)
        if episode == 0:
            Q_hat = q_dict_ini(d, a)

        if section_2c:
            Q_hat = function_q_5_bis(d, a, h, alpha, Q_hat)
        else:
            Q_hat = function_q_5(d, a, h, alpha, Q_hat, section_2b)

        mu = np.zeros([d.n, d.m])
        Q_hat_np = np.zeros([d.n, d.m])
        for i in range(d.m):
            for j in range(d.n):
                state = (i, j)
                q_a = []
                for action in a.action:
                    q_a.append(Q_hat.get((state, tuple(action))))
                mu[i, j] = np.argmax(q_a)
                Q_hat_np[i, j] = q_a[np.argmax(q_a)]

        J_n_hat = np.zeros([d.n, d.m])
        my_a = Agent3(mu)

        for i in range(d.m):
            for j in range(d.n):
                J_n_hat[i, j] = function_j(d, (i, j), my_a, N, stocha)

        norm_q.append(np.linalg.norm(Q_hat_np - Q, ord=np.inf))
        norm_j.append(np.linalg.norm(J_n_hat - J_n, ord=np.inf))
        norm_j_2.append(np.linalg.norm(J_n_hat - J_n, ord=2))

    if section_3:
        print(mu)
        print("J")
        print(J_n)
        print("J^")
        print(J_n_hat)

        plt.figure()
        plt.plot(norm_q)
        plt.xlabel('episode [-]')
        plt.ylabel(r'$||\^Q||_{\infty}$ [-]')

    plt.figure()
    plt.plot(norm_j)
    plt.xlabel('episode [-]')
    plt.ylabel(r'$||J_n^{µ_\^Q} - J_n^{µ*}||_{\infty}$ [-]')

    plt.figure()
    plt.plot(norm_j_2)
    plt.xlabel('episode [-]')
    plt.ylabel(r'$||J_n^{µ_\^Q} - J_n^{µ*}||_{2}$ [-]')
    plt.show()

    return


def bonus(stocha):
    d = Domain()
    a = Agent4()
    h = make_trajectory(d, a, 50000, stocha)

    Q = sarsa(d, a, h, 0.05, q_dict_ini(d, a))

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
    print("5.1")
    subsection1(stocha)
    print("5.2a")
    subsection2(stocha)
    print("5.2b")
    subsection2(stocha, section_2b=True)
    print("5.2c")
    subsection2(stocha, section_2c=True)
    print("5.3")
    subsection2(stocha, section_3=True)
    print("5.4")
    bonus(stocha)

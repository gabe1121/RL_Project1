import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from section1 import Domain
from section2 import function_j
from section3 import function_q, Agent3
from section3 import compute_mean_reward, compute_probability


def reward_estimator(trajectory, state, action):
    trajectory = [(trajectory[i], trajectory[i+1], trajectory[i+2]) for i in range(0, len(trajectory), 3)]
    matching_indices = [k for k, (s_k, a_k, _) in enumerate(trajectory) if (s_k, a_k) == (state, action)]

    if matching_indices:
        sum_rewards = sum([trajectory[k][2] for k in matching_indices])
        occurrences = len(matching_indices)
        return sum_rewards / occurrences
    else:
        return 0


def probability_estimator(trajectory, expected_state, state, action):
    trajectory = [(trajectory[i], trajectory[i+1], trajectory[i+3]) for i in range(0, len(trajectory)-4, 3)]
    matching_indices = [k for k, (s_k, a_k, _) in enumerate(trajectory) if (s_k, a_k) == (state, action)]

    if matching_indices:
        sum_expected_stated = sum([1 if tuple(trajectory[k][2]) == tuple(expected_state) else 0 for k in matching_indices])
        occurrences = len(matching_indices)
        return sum_expected_stated / occurrences
    else:
        return 0


@lru_cache(maxsize=10000)
def function_q_hat(domain, agent, trajectory, state, action, N):
    if N == 0:
        return 0
    else:
        possible_state = []
        [[possible_state.append((i, j)) for j in range(domain.m)] for i in range(domain.n)]
        possible_action = agent.action
        item_to_sum = []
        for expected_state in possible_state:
            probability = probability_estimator(trajectory, expected_state, state, action)
            q_n1 = -10**10
            if probability != 0:
                for action_i in possible_action:
                    q_n_i = function_q_hat(domain, agent, trajectory, (expected_state[0], expected_state[1]), (action_i[0], action_i[1]), N-1)
                    if q_n_i > q_n1:
                        q_n1 = q_n_i
            item_to_sum.append(probability*q_n1)

        return reward_estimator(trajectory, state, action) + domain.gamma * sum(item_to_sum)


class Agent4:
    def __init__(self):
        self.action = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
        ]

    def chose_action(self):
        action_id = np.random.randint(0, 4)
        return self.action[action_id]


def main(stocha):
    h = []
    d = Domain()
    a = Agent4()
    possible_state = []
    [[possible_state.append([i, j]) for j in range(d.m)] for i in range(d.n)]
    norm_reward = []
    norm_probability = []
    h_vec = range(1, 1001, 100)  # TO change depending on the case considered
                                 # (increase the max range and the step size for the stchastic case)

    for t_max in h_vec:
        for _ in range(t_max):
            if stocha:
                disturbance = np.random.random()
            else:
                disturbance = 0
            state, action, _, reward, _ = d.step(a.chose_action(), disturbance)
            h.append(state)
            h.append(action)
            h.append(reward)

        reward_diff = []
        for state in possible_state:
            for action in a.action:
                reward_estimated = reward_estimator(h, state, action)
                reward = compute_mean_reward(d, state, action, stocha)
                reward_diff.append(reward - reward_estimated)

        norm_reward.append(np.linalg.norm(reward_diff, ord=np.inf))

    plt.figure()
    plt.plot(h_vec, norm_reward)
    plt.xlabel('lenght of h [-]')
    plt.ylabel(r'$||r(s,a) - \^r(s,a)||_{\infty}$ [-]')
    plt.grid()
    plt.show()

    h = []
    for t_max in h_vec:
        for _ in range(t_max):
            if stocha:
                disturbance = np.random.random()
            else:
                disturbance = 0
            state, action, _, reward, _ = d.step(a.chose_action(), disturbance)
            h.append(state)
            h.append(action)
            h.append(reward)

        probability_diff = []
        for expected_state in possible_state:
            for state in possible_state:
                for action in a.action:
                    probability_estimated = probability_estimator(h, expected_state, state, action)
                    probability = compute_probability(d, expected_state, state, action, stocha)
                    probability_diff.append(probability - probability_estimated)

        norm_probability.append(np.linalg.norm(probability_diff, ord=np.inf))

    plt.figure()
    plt.plot(h_vec, norm_probability)
    plt.xlabel('lenght of h [-]')
    plt.ylabel(r"$||p(s'|s,a) - \^p(s'|s,a)||_{\infty}$ [-]")
    plt.grid()
    plt.show()

    Q = np.zeros([d.n, d.m])
    for i in range(d.m):
        for j in range(d.n):
            q_a = []
            for action in a.action:
                q_a.append(function_q(d, a, (i, j), (action[0], action[1]), 8, stocha))

            Q[i, j] = max(q_a)

    h = []
    for t_max in h_vec:
        for _ in range(t_max):
            if stocha:
                disturbance = np.random.random()
            else:
                disturbance = 0
            state, action, _, reward, _ = d.step(a.chose_action(), disturbance)
            h.append(tuple(state))
            h.append(tuple(action))
            h.append(reward)

        Q_hat = np.zeros([d.n, d.m])
        mu = np.zeros([d.n, d.m])
        mu_print = np.zeros([d.n, d.m],dtype=object)
        for i in range(d.m):
            for j in range(d.n):
                q_a = []
                for action in a.action:
                    q_a.append(function_q_hat(d, a, tuple(h), (i, j), (action[0], action[1]), 8))

                Q_hat[i, j] = max(q_a)
                mu[i, j] = np.argmax(q_a)
                mu_print[i,j] = a.action[int(mu[i, j])]

        print(f"len(h) = {t_max}, ||Q - Q_hat|| = ", format(np.max(np.max(Q-Q_hat)),'.2f'))

    print(mu)
    
    print(mu_print)
    J_n_hat = np.zeros([d.n, d.m])
    my_a = Agent3(mu)

    for i in range(d.m):
        for j in range(d.n):
            J_n_hat[i, j] = format(function_j(d, (i, j), my_a, 981, stocha),'.2f')

    print(J_n_hat)


if __name__ == "__main__":
    main(True)




import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from section1 import Domain
from section2 import function_j
from section3 import function_q, Agent3
from section3 import compute_mean_reward, compute_probability


def make_trajectory(domain, agent, N, stocha=False):
    trajectory = []
    for _ in range(N):
        if stocha:
            disturbance = np.random.random()
        else:
            disturbance = 0
        state, action, _, reward, _ = domain.step(agent.chose_action(), disturbance)
        trajectory.append(tuple(state))
        trajectory.append(tuple(action))
        trajectory.append(reward)
    return trajectory


def reward_estimator(trajectory, state, action):
    trajectory = [(trajectory[i], trajectory[i+1], trajectory[i+2]) for i in range(0, len(trajectory), 3)]
    matching_indices = [k for k, (s_k, a_k, _) in enumerate(trajectory) if (s_k, a_k) == (tuple(state), tuple(action))]

    if matching_indices:
        sum_rewards = sum([trajectory[k][2] for k in matching_indices])
        occurrences = len(matching_indices)
        return sum_rewards / occurrences
    else:
        return 0


def build_reward_dict(trajectory, domain, agent):
    reward_dict = {}
    possible_state = []
    [[possible_state.append([i, j]) for j in range(domain.m)] for i in range(domain.n)]
    possible_action = agent.action
    for state in possible_state:
        for action in possible_action:
            reward_dict[(tuple(state), tuple(action))] = reward_estimator(trajectory, state, action)
    return reward_dict


def probability_estimator(trajectory, expected_state, state, action):
    trajectory = [(trajectory[i], trajectory[i+1], trajectory[i+3]) for i in range(0, len(trajectory)-4, 3)]
    matching_indices = [k for k, (s_k, a_k, _) in enumerate(trajectory) if (s_k, a_k) == (tuple(state), tuple(action))]

    if matching_indices:
        sum_expected_stated = sum([1 if tuple(trajectory[k][2]) == tuple(expected_state) else 0 for k in matching_indices])
        occurrences = len(matching_indices)
        return sum_expected_stated / occurrences
    else:
        return 0


def build_probability_dict(trajectory, domain, agent):
    probability_dict = {}
    possible_state = []
    [[possible_state.append([i, j]) for j in range(domain.m)] for i in range(domain.n)]
    possible_action = agent.action
    for state in possible_state:
        for next_state in possible_state:
            for action in possible_action:
                probability_dict[(tuple(state), tuple(action), tuple(next_state))] = probability_estimator(trajectory, next_state, state, action)
    return probability_dict


@lru_cache(maxsize=10000)
def function_q_hat(reward_dict_in, probability_dict_in, domain, agent, trajectory, state, action, N):
    if N == 0:
        return 0
    else:
        reward_dict = {key: value for _, (key, value) in enumerate(reward_dict_in)}
        probability_dict = {key: value for _, (key, value) in enumerate(probability_dict_in)}
        possible_state = []
        [[possible_state.append((i, j)) for j in range(domain.m)] for i in range(domain.n)]
        possible_action = agent.action
        item_to_sum = []
        for expected_state in possible_state:
            probability = probability_dict.get((tuple(state), tuple(action), tuple(expected_state)))
            q_n1 = -10**10
            if probability != 0:
                for action_i in possible_action:
                    q_n_i = function_q_hat(reward_dict_in, probability_dict_in, domain, agent, trajectory, (expected_state[0], expected_state[1]), (action_i[0], action_i[1]), N-1)
                    if q_n_i > q_n1:
                        q_n1 = q_n_i
            item_to_sum.append(probability*q_n1)

        return reward_dict.get((tuple(state), tuple(action))) + domain.gamma * sum(item_to_sum)


class Agent4:
    def __init__(self):
        self.action = [
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
        ]

    def chose_action(self):
        action_id = np.random.randint(0, 4)
        return self.action[action_id]


def main(stocha):
    d = Domain()
    a = Agent4()
    possible_state = []
    [[possible_state.append([i, j]) for j in range(d.m)] for i in range(d.n)]
    norm_reward = []
    norm_probability = []
    h_vec = range(1, 5002, 100)  # TO change depending on the case considered
                                 # (increase the max range and the step size for the stchastic case)
    if stocha:
        h_vec = [1, 100, 1000, 10000, 50000, 100000, 500000]

    for t_max in h_vec:
        h = make_trajectory(d, a, t_max, stocha)
        reward_dict = build_reward_dict(h, d, a)
        probability_dict = build_probability_dict(h, d, a)

        reward_diff = []
        probability_diff = []

        for expected_state in possible_state:
            for state in possible_state:
                for action in a.action:
                    reward = compute_mean_reward(d, state, action, stocha)
                    reward_diff.append(reward - reward_dict.get((tuple(state), tuple(action))))

                    probability = compute_probability(d, expected_state, state, action, stocha)
                    probability_diff.append(probability - probability_dict.get((tuple(state), tuple(action), tuple(expected_state))))

        norm_reward.append(np.linalg.norm(reward_diff, ord=np.inf))
        norm_probability.append(np.linalg.norm(probability_diff, ord=np.inf))

    plt.figure()
    plt.plot(h_vec, norm_reward)
    plt.xlabel('lenght of h [-]')
    plt.ylabel(r'$||r(s,a) - \^r(s,a)||_{\infty}$ [-]')
    plt.show()

    plt.figure()
    plt.plot(h_vec, norm_probability)
    plt.xlabel('lenght of h [-]')
    plt.ylabel(r"$||p(s'|s,a) - \^p(s'|s,a)||_{\infty}$ [-]")
    plt.show()

    Q = np.zeros([d.n, d.m])
    for i in range(d.m):
        for j in range(d.n):
            q_a = []
            for action in a.action:
                q_a.append(function_q(d, a, (i, j), (action[0], action[1]), 8, stocha))

            Q[i, j] = max(q_a)

    for t_max in h_vec:
        h = make_trajectory(d, a, t_max, stocha)
        reward_dict = build_reward_dict(h, d, a)
        reward_dict = tuple(sorted(reward_dict.items()))

        probability_dict = build_probability_dict(h, d, a)
        probability_dict = tuple(sorted(probability_dict.items()))

        Q_hat = np.zeros([d.n, d.m])
        mu = np.zeros([d.n, d.m])
        for i in range(d.m):
            for j in range(d.n):
                q_a = []
                for action in a.action:
                    q_a.append(function_q_hat(reward_dict, probability_dict, d, a, tuple(h), (i, j), (action[0], action[1]), 8))

                Q_hat[i, j] = max(q_a)
                mu[i, j] = np.argmax(q_a)

        print(f"len(h) = {t_max}, ||Q - Q_hat|| = ", np.max(np.max(Q-Q_hat)))

    print(mu)
    J_n_hat = np.zeros([d.n, d.m])
    my_a = Agent3(mu)

    for i in range(d.m):
        for j in range(d.n):
            J_n_hat[i, j] = function_j(d, (i, j), my_a, 50, stocha)

    print(J_n_hat)


if __name__ == "__main__":
    main(False)




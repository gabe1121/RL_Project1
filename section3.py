import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from section1 import Domain, Agent


def compute_mean_reward(domain, state, action, stocha=False):
    if stocha:
        disturbances = [domain.w - 10**-3, domain.w + 10**-3]
        probabilities = [(1-domain.w), domain.w]
    else:
        disturbances = [0]
        probabilities = [1]

    return np.sum([probability * domain.reward(state, action, disturbance)
                   for probability, disturbance in zip(probabilities, disturbances)])


def compute_probability(domain, expected_state, state, action, stocha=False):
    if stocha:
        disturbances = [domain.w - 10**-3, domain.w + 10**-3]
        probabilities = [(1-domain.w), domain.w]
    else:
        disturbances = [0]
        probabilities = [1]
    conditions = [1 if expected_state == domain.dynamic(state, action, w) else 0 for w in disturbances]

    return np.sum([probability * condition for probability, condition in zip(probabilities, conditions)])


def function_q(domain, agent, state, action, N, stocha=False):
    if N == 0:
        return 0
    else:
        possible_state = []
        [[possible_state.append([i, j]) for j in range(domain.m)] for i in range(domain.n)]
        possible_action = agent.action
        item_to_sum = []
        for expected_state in possible_state:
            probability = compute_probability(domain, expected_state, state, action, stocha)
            q_n1 = -10**10
            if probability != 0:
                for action_i in possible_action:
                    q_n_i = function_q(domain, agent, expected_state, action_i, N-1, stocha)
                    if q_n_i > q_n1:
                        q_n1 = q_n_i
            item_to_sum.append(probability*q_n1)

        return compute_mean_reward(domain, state, action, stocha) + domain.gamma * np.array(item_to_sum).sum()


d = Domain()
a = Agent()
state = [1, 3]
# print(compute_mean_reward(d, state, [0, 1]))

for n in range(10):
    # print(function_q(d, a, state, [0, 1], n))
    q_a = []
    for action in a.action:
        q_a.append(function_q(d, a, state, action, n, True))

    print(q_a)
    print(np.argmax(q_a))

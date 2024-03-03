import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from section1 import Domain
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
        sum_expected_stated = sum([1 if trajectory[k][2] == expected_state else 0 for k in matching_indices])
        occurrences = len(matching_indices)
        return sum_expected_stated / occurrences
    else:
        return 0


def function_q(domain, agent, trajectory, state, action, N):
    if N == 0:
        return 0
    else:
        possible_state = []
        [[possible_state.append([i, j]) for j in range(domain.m)] for i in range(domain.n)]
        possible_action = agent.action
        item_to_sum = []
        for expected_state in possible_state:
            probability = probability_estimator(trajectory, expected_state, state, action)
            q_n1 = -10**10
            if probability != 0:
                for action_i in possible_action:
                    q_n_i = function_q(domain, agent, trajectory, expected_state, action_i, N-1)
                    if q_n_i > q_n1:
                        q_n1 = q_n_i
            item_to_sum.append(probability*q_n1)

        return reward_estimator(trajectory, state, action) + domain.gamma * sum(item_to_sum)


class Agent:
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


stocha = True
h = []
d = Domain()
a = Agent()
possible_state = []
[[possible_state.append([i, j]) for j in range(d.m)] for i in range(d.n)]
norm_reward = []
norm_probability = []

# for t_max in range(0, 1000, 10):
#     print(t_max)
#     for _ in range(t_max):
#         if stocha:
#             disturbance = np.random.random()
#         else:
#             disturbance = 0
#         state, action, _, reward, _ = d.step(a.chose_action(), disturbance)
#         h.append(state)
#         h.append(action)
#         h.append(reward)
#
#     reward_diff = []
#     for state in possible_state:
#         for action in a.action:
#             reward_estimated = reward_estimator(h, state, action)
#             reward = compute_mean_reward(d, state, action, stocha)
#             reward_diff.append(reward - reward_estimated)
#
#     norm_reward.append(np.linalg.norm(reward_diff, ord=np.inf))
#
# plt.figure()
# plt.plot(norm_reward)
# plt.xlabel('lenght of h [-]')
# plt.ylabel(r'$||r(s,a) - \^r(s,a)||_{\infty}$ [-]')
# plt.show()

# for t_max in range(0, 100):
#     print(t_max)
#     for _ in range(t_max):
#         if stocha:
#             disturbance = np.random.random()
#         else:
#             disturbance = 0
#         state, action, _, reward, _ = d.step(a.chose_action(), disturbance)
#         h.append(state)
#         h.append(action)
#         h.append(reward)
#
#     probability_diff = []
#     for expected_state in possible_state:
#         for state in possible_state:
#             for action in a.action:
#                 probability_estimated = probability_estimator(h, expected_state, state, action)
#                 probability = compute_probability(d, expected_state, state, action, stocha)
#                 probability_diff.append(probability - probability_estimated)
#
#     norm_reward.append(np.linalg.norm(probability_diff, ord=np.inf))
#
# plt.figure()
# plt.plot(norm_reward)
# plt.xlabel('lenght of h [-]')
# plt.ylabel(r"$||p(s'|s,a) - \^p(s'|s,a)||_{\infty}$ [-]")
# plt.show()


for _ in range(100000):
    if stocha:
        disturbance = np.random.random()
    else:
        disturbance = 0
    state, action, _, reward, _ = d.step(a.chose_action(), disturbance)
    h.append(state)
    h.append(action)
    h.append(reward)


q_a = []
for action in a.action:
    q_a.append(function_q(d, a, h, [3, 0], action, 3))

print(q_a)
print(np.argmax(q_a))



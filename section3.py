import numpy as np
from section1 import Domain, Agent
from section2 import function_j
from functools import lru_cache


class Agent3:
    """
    This agent is defined such that it always chose the best policy
    """
    def __init__(self, mu):
        self.action = [
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
        ]
        self.mu = mu

    def chose_action(self, state):
        return self.action[int(self.mu[state[0], state[1]])]


def compute_mean_reward(domain, state, action, stocha=False):
    if stocha:
        disturbances = [domain.w - 10**-3, domain.w + 10**-3]
        probabilities = [(1-domain.w), domain.w]
    else:
        disturbances = [0]
        probabilities = [1]

    return sum([probability * domain.reward(state, action, disturbance)
                   for probability, disturbance in zip(probabilities, disturbances)])


def compute_probability(domain, expected_state, state, action, stocha=False):
    if stocha:
        disturbances = [domain.w - 10**-3, domain.w + 10**-3]
        probabilities = [(1-domain.w), domain.w]
    else:
        disturbances = [0]
        probabilities = [1]
    conditions = [1 if expected_state == domain.dynamic(state, action, w) else 0 for w in disturbances]

    return sum([probability * condition for probability, condition in zip(probabilities, conditions)])


@lru_cache(maxsize=10000)
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
                    q_n_i = function_q(domain, agent, (expected_state[0], expected_state[1]), (action_i[0], action_i[1]), N-1, stocha)
                    if q_n_i > q_n1:
                        q_n1 = q_n_i
            item_to_sum.append(probability*q_n1)

        return compute_mean_reward(domain, state, action, stocha) + domain.gamma * sum(item_to_sum)


def main(stocha):
    d = Domain()
    a = Agent()

    Q = np.zeros([d.n, d.m])
    mu = np.zeros([d.n, d.m])

    mu_1 = np.zeros([d.n, d.m])

    for n in range(1, 10):
        for i in range(d.m):
            for j in range(d.n):
                q_a = []
                for action in a.action:
                    q_a.append(function_q(d, a, (i, j), (action[0], action[1]), n, stocha))

                Q[i, j] = max(q_a)
                mu[i, j] = np.argmax(q_a)

        if np.array_equal(mu, mu_1):
            print(f"N = {n}, True")
        else:
            print(f"N = {n}, False")

        mu_1 = mu.copy()

    print("\n")
    print(mu)
    print("\n")

    J_n = np.zeros([d.n, d.m])
    my_a = Agent3(mu)

    for i in range(d.m):
        for j in range(d.n):
            J_n[i, j] = function_j(d, (i, j), my_a, 8, stocha)

    print(J_n)


if __name__ == "__main__":
    main(True)

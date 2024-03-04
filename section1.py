import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Domain:
    def __init__(self):
        self.current_state = [3, 0]
        self.gamma = 0.99
        self.w = 0.5
        self.n = 5
        self.m = 5

    def get_current_state(self):
        return self.current_state

    def reward(self, state, action, disturbance):
        reward_list = [
            [-3, 1, -5, 0, 19],
            [6, 3, 8, 9, 10],
            [5, -8, 4, 1, -8],
            [6, -9, 4, 19, -5],
            [-20, -17, -4, -3, 9],
        ]
        x, y = self.dynamic(state, action, disturbance)
        return reward_list[x][y]

    def step(self, action, disturbance):
        initial_state = self.current_state
        new_state = self.dynamic(initial_state, action, disturbance)
        self.current_state = new_state
        return (initial_state, action, new_state, self.reward(initial_state, action, disturbance), disturbance)

    def dynamic(self, state, action, disturbance):
        """
        It can be seen from the problem definition that the stochastic and the deterministic domains are the same if the
        disturbance is smaller or equal than 0.5. Therefore, only one function can be defined to compute the domain of
        the problem and the disturbance will not be chosen at random but will be fixed at a value smaller than 0.5 when
        treating the deterministic cases.

        :param state: state we are visiting
        :param action: action to take
        :param disturbance: random disturbance
        :return: next state
        """
        if disturbance <= self.w:
            return [min(max(state[0] + action[0], 0), self.n - 1), min(max(state[1] + action[1], 0), self.m - 1)]
        else:
            return [0, 0]


class Agent:
    def __init__(self):
        self.action = [
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
        ]

    def chose_action(self, state):
        return self.action[0]  # Always go right


def main(stocha):
    d = Domain()
    a = Agent()
    for i in range(10):
        current_action = a.chose_action(d.get_current_state())
        if stocha:
            print(d.step(current_action, np.random.random()))
        else:
            print(d.step(current_action, 0))


if __name__ == "__main__":
    main(True)








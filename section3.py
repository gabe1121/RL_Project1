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

        return compute_mean_reward(domain, state, action, stocha) + domain.gamma * sum(item_to_sum)

class mdpagent:
    def __init__(self,state_space,action_space,Qfunction):
        self.q=Qfunction
        self.states=state_space
        self.actions=action_space
    def chose_action(self,state):
        s=self.states.index(state)
        policy= self.actions[np.argmax(self.q[s])]
        return policy
    
d = Domain()
a = Agent()

state_space=[(x, y) for x in range(d.m) for y in range(d.n)] #[state_ini]
action_space=a.action
save_q=np.zeros((4,5,5))
save_policy=np.empty((5,5),dtype=object)
q_a_=np.zeros((len(state_space),len(action_space)))
# state = [0, 4]
for state in state_space:
# # print(compute_mean_reward(d, state, [0, 1]))
    s=state_space.index(state)
    for j in range (len(a.action)):
        for n in [9]:#range(10):
#           # print(function_q(d, a, state, [0, 1], n))
            q_a = []
            q_a_[s][j]=function_q(d, a, state, a.action[j], n, True)
            #q_a.append(q_a_[state][j])
        save_q[j,state[0],state[1]]=format(q_a_[s][j], '.2f')
print(save_q)
#******************************************************************************
#Optimal policy
#******************************************************************************
a1 = mdpagent(state_space,action_space,q_a_)
for state in state_space:
    s=state_space.index(state)
    save_policy[state[0],state[1]]=a1.chose_action(state)

print('Optimal policy')
print(save_policy)

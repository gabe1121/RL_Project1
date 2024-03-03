import numpy as np
from section1 import domain
from section2 import ereturn

class mdp_env:
    def __init__(self,state_space, action_space,stochastic=False):
        if stochastic:
            w=np.random.uniform(0,1)
        else:
            w=0
        self.d=domain(w)
        self.gamma = 0.95
        self.states=state_space
        self.actions=action_space

    def ptransition(self):
        num_states,num_actions=len(self.states),len(self.actions)
        transition_prob_matrix = np.zeros((num_states,num_actions,num_states),dtype=object)
        reward_matrix = np.zeros((num_states,num_actions))
        for s in range(num_states):
            for a in range(num_actions):
                reward_matrix[s,a]=self.d.reward(self.states[s],self.actions[a])
                for s_prima in range(num_states):
                    new_state=self.d.dynamic(self.states[s],self.actions[a])
                    if self.states[s_prima]==new_state:
                        transition_prob_matrix[s, a, s_prima] = 1
                    else:
                        transition_prob_matrix[s, a, s_prima] = 0
        return transition_prob_matrix, reward_matrix

    def function_q(self,Q,transition,reward, discount,N):
        num_states,num_actions=len(self.states),len(self.actions)
        # Iterate over steps
        Q_updated = np.zeros_like(Q)
        for e in range(N):
            
            for s in range(num_states):
                for a in range(num_actions):
                    Q_updated[s, a] = reward[s, a] + discount * np.sum(transition[s, a, :] * np.max(Q, axis=1))
            Q = Q_updated
        return Q

class mdpagent:
    def __init__(self,state_space,action_space,Qfunction):
        self.q=Qfunction
        self.states=state_space
        self.actions=action_space
    def chose_action(self,state):
        s=self.states.index(state)
        policy= self.actions[np.argmax(self.q[s])]
        return policy

#Get equivalent MDP with given state and action spaces----------------
state_space=[(x, y) for x in range(domain().m) for y in range(domain().n)]
action_space=[(1,0),(-1,0),(0,1),(0,-1)]
stochastic=False
MDP1=mdp_env(state_space, action_space,stochastic)
transitionmdp,rewardmdp = MDP1.ptransition()
#print(transitionmdp)
#print(rewardmdp)

#Compute QN - functions------------------------------------------------
Q=np.zeros((len(state_space),len(action_space)))
Nmax=100
policy=np.empty((Nmax,len(state_space)),dtype=object)
evaluate=np.empty(len(state_space))
expected_return=np.empty(len(state_space))
JN = ereturn()

for n in range (1,Nmax):
    QN = MDP1.function_q(Q,transitionmdp,rewardmdp, MDP1.gamma,n)
    #Optimal Policy-----------------------------------------------------
    agent1=mdpagent(state_space, action_space,QN)    
    for s in range(len(state_space)):
        policy[n][s]=[state_space[s],agent1.chose_action(state_space[s])]
        if n>1 and str(policy[n][s]) == str(policy[n-1][s]):
            evaluate[s]=1
        else:
            evaluate[s]=0
    if evaluate.all()==1:
        print("The smallest N such that a greater value does not change the policy is: ",n)
        break

save_q=np.zeros((4,5,5))
save_policy=np.empty((5,5),dtype=object)
save_j_s=np.zeros(((5,5)))
for state in state_space:
    save_j_s[state[0],state[1]]=format(JN.function_j(state, agent1, n,stochastic),'.2f')
    for j in range(len(action_space)):
        save_q[j,state[0],state[1]]=format(QN[state_space.index(state),j], '.2f')
    save_policy[state[0],state[1]]=agent1.chose_action(state)
print('QN')
print(save_q)
print('Optimal policy')
print(save_policy)
print('JN')
print(save_j_s)
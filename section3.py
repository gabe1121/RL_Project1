import numpy as np
from section1 import domain
from section2 import ereturn

class mdp_env:
    def __init__(self,state_space, action_space,stochastic=False):
        self.d=domain()
        self.gamma = 0.99
        self.states=state_space
        self.actions=action_space
        if stochastic:
            self.pw=[0.5,0.5] #[alpha,1-alpha] alpha is p(w<=0.5)
            self.w_options=[0.4,0.6] #<= 0.5 and >0.5
        else:
            self.pw=[1] # alpha is 1 
            self.w_options=[0.4] #<= 0.5

    def ptransition(self):
        num_states,num_actions=len(self.states),len(self.actions)
        transition_prob_matrix = np.zeros((num_states,num_actions,num_states),dtype=object)
        for s in range(num_states):
            for a in range(num_actions):
                for s_prima in range(num_states):
                    ptot=0
                    for w_ in range(len(self.w_options)):
                        new_state=self.d.dynamic(self.states[s],self.actions[a],self.w_options[w_])
                        if self.states[s_prima]==new_state:
                            ipw=1
                        else:
                            ipw=0
                        ptot+=ipw*self.pw[w_]
                    transition_prob_matrix[s, a, s_prima] = ptot
        return transition_prob_matrix

    def rewardsmdp(self):
        num_states,num_actions=len(self.states),len(self.actions)
        reward_matrix = np.zeros((num_states,num_actions))
        for s in range(num_states):
            for a in range(num_actions):
                rtot=0
                for w_ in range(len(self.w_options)):
                    rtot+=self.pw[w_]*self.d.reward(self.states[s],self.actions[a],self.w_options[w_])
                reward_matrix[s,a]=rtot
        return reward_matrix
    
    def function_q(self,Q,transition,reward, discount,N):
        num_states,num_actions=len(self.states),len(self.actions)
        # Iterate over steps
        for e in range(N):
            Q_updated = np.zeros_like(Q)
            for s in range(num_states):
                for a in range(num_actions):
                    Q_updated[s,a] = reward[s,a]+discount * np.sum(transition[s,a,:]*np.max(Q,axis=1))
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

"""
Get equivalent MDP with given state and action spaces
"""
m,n=5,5
state_space=[(x, y) for x in range(m) for y in range(n)]
action_space=[(1,0),(-1,0),(0,1),(0,-1)]
stochastic=True
domaintype=['deterministic', 'non-deterministic']
MDP1=mdp_env(state_space, action_space, stochastic)
transitionmdp,rewardmdp = MDP1.ptransition(),MDP1.rewardsmdp()

"""
Find smallest N such that a greater value does not change the policy inferred 
from the last Q-function of the sequence, test for Nmax steps
"""
Q=np.zeros((len(state_space),len(action_space)))
Nmax=100
policy=np.empty((Nmax,m,n),dtype=object)
evaluate=np.empty(len(state_space))

for n in range (1,Nmax):
    #Compute QN - functions---------------------------------------------
    QN = MDP1.function_q(Q,transitionmdp,rewardmdp, MDP1.gamma,n)
    #Optimal Policy-----------------------------------------------------
    agent1=mdpagent(state_space, action_space,QN)    
    for s in range(len(state_space)):
        pos=state_space[s]
        policy[n,pos[0],pos[1]]=agent1.chose_action(state_space[s])
        if n>1 and str(policy[n,pos[0],pos[1]]) == str(policy[n-1,pos[0],pos[1]]):
            evaluate[s]=1
        else:
            evaluate[s]=0
    if evaluate.all()==1:
        print("The smallest N such that a greater value does not change the policy is: ",n-1)
        break


"""
Print the last tree policies for the QN sequence
"""
#"""
for p in [n-2,n-1,n]:
    print('N=',p)
    print(policy[p])

#"""
    
steps=n-1
print('Domain: ',domaintype[stochastic], 'N=',steps)
"""
Get expected return for each state and display value functions J and Q and optimal policy
"""
JN = ereturn()
save_q=np.zeros((4,5,5))
save_op_policy=np.empty((5,5),dtype=object)
save_j_s=np.zeros(((5,5)))
for state in state_space:
    save_j_s[state[0],state[1]]=format(JN.function_j(state, agent1, steps,stochastic),'.2f')
    for j in range(len(action_space)):
        save_q[j,state[0],state[1]]=format(QN[state_space.index(state),j], '.2f')
    save_op_policy[state[0],state[1]]=agent1.chose_action(state)
print('Action space:',action_space)
print('QN(s,a)')
print(save_q)
print('Optimal policy(s)')
print(save_op_policy)
print('JN(s)_*','N=',steps)
print(save_j_s)
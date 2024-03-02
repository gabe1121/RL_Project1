import numpy as np
from section1 import domain,agent
import matplotlib.pyplot as plt

class ereturn:
    def __init__(self):
        self.gamma = 0.99

    def function_j(self, state, agent, N,sthocastic):
        if sthocastic:
            dist=np.random.uniform(0,1)
        else:
            dist=0
        self.d=domain(w=dist)
        action=agent.chose_action(state)
        new_state=self.d.dynamic(state,action)
        r=self.d.reward(new_state,action)
        if N==0:
            return 0
        else:
            #print(N,new_state,action,r)
            rew=r+self.gamma*self.function_j(new_state,agent,N-1,sthocastic)    
        return rew

#Find steps
gamma = ereturn().gamma
reward_bound=19
epsilon,max_N=9e-2,1e10
j_diff_bound=[]
for N in range(int(max_N)):
    j_diff_bound.append((reward_bound/(1-gamma))*gamma**N)
    if j_diff_bound[-1] <=epsilon:
        break

#plt.figure()
#plt.plot(j_diff_bound)
#plt.xlabel('N [-]')
#plt.ylabel(r"$\frac{\gamma^N}{1-\gamma}B_r$ [-]")
#plt.grid()
#plt.show()

steps=10#N
initial_state=(3,0)
sthocastic=True
domaintype=['deterministic', 'non-deterministic']
if sthocastic:
    episodes=20000
else:
    episodes=1
save_j=np.zeros((episodes))

state_space=[(x, y) for x in range(domain().m) for y in range(domain().n)]
save_j_s_av=np.zeros((domain().m,domain().n))
save_j_s_sd=np.zeros((domain().m,domain().n))
for initial_state in state_space:
    #print("Initial state: ",initial_state,'Domain: ',domaintype[sthocastic])
    for e in range (episodes):
         a = agent()
         er=ereturn()
         cum_return=er.function_j(initial_state, a, steps,sthocastic)
         save_j[e]=(cum_return)
         #print('for N=',steps, 'return is ', cum_return )
    mean=np.average(save_j)
    sd=np.sqrt(np.var(save_j))
    #print('Average total reward for',episodes,'episodes','of',steps,'steps each:',
    #   format(mean, '.2f'),'SD',format(sd, '.2f'))
    save_j_s_av[initial_state[0],initial_state[1]]=format(mean, '.2f')
    save_j_s_sd[initial_state[0],initial_state[1]]=format(sd, '.2f')
print(save_j_s_av)
print(save_j_s_sd)
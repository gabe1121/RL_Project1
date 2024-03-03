import numpy as np
from section1 import domain,agent
import matplotlib.pyplot as plt

class ereturn:
    def __init__(self):
        self.gamma = 0.99
        self.d=domain()
    def function_j(self, state, agent, N,stochastic):
        if stochastic:
            pw=[0.5,0.5] #[alpha,1-alpha] alpha is p(w<=0.5)
            w_options=[0.4,0.6] #<= 0.5 and >0.5
        else:
            pw=[1]  #alpha is 1 
            w_options=[0.4] #<= 0.5
        if N==0:
            return 0
        else:
            action=agent.chose_action(state)
            rew=0
            for w_ in range (len(w_options)):
                new_state=self.d.dynamic(state,action,w_options[w_])
                r=self.d.reward(new_state,action,w_options[w_])
                #print(N,new_state,action,r)
                rew+=pw[w_]*(r+self.gamma*self.function_j(new_state,agent,N-1,stochastic))   
        return rew

compute=False
if compute:
    #Find steps
    gamma = 0.99
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

    steps=20
    initial_state=(3,0)
    stochastic=True
    domaintype=['deterministic', 'non-deterministic']

    m,n=5,5
    state_space=[(x, y) for x in range(m) for y in range(n)]
    save_j=np.zeros((m,n))
    for initial_state in state_space:
        #print("Initial state: ",initial_state,)
        a = agent()
        er=ereturn()
        cum_return=er.function_j(initial_state, a, steps,stochastic)
        #print('for N=',steps, 'return is ', cum_return )
        #print('Average total reward for',episodes,'episodes','of',steps,'steps each:',
        #   format(mean, '.2f'),'SD',format(sd, '.2f'))
        save_j[initial_state[0],initial_state[1]]=format(cum_return, '.2f')
    print('Cumulative reward J_N(s) for Domain: ',domaintype[stochastic], 'N=',steps)
    print(save_j)
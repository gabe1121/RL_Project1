import numpy as np
from section1 import domain
from section1 import agent

class ereturn:
    def __init__(self):
        self.gamma = 0.95
        self.d=domain()
    def function_j(self, state, agent, N):
        self.d.current_state=state
        action=agent.chose_action(self.d.get_current_state())
        self.d.step(action)
        r=self.d.reward(self.d.get_current_state(),action)
        if N==0:
            return 0
        else:
            rew=r+self.gamma*self.function_j(self.d.get_current_state(),agent,N-1)
            return rew
a = agent()
init_state=(3,0)
n=8
print('for N=',n, 'return is ', ereturn().function_j(init_state, a, n))

#max_reward=19
#cum_return=[]
#episodes=np.arange(10,10000,10)
# for n in episodes: 
#     bound=ereturn().gamma**n*max_reward/(1-ereturn().gamma)
#     cum_return.append(ereturn().function_j(init_state, a, n))
#     print('for N=',n, 'return is ', ereturn().function_j(init_state, a, n))
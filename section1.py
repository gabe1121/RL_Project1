import numpy as np
class domain: 
    def __init__(self):
        self.current_state = (3,0)
        self.m,self.n=5,5
        self.w=np.random.uniform(0,1)
                                        
    def get_current_state(self):
        return self.current_state
        
    def reward(self, state, action):
        r=np.array([[-3,1,-5,0,19],
                   [6,3,8,9,10],
                   [5,-8,4,1,-8],
                   [6,-9,4,19,-5],
                   [-20,-17,-4,-3,9]])
        return r[state[0]][state[1]]
        
    def step(self, action):
        current_state=self.get_current_state()
        new_state=self.dynamic(current_state,action)
        self.current_state=new_state
        return (current_state,action,new_state,self.reward(new_state,action))

    def dynamic(self,state, action):
        print(self.w)
        if self.w<=0.5:
            return (min(max(state[0]+action[0],0),self.n-1),min(max(state[1]+action[1],0),self.m-1))
        else:
            return (0,0)
    
class agent:
    def __init__(self):
        pass
        
    def chose_action(self,state):
        self.action_space=np.array([(1,0),(-1,0),(0,1),(0,-1)])
        policy= (0,1)#self.action_space[np.random.randint(0,4)]
        return policy

d = domain()
a = agent()
for i in range(10):
    current_action = a.chose_action(d.get_current_state())
    #print(d.step(current_action))
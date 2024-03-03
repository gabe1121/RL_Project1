import numpy as np
class domain: 
    def __init__(self):
        self.current_state = (0,0)
        self.m,self.n=5,5
                                        
    def get_current_state(self):
        return self.current_state
        
    def reward(self, state, action,w):
        r=np.array([[-3,1,-5,0,19],
                   [6,3,8,9,10],
                   [5,-8,4,1,-8],
                   [6,-9,4,19,-5],
                   [-20,-17,-4,-3,9]])
        return r[state[0]][state[1]]
        
    def step(self, action,w):
        current_state=self.get_current_state()
        new_state=self.dynamic(current_state,action,w)
        self.current_state=new_state
        return (current_state,action,new_state,self.reward(new_state,action,w))

    def dynamic(self,state, action,w):
        if w<=0.5:
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

compute=False
if compute:
    stochastic=True
    steps=10
    initial_state=(3,0)
    domaintype=['deterministic', 'non-deterministic']
    if stochastic:
        episodes=10
    else:
        episodes=1
    savereward=np.zeros((episodes))
    etoprint=np.random.randint(0,episodes)#pick a random episode to print
    print("Initial state: ",initial_state,'Domain: ',domaintype[stochastic])
    print('Printing episode',etoprint+1,'of',episodes)
    for e in range (episodes):
        d = domain() #independent episodes
        a = agent()
        d.current_state=initial_state
        for i in range(steps):
            if stochastic:
                w=np.random.uniform(0,1)
            else:
                w=0
            current_action = a.chose_action(d.get_current_state())
            step=d.step(current_action,w)
            if e== etoprint: #Print only one episode
                print('From state', step[0],'take action', step[1],
                ' New state:', step[2], 'with reward',step[3])
            savereward[e]+=step[3]

    mean=np.average(savereward)
    sd=np.sqrt(np.var(savereward))
    print('Average total reward for',episodes,'episodes','of',steps,'steps each:',
        format(mean, '.2f'),'SD',format(sd, '.2f'))
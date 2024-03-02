import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from section1 import Domain, Agent


d = Domain()
a = Agent()
state_ini = [3, 0]

#Find steps (N)
gamma =d.gamma
reward_bound=19
epsilon,max_N=9e-2,1e10
j_diff_bound=[]
for N in range(int(max_N)):
    j_diff_bound.append((reward_bound/(1-gamma))*gamma**N)
    if j_diff_bound[-1] <=epsilon:
        break

plt.figure()
plt.plot(j_diff_bound)
plt.xlabel('N [-]')
plt.ylabel(r"$\frac{\gamma^N}{1-\gamma}B_r$ [-]")
plt.grid()
plt.show()
#N = 900

J_N = [0]
#for n in range(1, N):
#    print(n)
#    J_N.append(d.function_j(state_ini, a, n, False))
#    print(J_N[-1]-J_N[-2])

sthocastic=True
domaintype=['deterministic', 'non-deterministic']
if sthocastic:
    N=10
print('N=',N)
state_space=[(x, y) for x in range(Domain().m) for y in range(Domain().n)] #[state_ini]
save_j_s=np.zeros((Domain().m,Domain().n))
for state_ini in state_space:
    #print("Initial state: ",state_ini,'Domain: ',domaintype[sthocastic])
    J_N=d.function_j(state_ini, a, N,sthocastic)
    save_j_s[state_ini[0],state_ini[1]]=format(J_N, '.2f')
print(save_j_s)




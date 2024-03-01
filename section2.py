import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from section1 import Domain, Agent


d = Domain()
a = Agent()
state_ini = [3, 0]
N = 900

J_N = [0]

for n in range(1, N):
    print(n)
    J_N.append(d.function_j(state_ini, a, n, False))
    print(J_N[-1]-J_N[-2])

plt.figure()
plt.plot(J_N)
plt.xlabel('N [-]')
plt.ylabel(r'$J_N^µ$ [-]')
plt.show()


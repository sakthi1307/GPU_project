import networkx as nx
from kernel import *
from ode import *
import numpy as np
from numba import cuda
import sys


N = 1000
#15x15
n = 15 
G = nx.watts_strogatz_graph(n = n, p = 0.5, k =4 )
Am = nx.to_numpy_matrix(G)
Am = np.array(Am, dtype=np.float32)

tspan = 100000

ϵ=0.005#Coupling Strength
a = -0.025
b = 0.0065
α = 0.8
fpa = 0.02
ω = 0.001
gamma = 1
R = 25
p= [α,n,a,b,ω,ϵ,fpa,gamma,R]
p = np.array(p, dtype=np.float32)


# x = cuda.device_array((N, tspan+1, n, 2), dtype=np.float32)

# x = np.zeros(N,tspan+1,n,2)
x = np.zeros(shape=(N,tspan+1,n,2), dtype=np.float32)
x0 = np.random.rand(N,n,2)
x0 = np.array(x0, dtype=np.float32)
print("x0",x0[1,1])
solve[N,n](x,x0,p,Am,tspan)




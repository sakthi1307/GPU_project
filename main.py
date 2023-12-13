import networkx as nx
from kernel import *
from ode import *
import numpy as np
from numba import cuda
import sys


N = 100
#15x15
n = 15 
G = nx.watts_strogatz_graph(n = n, p = 0.5, k =4 )
Am = nx.to_numpy_matrix(G)
Am = np.array(Am,dtype=object)

tspan = 100

ϵ=0.005#Coupling Strength
a = -0.025
b = 0.0065
α = 0.8
fpa = 0.02
ω = 0.001
gamma = 1
R = 25
p= [α,n,a,b,ω,ϵ,fpa,gamma,R,Am]
p = np.array(p)

x = np.random.rand(N,tspan+1,n,2)
x = np.array(x)
x0 = np.random.rand(N,n,2)
x0 = np.array(x0)

solve[1,n](lo,p,Am,tspan,x0,n,x,True)

res = d_x.copy_to_host()



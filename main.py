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
p = np.array(p,dtype=object)


x = np.random.rand(N,tspan+1,n,2)
x = np.array(x,dtype=object)
x0 = np.random.rand(N,n,2)
x0 = np.array(x0,dtype=object)
print("takes this size",sys.getsizeof(x))
d_x = cuda.to_device(x) #,dtype=object)
d_x0 = cuda.to_device(x0) #,dtype=object)
d_p = cuda.to_device(p) #,dtype=object)
Am = cuda.to_device(Am)
# print(d_p.shape)
solve[1,n]()#lo,d_p,Am,tspan,d_x0,n,d_x,True,1e-8)

res = d_x.copy_to_host()



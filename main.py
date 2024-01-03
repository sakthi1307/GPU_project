import networkx as nx
from kernel import *
from ode import *
import numpy as np
from numba import cuda
import sys
import time
from peaks_1d import *
from seq import *

N = 100
#15x15 nodes/particles
n = 15
G = nx.watts_strogatz_graph(n = n, p = 0.5, k =4 )
Am = nx.to_numpy_matrix(G)
#adajency matrix
Am = np.array(Am, dtype=np.float32)

tspan = 10000

ϵ=0.0012#Coupling Strength
a = -0.025
b = 0.0065
α = 0.8
fpa = 0.02
ω = 0.001
gamma = 1
R = 25
#parameters list
p= [α,n,a,b,ω,ϵ,fpa,gamma,R]
p = np.array(p, dtype=np.float32)

x = np.zeros(shape=(N,tspan+1,n,2), dtype=np.float32)
#initial conditions
x0 = np.random.rand(N,n,2)
x0 = np.array(x0, dtype=np.float32)
#initial conditions copy for sqeual implementation
x0_2 = np.array(x0, dtype=np.float32)

timer = -time.time()
solve[N,n](x,x0,p,Am,tspan)
timer += time.time()
print("execution time for CUDA implementation:",timer)



# testing CPU sequential implementation
x = np.zeros(shape=(N,tspan+1,n,2), dtype=np.float32)
timer = -time.time()
for k in range(N):
    solve_seq(x,x0_2,p,Am,tspan,k)
timer += time.time()
print("execution time for sequential implementation:",timer)





num_elements = 1000
x_host = np.ascontiguousarray(x[0,:,1,0])

# Host arrays for results
midpoints_host = np.zeros(1, dtype=np.int32)
left_edges_host = np.zeros(1, dtype=np.int32)
right_edges_host = np.zeros(1, dtype=np.int32)

# Transfer host arrays to device
x_device = cuda.to_device(x_host)
midpoints_device = cuda.to_device(midpoints_host)
left_edges_device = cuda.to_device(left_edges_host)
right_edges_device = cuda.to_device(right_edges_host)

# Set up the grid and block dimensions
threads_per_block = 32
blocks_per_grid = 1  # One block

# Launch the kernel
find_peaks_cuda[blocks_per_grid, threads_per_block](x_device, midpoints_device, left_edges_device, right_edges_device)

# Copy the results back to the host
midpoints_device.copy_to_host(midpoints_host)
left_edges_device.copy_to_host(left_edges_host)
right_edges_device.copy_to_host(right_edges_host)

# Print or use the results as needed
print("Midpoint:", midpoints_host[0])
print("Left Edge:", left_edges_host[0])
print("Right Edge:", right_edges_host[0])

import numpy as np
from numba import njit
n1 = 100
dim = 2

def lo(l_x,u,p,Am,i):
    α,n,a,b,ω,ϵ,fpa,gamma,R=p
    second_degree = 0
    for j in range(n1):
        second_degree = second_degree + Am[i,j]*((u[j,1]-u[i,0]))
    l_x[0] = (-u[i,0]*(u[i,0]-a)*(u[i,0]-1))-(gamma*u[i,1]) + (ϵ/2)*second_degree
    l_x[1] = b*((gamma*u[i,0]) - fpa*u[i,1])
    return(l_x)

def solve_seq(x,x0,p,Am,tspan,sys_id):
    ds_X = np.zeros(shape=(n1,dim),dtype=np.float32)
    ds_X0 = np.zeros(shape=(n1,dim),dtype=np.float32)
    ds_step = np.zeros(shape=(n1,dim),dtype=np.float32)
    l_x = np.zeros(shape=(dim,1),dtype=np.float32)
    for i in range(n1):
        for k in range(dim):
            x[sys_id,0,i,k] = x0[sys_id,i,k]
            ds_X0[i,k] = x0[sys_id,i,k] 
        
        
    for t in range(tspan):
        for i in range(n1):
            k1 = lo(l_x,ds_X0,p,Am,i)    
            for k in range(dim):
                ds_X0[i,k] = k1[k][0]  
                x[sys_id,t,i,k] = k1[k][0]
    
    return x    

@njit   
def lo_optimized(l_x,u,p,Am,i):
    α,n,a,b,ω,ϵ,fpa,gamma,R=p
    second_degree = 0
    for j in range(n1):
        second_degree = second_degree + Am[i,j]*((u[j,1]-u[i,0]))
    l_x[0] = (-u[i,0]*(u[i,0]-a)*(u[i,0]-1))-(gamma*u[i,1]) + (ϵ/2)*second_degree
    l_x[1] = b*((gamma*u[i,0]) - fpa*u[i,1])
    return(l_x)


@njit(parallel=True)    
def solve_seq(x,x0,p,Am,tspan,sys_id):
    ds_X = np.zeros(shape=(n1,dim),dtype=np.float32)
    ds_X0 = np.zeros(shape=(n1,dim),dtype=np.float32)
    ds_step = np.zeros(shape=(n1,dim),dtype=np.float32)
    l_x = np.zeros(shape=(dim,1),dtype=np.float32)
    for i in range(n1):
        for k in range(dim):
            x[sys_id,0,i,k] = x0[sys_id,i,k]
            ds_X0[i,k] = x0[sys_id,i,k] 
        
        
    for t in range(tspan):
        for i in range(n1):
            k1 = lo_optimized(l_x,ds_X0,p,Am,i)    
            for k in range(dim):
                ds_X0[i,k] = k1[k][0]  
                x[sys_id,t,i,k] = k1[k][0]
    
    return x    


import numba
from numba import cuda,float32
import numpy as np
n1 = 15
dim = 2

@cuda.jit
def lo(l_x,u,p,Am,i):
    α,n,a,b,ω,ϵ,fpa,gamma,R=p
    second_degree = 0
    for j in range(n):
        second_degree = second_degree + Am[i,j]*((u[2,j]-u[1,i]))
    l_x[0] = (-u[1,i]*(u[1,i]-a)*(u[1,i]-1))-(gamma*u[2,i]) + (ϵ/(2))*second_degree
    l_x[1] = b*((gamma*u[1,i]) - fpa*u[2,i])
    return(l_x)

@cuda.jit
def solve(x,x0,p,Am,tspan):
    sys_id = cuda.blockIdx.x
    ds_X = cuda.shared.array(shape=(n1,dim),dtype=float32)
    ds_X0 = cuda.shared.array(shape=(n1,dim),dtype=float32)
    ds_step = cuda.shared.array(shape=(n1,dim),dtype=float32)
    l_x = cuda.local.array(shape=(dim,1),dtype=float32)
    i = cuda.threadIdx.x
    # load the network into shared memory assuming that always blockdim == n
    for k in range(dim):
        x[sys_id,0,i,k] = x0[sys_id,i,k]
    for k in range(dim):
        ds_X0[i,k] = x0[sys_id,i,k] 
    cuda.syncthreads()
    
    for t in range(tspan):
        k1 = lo(l_x,ds_X0,p,Am,i)    
        for k in range(dim):
            ds_X0[i,k] = k1[0,k]  
        for k in range(dim):
            x[sys_id,t,i,k] = k1[0,k]
        cuda.syncthreads()


@cuda.jit
def solve_ode(x,x0,p,Am,tspan):
        if dim==2:
            h = ds_step[i]
            k1 = f(ds_X0[0],ds_X0[1],p,i)
            k2 = f(x+h/4,y+h/4*k1,p,i)
            k3 = f(x+3*h/8,y+3*h/32*k1+9*h/32*k2,p,i)
            k4 = f(x+12*h/13,y+1932/2197*h*k1-7200/2197*h*k2+7296/2197*h*k3,p,i)
            k5 = f(x+h,y+439/216*h*k1-8*h*k2+3680/513*h*k3-845/4104*h*k4,p,i)
            k6 = f(x+h/2,y-8/27*h*k1+2*h*k2-3544/2565*h*k3+1859/4104*h*k4-11/40*h*k5,p,i)
            y1 = y+h*(25/216*k1+1408/2565*k3+2197/4104*k4-1/5*k5)
            z1 = y+h*(16/135*k1+6656/12825*k3+28561/56430*k4-9/50*k5+2/55*k6)
            s = 0.84*(h*0.0001/abs(z1-y1))**(1/4)
            ds_step[i] = s*h

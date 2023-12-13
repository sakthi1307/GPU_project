
import numba
from numba import cuda
import numpy as np
n1 = 15
@cuda.jit
def solve(f,p,Am,tspan,x0,n,d_x,discrete):
    sys_id = cuda.blockIdx.x
    dim = x0.shape[0]
    i32_arr = cuda.shared.array(0, dtype=np.int32)
    ds_X = cuda.shared.array(shape=10)
    print(sys_id)
    ds_X0 = cuda.shared.array(shape=d_x0.shape)
    ds_step = cuda.shared.array(shape=n1)
    i = cuda.threadIdx.x
    d_x[sys_id,0,:,0] = x0[:,0]
    d_x[sys_id,0,:,1] = x0[:,1]
    for k in range(dim):
        ds_X0[i,k] = x0[sys_id,i,k]  
    cuda.syncthreads()
    if discrete:
        for t in range(tspan):
            # load the network into shared memory assuming that always blockdim == n
            #f(x_i+1,y_i+1)=[xi,yi]
            k1 = f(ds_X,ds_X0,p,Am,i)
            for k in range(dim):
                ds_X0[i,k] = k1[k]  
            cuda.syncthreads()
            for k in range(dim):
                d_x[sys_id,t,:,k] = ds_X0[:,k]


    # else:
    #     if dim==2:
    #     #z_i+1=f(x_i,y_i)
    #         h = ds_step[i]
    #         k1 = f(ds_X0[0],ds_X0[1],p,i)
    #         k2 = f(x+h/4,y+h/4*k1,p,i)
    #         k3 = f(x+3*h/8,y+3*h/32*k1+9*h/32*k2,p,i)
    #         k4 = f(x+12*h/13,y+1932/2197*h*k1-7200/2197*h*k2+7296/2197*h*k3,p,i)
    #         k5 = f(x+h,y+439/216*h*k1-8*h*k2+3680/513*h*k3-845/4104*h*k4,p,i)
    #         k6 = f(x+h/2,y-8/27*h*k1+2*h*k2-3544/2565*h*k3+1859/4104*h*k4-11/40*h*k5,p,i)
    #         y1 = y+h*(25/216*k1+1408/2565*k3+2197/4104*k4-1/5*k5)
    #         z1 = y+h*(16/135*k1+6656/12825*k3+28561/56430*k4-9/50*k5+2/55*k6)
    #         s = 0.84*(h*0.0001/abs(z1-y1))**(1/4)
    #         ds_step[i] = s*h

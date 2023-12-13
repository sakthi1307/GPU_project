import numba

@numba.jit
def lo(ds_X,u,p,Am,i):
    α,n,a,b,ω,ϵ,fpa,gamma,R=p
    ds_X[i,0] = (-u[1,i]*(u[1,i]-a)*(u[1,i]-1))-(gamma*u[2,i]) + (ϵ/(2))*sum([(Am[i,j]*((u[2,j]-u[1,i]))) for j in range(n)]) 
    ds_X[i,1] = b*((gamma*u[1,i]) - fpa*u[2,i])
    return ds_X
        

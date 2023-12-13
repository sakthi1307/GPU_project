from numba import cuda
import numpy as np
from scipy.spatial import ConvexHull

random_points = np.random.uniform(0, 6, (1000, 2))
result = np.full(random_points.shape[0], True)
hull = ConvexHull(np.array([(1, 2), (3, 4), (3, 6), (2, 4.5), (2.5, 5)]))

@cuda.jit
def my_kernel(points, eq, result):
    pos = cuda.grid(1)

    if pos < points.shape[0]:
        for i in range(eq.shape[0]):
            temp = 0.
            for j in range(points.shape[1]):
                temp += eq[i,j]*points[pos,j]
            temp += eq[i,-1]
            if temp > 1e-12:
                result[pos] = False
    

# Host code   
threadsperblock = 256
blockspergrid = int(np.ceil(random_points.shape[0] / threadsperblock))
my_kernel[blockspergrid, threadsperblock](random_points, hull.equations, result)
print(result)
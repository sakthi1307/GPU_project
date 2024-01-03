import numpy as np
from numba import cuda, float64, int32

@cuda.jit
def find_peaks_cuda(x, midpoints, left_edges, right_edges):
    i_global = cuda.grid(1)
    stride = cuda.blockDim.x * cuda.gridDim.x
    i_max = x.shape[0] - 1

    while i_global < i_max:
        i = i_global  # Pointer to current sample, first one can't be maxima
        i_max_local = min(i + stride, i_max)  # Last sample can't be maxima
        while i < i_max_local:
            # Test if previous sample is smaller
            if x[i - 1] < x[i]:
                i_ahead = i + 1  # Index to look ahead of current sample

                # Find next sample that is unequal to x[i]
                while i_ahead < i_max_local and x[i_ahead] == x[i]:
                    i_ahead += 1

                # Maxima is found if next unequal sample is smaller than x[i]
                if x[i_ahead] < x[i]:
                    idx = cuda.atomic.add(midpoints, 1, 0)
                    left_edges[idx] = i
                    right_edges[idx] = i_ahead - 1
                    # Skip samples that can't be maximum
                    i = i_ahead
            i += 1

        i_global += stride
        cuda.syncthreads()  # Synchronize threads to ensure consistent results

# Example usage
# num_elements = 1000
# x_host = np.random.rand(num_elements).astype(np.float64)

# # Host arrays for results
# midpoints_host = np.zeros(1, dtype=np.int32)
# left_edges_host = np.zeros(1, dtype=np.int32)
# right_edges_host = np.zeros(1, dtype=np.int32)

# # Transfer host arrays to device
# x_device = cuda.to_device(x_host)
# midpoints_device = cuda.to_device(midpoints_host)
# left_edges_device = cuda.to_device(left_edges_host)
# right_edges_device = cuda.to_device(right_edges_host)

# # Set up the grid and block dimensions
# threads_per_block = 32
# blocks_per_grid = 1  # One block

# # Launch the kernel
# find_peaks_cuda[blocks_per_grid, threads_per_block](x_device, midpoints_device, left_edges_device, right_edges_device)

# # Copy the results back to the host
# midpoints_device.copy_to_host(midpoints_host)
# left_edges_device.copy_to_host(left_edges_host)
# right_edges_device.copy_to_host(right_edges_host)

# # Print or use the results as needed
# print("Midpoint:", midpoints_host[0])
# print("Left Edge:", left_edges_host[0])
# print("Right Edge:", right_edges_host[0])
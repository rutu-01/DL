import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# CUDA kernel for BFS traversal
cuda_kernel = """
__global__ void bfs(int* graph, int* visited, int* queue, int* result, int start, int num_nodes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        visited[start] = 1;
        queue[0] = start;
        result[0] = start;
    }
    __syncthreads();
    
    int queue_front = 0;
    int queue_rear = 1;
    
    while (queue_front < queue_rear) {
        int current_node = queue[queue_front];
        queue_front++;

        for (int neighbor = 0; neighbor < num_nodes; neighbor++) {
            if (graph[current_node * num_nodes + neighbor] == 1 && visited[neighbor] == 0) {
                visited[neighbor] = 1;
                queue[queue_rear] = neighbor;
                queue_rear++;
                result[queue_rear - 1] = neighbor;
            }
        }
        __syncthreads();
    }
}
"""

# Compile CUDA kernel
mod = SourceModule(cuda_kernel)

# Get kernel function
bfs_kernel = mod.get_function("bfs")

def bfs_cuda(graph, start):
    num_nodes = len(graph)
    
    # Allocate memory on GPU
    graph_gpu = cuda.to_device(np.array(graph, dtype=np.int32))
    visited_gpu = cuda.mem_alloc(num_nodes * np.int32().itemsize)
    queue_gpu = cuda.mem_alloc(num_nodes * np.int32().itemsize)
    result_gpu = cuda.mem_alloc(num_nodes * np.int32().itemsize)

    # Initialize arrays on GPU
    visited_init = np.zeros(num_nodes, dtype=np.int32)
    queue_init = np.zeros(num_nodes, dtype=np.int32)

    cuda.memcpy_htod(visited_gpu, visited_init)
    cuda.memcpy_htod(queue_gpu, queue_init)

    # Launch kernel
    block_size = 256
    grid_size = (num_nodes + block_size - 1) // block_size
    bfs_kernel(graph_gpu, visited_gpu, queue_gpu, result_gpu, np.int32(start), np.int32(num_nodes), block=(block_size, 1, 1), grid=(grid_size, 1))

    # Retrieve result from GPU
    result = np.empty(num_nodes, dtype=np.int32)
    cuda.memcpy_dtoh(result, result_gpu)

    return result

# Example graph represented as an adjacency matrix
graph = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]

start_node = 0
result = bfs_cuda(graph, start_node)
print("BFS result starting from node {}: {}".format(start_node, result))

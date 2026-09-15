#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// -----------------------------
// BFS KERNEL
// -----------------------------
__global__ void bfs_kernel(
    const int *row_ptr,
    const int *col_idx,
    int *frontier,
    int frontier_size,
    int *next_frontier,
    int *next_size,
    int *visited)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int u = frontier[tid];

    int start = row_ptr[u];
    int end   = row_ptr[u + 1];

    for (int e = start; e < end; e++)
    {
        int v = col_idx[e];

        if (atomicCAS(&visited[v], 0, 1) == 0)
        {
            int pos = atomicAdd(next_size, 1);
            next_frontier[pos] = v;
        }
    }
}

// -----------------------------
// HOST BFS DRIVER
// -----------------------------
void bfs_gpu(
    int num_nodes,
    int *d_row_ptr,
    int *d_col_idx,
    int source)
{
    int *d_frontier, *d_next_frontier;
    int *d_visited;
    int *d_frontier_size, *d_next_size;

    cudaMalloc(&d_frontier, num_nodes * sizeof(int));
    cudaMalloc(&d_next_frontier, num_nodes * sizeof(int));
    cudaMalloc(&d_visited, num_nodes * sizeof(int));

    cudaMalloc(&d_frontier_size, sizeof(int));
    cudaMalloc(&d_next_size, sizeof(int));

    cudaMemset(d_visited, 0, num_nodes * sizeof(int));

    int h_frontier_size = 1;
    cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier_size, &h_frontier_size, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_next_size, 0, sizeof(int));
    cudaMemset(&d_visited[source], 1, sizeof(int));

    const int blockSize = 256;

    while (true)
    {
        int h_size;
        cudaMemcpy(&h_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_size == 0) break;

        int gridSize = (h_size + blockSize - 1) / blockSize;

        cudaMemset(d_next_size, 0, sizeof(int));

        bfs_kernel<<<gridSize, blockSize>>>(
            d_row_ptr,
            d_col_idx,
            d_frontier,
            h_size,
            d_next_frontier,
            d_next_size,
            d_visited
        );

        cudaDeviceSynchronize();

        std::swap(d_frontier, d_next_frontier);
        cudaMemcpy(d_frontier_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_visited);
    cudaFree(d_frontier_size);
    cudaFree(d_next_size);
}

// -----------------------------
// SIMPLE GRAPH (TOY CSR)
// -----------------------------
void build_toy_graph(
    std::vector<int> &row_ptr,
    std::vector<int> &col_idx)
{
    // Grafo semplice:
    // 0 -> 1,2
    // 1 -> 3
    // 2 -> 3
    // 3 -> (none)

    row_ptr = {0, 2, 3, 4, 4};
    col_idx = {1,2, 3, 3};
}

// -----------------------------
// MAIN
// -----------------------------
int main()
{
    std::vector<int> h_row_ptr;
    std::vector<int> h_col_idx;

    build_toy_graph(h_row_ptr, h_col_idx);

    int num_nodes = 4;

    int *d_row_ptr, *d_col_idx;

    cudaMalloc(&d_row_ptr, h_row_ptr.size() * sizeof(int));
    cudaMalloc(&d_col_idx, h_col_idx.size() * sizeof(int));

    cudaMemcpy(d_row_ptr, h_row_ptr.data(),
               h_row_ptr.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_col_idx, h_col_idx.data(),
               h_col_idx.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    int source = 0;

    bfs_gpu(num_nodes, d_row_ptr, d_col_idx, source);

    std::cout << "BFS finished\n";

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);

    return 0;
}
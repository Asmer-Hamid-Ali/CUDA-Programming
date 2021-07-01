#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
using namespace std;

//Static Shared Memory Calculations (in our case 16x16 matrix)
#define SHMEM_SIZE 16 * 16 * 4

__global__ void MatMulTiled(int* a, int* b, int* c, int n, int tile_size)
{
    //Statistically assigned shared memory
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];



    //Calculating rows and columns for a particular thread
    int row = (blockIdx.x * blockDim.y) + threadIdx.x; //Calculating row
    int col = (blockIdx.x * blockDim.x) + threadIdx.x; //Calculating column

    int sum = 0;

    // Sweep tile across matrix
    for (int i = 0; i < (n / tile_size); i ++) //tile_size is same as block size. In our case block is 16x16. Therefore, block size/tile_size is 16
    {
        // Load in elements for this tile
        /*
        For A matrix: 
                    row*n = Select each row for each iteration eg:row0, row1, row2 ...
                    i*tile_size = Select subset of column for each iteration 
                    threadIdx.x = Select each column for each iteration eg: row0column0, row0column1, ... 

        For B matrix:
                    i*tile_size*n = Select each row
                    threadIdx.y = Select row within the set
                    col = Select which column
                    
        */
        A[threadIdx.y * blockDim.x + threadIdx.x] = a[row * n + (i * tile_size + threadIdx.x)];
        B[threadIdx.y * blockDim.x + threadIdx.x] = b[(i * tile_size * n + threadIdx.y * n) + col];

        // Wait for both tiles to be loaded in before doing further computation
        __syncthreads();

        // Do matrix multiplication on the small matrix
        for (int j = 0; j < blockDim.x; j++) {
            sum += A[threadIdx.y * blockDim.x + j] * B[j * blockDim.x + threadIdx.x];
        }

        // Wait for all threads to finish using current tiles before loading in new
        // ones
        __syncthreads();
    }

    // Write back results
    c[row * n + col] = sum;
}

int main()
{
    //Matrix of size 1024 x 1024
    int n = 1 << 10;

    //Size (in bytes) of mathrix
    size_t bytes = n * n * sizeof(int);

    //CPU pointers
    int* a, * b, * c;

    //Allocate CPU memory
    a = (int*)malloc(bytes);
    b = (int*)malloc(bytes);
    c = (int*)malloc(bytes);

    //Device pointer
    int* da, * db, * dc;

    //Allocate device memory
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);

    //Generating random matrix
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            a[i * n + j] = rand() % 100;
            b[i * n + j] = rand() % 100;
        }
    //Copy data to device from host
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

    //Thread per blocks
    int block_size = 16; //Since we are using 2d array, therefore 16 * 16 = 256 threads per block

    //Blocks in each dimensions
    int grid_size = (int)ceil(n / block_size); // Dividing total elements by number of threads to get blocks for each element

    dim3 grid(grid_size, grid_size); //Dimension of grid
    dim3 threads(block_size, block_size); //Dimension of block

    //Launching kernel
    MatMulTiled << < grid, threads >> > (a, b, c, n, block_size);

    //Copy back to host
    cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost);

    cout << "Done successfully" << endl;

    //Free up the memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
using namespace std;


__global__ void multKernel(int *a, int *b, int *c, int n)
{
    //Calculating rows and columns for a particular thread
    int row = (blockIdx.x * blockDim.y) + threadIdx.x; //Calculating row
    int col = (blockIdx.x * blockDim.x) + threadIdx.x; //Calculating column

    int sum = 0;

    //Checking boundary condition
    if ((row < n) && (col < n)) {
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main()
{
    //Matrix of size 1024 x 1024
    //int n = 1 << 10;
    constexpr int n = 1 << 10; 

    //Size (in bytes) of mathrix
    //size_t bytes = n * n * sizeof(int); 
    constexpr size_t bytes = n * n * sizeof(int); 

    //CPU pointers
    int* a, * b, * c; 

    // Allocating memory for these host pointers
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    /*
    //Allocate CPU memory
    a = (int*)malloc(bytes);
    b = (int*)malloc(bytes);
    c = (int*)malloc(bytes);*/

    // device ID (of GPU) for prefetching
    int id = cudaGetDevice(&id);

    /*
    //Device pointer
    int* da, * db, * dc;
    
    //Allocate device memory
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);
    */

    //Generating random matrix
    for(int i =0; i < n; i++)
        for (int j = 0; j < n; j++) {
            a[i * n + j] = rand() % 100;
            b[i * n + j] = rand() % 100;
        }
    /*
    //Copy data to device from host
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);
    */

    //Thread per blocks
    int block_size = 16; //Since we are using 2d array, therefore 16 * 16 = 256 threads per block

    //Blocks in each dimensions
    int grid_size = (int)ceil(n / block_size); // Dividing total elements by number of threads to get blocks for each element

    dim3 grid(grid_size, grid_size); //Dimension of grid
    dim3 threads(block_size, block_size); //Dimension of block

    // prefetching 'a', 'b' and 'c'
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);
    cudaMemPrefetchAsync(c, bytes, id);

    //Launching kernel
    multKernel <<< grid, threads >>> (a, b, c, n);

    //Synchronization needed because to make sure kernel is done
    cudaDeviceSynchronize();

    /*
    //Copy back to host
    cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost);
    */

    //Prefetching back to CPU because we know that kernel is completed
    //Needed when we are comparing the matrix between GPU and CPU
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId); //cudaCpuDeviceId we don't need to calculate, the system automatically knows it

    cout << "Done successfully" << endl;
    
    //Free up the memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}

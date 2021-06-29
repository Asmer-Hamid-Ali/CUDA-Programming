#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>


__global__ void add(int *x, int *y, int *z)
{
    *z = *x + *y;
    printf("z is %d\n", *z);
}


int main()
{
    //Declaration
    int *a, *b, *c;
    int *deva, *devb, *devc;

    //Dynamic Memory Allocation in Host
    a = (int *)malloc(sizeof(int));
    b = (int *)malloc(sizeof(int));
    c = (int *)malloc(sizeof(int));

    //Reserving Memory in Device
    cudaMalloc((int **)&deva, sizeof(int));
    cudaMalloc((int **)&devb, sizeof(int));
    cudaMalloc((int **)&devc, sizeof(int));


    //Inputting values from user
    printf("Enter value of a and b\n");
    scanf("%d %d", a, b);

    /**c = *a + *b;
    printf("answer: %d\n", *c);*/

    //Coping values from HostToDevice
    cudaMemcpy(deva, a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devb, b, sizeof(int), cudaMemcpyHostToDevice);

    //Calling Kernel
    add<<<1,1>>>(deva, devb, devc);

    //Coping values from DeviceToHost
    cudaMemcpy(c, devc, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result is: %d\n", *c);

    //Free-up the memory
    cudaFree(deva), cudaFree(devb), cudaFree(devc);

    return 0;
}

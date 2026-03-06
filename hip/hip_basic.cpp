#include <hip/hip_runtime.h>
#include <iostream>

__global__ void hello() {
    printf("Hello from HIP on NVIDIA GPU!\n");
}

int main() {
    hello<<<1,1>>>();
    hipDeviceSynchronize();
    std::cout << "HIP test complete\n";
    return 0;
}
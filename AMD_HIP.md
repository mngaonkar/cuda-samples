# AMD Heterogeneous-Compute Interface for Portability (HIP)

## Installation (with Nvidia CUDA backend)
```bash
sudo pacman -S hip-runtime-nvidia hip-dev
```

## Set the HIP platform to NVIDIA (very important!)
Add this to your ~/.bashrc / ~/.zshrc:

```bash
export HIP_PLATFORM=nvidia
export HIP_COMPILER=nvcc
export HIP_RUNTIME=cuda   # optional but recommended
```

Appy
```bash
source ~/.bashrc
```

Verify setup:
```bash
hipconfig --full
```

```bash
hipcc --version

HIP version: 7.2.26043-9999
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_Dec_16_07:23:41_PM_PST_2025
Cuda compilation tools, release 13.1, V13.1.115
Build cuda_13.1.r13.1/compiler.37061995_0
```

## Example code
```cpp
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
```

Compile
```bash
hipcc -x cu -Wno-deprecated-gpu-targets hip_basic.cpp -o hip_basic
```

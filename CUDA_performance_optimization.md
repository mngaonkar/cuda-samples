# CUDA Performance Optimization
- To achieve optimal performance on NVIDIA GPUs, it's crucial to understand the underlying architecture and how to write efficient CUDA code.
- This guide covers key concepts and best practices for optimizing CUDA performance.

## CUDA Architecture Recap
- NVIDIA GPUs consist of multiple Streaming Multiprocessors (SMs), each containing CUDA cores, shared memory, registers, and warp schedulers.
- The GPU has a hierarchical memory structure: registers, shared memory, L1 cache, L2 cache, and global memory (VRAM).
- CUDA cores execute instructions in warps (groups of 32 threads) in a SIMT (Single Instruction Multiple Thread) fashion.
- Warp schedulers manage the execution of warps and help hide latency by switching between warps when one is waiting for memory or other resources.

## Performance Optimization Strategies
1. **Maximize Occupancy**: Ensure that enough warps are active to fully utilize the GPU's resources. This can be achieved by choosing appropriate block sizes and managing shared memory and register usage.
2. **Minimize Divergence**: Avoid branching within warps (e.g., if/else statements) that can cause threads to take different execution paths, leading to serialization and reduced performance.
3. **Optimize Memory Access**: Use shared memory to reduce global memory access latency, and ensure coalesced memory accesses when reading/writing global memory.
4. **Use Tensor Cores**: For matrix operations, leverage Tensor Cores (available in Volta and later architectures) to significantly speed up computations, especially for deep learning workloads.
5. **Profile and Benchmark**: Use NVIDIA profiling tools (e.g., Nsight Compute, Nsight Systems) to identify bottlenecks and optimize accordingly.

### Warp Divergence Example
```cpp
__global__ void exampleKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx % 2 == 0) {
        // Even threads execute this branch
        data[idx] += 1;
    } else {
        // Odd threads execute this branch
        data[idx] -= 1;
    }
}
```
- In this example, even and odd threads take different execution paths, causing warp divergence and reducing performance. To optimize, you could restructure the code to minimize divergence or use separate kernels for even and odd threads.

### Memory Access Optimization
```cpp
__global__ void optimizedKernel(float *input, float *output) {
    __shared__ float tile[32][32]; // Shared memory tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int idx = blockIdx.x * blockDim.x + tx;
    int idy = blockIdx.y * blockDim.y + ty;
    // Load data into shared memory
    tile[ty][tx] = input[idy * width + idx];
    __syncthreads();
    // Perform computation using shared memory
    output[idy * width + idx] = tile[ty][tx] * 2;
};
```
- In this example, we load data into shared memory to reduce global memory access latency, which can significantly improve performance for certain workloads.

### Occupancy

Occupancy = ratio of active warps on an SM to the maximum possible warps.

```
Occupancy = Active warps per SM / Max warps per SM
```

Higher occupancy → more warps available to hide memory latency.

**What limits occupancy:**

| Limiter | Explanation |
|---|---|
| Register usage | More registers per thread → fewer threads fit per SM |
| Shared memory | More shared mem per block → fewer blocks fit per SM |
| Block size | Too small → not enough warps to saturate SM |

### Memory Coalescing
When threads in a warp access global memory, the GPU tries to combine (coalesce) those accesses into as few transactions as possible.

```
COALESCED (fast):
Thread 0 → addr[0]
Thread 1 → addr[1]   →  single 128-byte transaction
Thread 2 → addr[2]
...
Thread 31 → addr[31]

UNCOALESCED (slow — strided):
Thread 0 → addr[0]
Thread 1 → addr[32]  →  32 separate transactions
Thread 2 → addr[64]
```
Rule: consecutive threads should access consecutive memory addresses.

### Latency Hiding
GPUs hide memory latency by switching to another warp while the current warp waits for data.
This is zero-cost context switching — unlike CPUs which stall.

```
Cycle timeline:
Warp 0:  [issue load] ----waiting (800 cycles)---- [execute]
Warp 1:           [execute] [issue load] ---------- [execute]
Warp 2:                     [execute] [issue load]  [execute]
...
SM never idles — always has another warp ready to run
```

Key insight: This is why high occupancy matters — you need enough warps in flight to cover memory latency windows.

Key point: Parallelism is at two levels:

- Intra-warp: 32 threads within a warp execute in parallel (SIMD)
- Inter-warp: Multiple warps on the same SM can execute concurrently
This warp-level parallelism is critical for hiding memory latency—while one warp waits for data, another warp computes.

## Host ↔ Device Transfer Bottleneck (PCIe)
```
CPU RAM  ←─── PCIe ───→  GPU VRAM
          ~32 GB/s          HBM: ~3.35 TB/s (H100)
```

PCIe bandwidth (~32 GB/s) is ~100x slower than GPU memory bandwidth.
This is a major bottleneck if not managed carefully.
Strategies to minimize PCIe transfers:

- Keep data on GPU as long as possible between operations
- Use non_blocking=True with pinned memory for async transfers
- Prefetch next batch while computing current batch (streams)
- Avoid unnecessary .cpu() calls in training loops

## NVLink & Multi-GPU Communication
PCIe is too slow for multi-GPU gradient exchange.
NVLink is NVIDIA's high-speed GPU-to-GPU interconnect.

```
NVLink Generations:
NVLink 3.0 (A100):  600 GB/s bidirectional
NVLink 4.0 (H100):  900 GB/s bidirectional
NVLink 5.0 (B100):  1,800 GB/s bidirectional

vs PCIe Gen5:       128 GB/s bidirectional
```

NVSwitch: fabric chip connecting all GPUs in a node (DGX systems).
Every GPU can communicate with every other GPU at full NVLink speed simultaneously.

## NCCL — Multi-GPU Collective Communications
NVIDIA Collective Communications Library.
Implements AllReduce, Broadcast, Gather, Scatter optimized for NVLink + PCIe + InfiniBand.

```
AllReduce (DDP gradient sync):
GPU 0: grad_A  ─┐
GPU 1: grad_B   ├─ AllReduce ─→  all GPUs get (grad_A + grad_B + grad_C + grad_D) / 4
GPU 2: grad_C   │
GPU 3: grad_D  ─┘
```

```python
# PyTorch DDP uses NCCL under the hood
import torch.distributed as dist

dist.init_process_group(backend='nccl')   # NCCL backend

model = torch.nn.parallel.DistributedDataParallel(model)
# Gradient AllReduce happens automatically after loss.backward()
```

## Precision Formats
Modern GPUs support multiple precision formats — choice affects speed, memory, and accuracy.

| Format | Bits | Range / Precision                  | Use Case                                      | Notes / Remarks                                      |
|--------|------|------------------------------------|-----------------------------------------------|------------------------------------------------------|
| **FP64** | 64   | Full double precision              | Scientific computing, high-accuracy simulations | IEEE 754 double (1-11-52)                           |
| **FP32** | 32   | Full single precision              | Traditional DL training baseline              | IEEE 754 single (1-8-23), widely supported           |
| **TF32** | 19   | FP32 range, reduced mantissa       | Default for matrix multiplies on A100+        | Transparent to user, Tensor Cores accelerated        |
| **BF16** | 16   | FP32 dynamic range, less mantissa  | LLM training & fine-tuning (very common)      | Google Brain / bfloat16 format (1-8-7)               |
| **FP16** | 16   | Narrower range than FP32           | Mixed-precision training & inference          | Needs loss scaling in training, Tensor Core support  |
| **INT8** | 8    | Integer only                       | Quantized inference (very fast)               | Used in TensorRT, ONNX Runtime, post-training quant  |
| **FP8**  | 8    | Float, very low precision          | H100+ training & inference (Transformer Engine) | E4M3 / E5M2 variants, NVIDIA-specific acceleration   |
| **FP4**  | 4    | Ultra-low precision float          | B100+ (Blackwell architecture) training       | Emerging / experimental, extreme quantization        |

Why BF16 over FP16 for LLMs:
BF16 has the same exponent range as FP32 — no overflow/underflow issues.
FP16's narrow range causes gradient underflow without careful loss scaling.


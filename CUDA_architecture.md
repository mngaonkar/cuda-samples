## CUDA GPU
```
GPU
 ├── Multiple SMs (Streaming Multiprocessors)
 │     ├── CUDA Cores
 │     ├── Shared Memory
 │     ├── Registers
 │     ├── Warp Schedulers
 │
 ├── Global Memory (VRAM)
 ├── L2 Cache
 └── Memory Controllers
```

```
┌─────────────────────────────────────────────────────────────────┐
│                          GPU DEVICE                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Streaming Multiprocessor (SM)              │    │
│  │                  (multiple per GPU)                     │    │
│  │                                                         │    │
│  │  ┌──────────────────────────────────────────────────┐   │    │
│  │  │                  Warp (32 threads)               │   │    │
│  │  │  Thread Thread Thread ... Thread  (x32)          │   │    │
│  │  └──────────────────────────────────────────────────┘   │    │
│  │                                                         │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │    │
│  │  │  CUDA    │  │  Tensor  │  │  Shared Memory /     │  │    │
│  │  │  Cores   │  │  Cores   │  │  L1 Cache (per SM)   │  │    │
│  │  └──────────┘  └──────────┘  └──────────────────────┘  │    │
│  │                                                         │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │    │
│  │  │  Warp    │  │  Register│  │  RT Cores (Ampere+)  │  │    │
│  │  │Schedulers│  │   File   │  │                      │  │    │
│  │  └──────────┘  └──────────┘  └──────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Global Memory (VRAM)                    │   │
│  │           (HBM2/HBM3 — e.g. 80GB on A100)               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

```
HARDWARE          ←→    SOFTWARE
─────────────────────────────────
GPU Device        ←→    Grid
SM                ←→    Block
CUDA Core/Warp    ←→    Thread
```

## Streaming Multiprocessor (SM)
- Fundamental computing unit of NVIDIA GPUs.
- Contains CUDA cores, shared memory, registers, warp schedulers.
- Executes warps (groups of 32 threads) in SIMT fashion.
- Each SM can execute multiple warps concurrently, hiding latency.
- Warp schedulers manage instruction execution and resource allocation.

```python
torch.cuda.get_device_properties(0).multi_processor_count
```

## CUDA Cores
CUDA cores execute arithmetic operations (FP32 or INT32).

## Block and Thread
CUDA organizes work like this:
```
Grid
 ├── Blocks
 │     ├── Threads
```
Example:
```
kernel<<<num_blocks, threads_per_block>>>(...)
```

### Block
Group of threads that:
- Synchronize together.
- Share memory.
- Execute on the same SM.

### Thread
Smallest execution unit.

### Warp
```
Warp = 32 threads
```
- Executes instructions in lockstep.
- The atomic scheduling unit — you don't schedule threads, you schedule warps
- Critical: if threads in a warp diverge (if/else), both branches execute serially → warp divergences
- Best to minimize divergence for optimal performance.
- This is SIMT (Single Instruction Multiple Thread).

```python
torch.cuda.get_device_properties(0).warp_size
```

## CUDA Streams
A stream is a sequence of operations that execute in order on the GPU.
Operations in different streams can execute concurrently.
- Useful for overlapping computation and data transfer.

```python
import torch

# PyTorch streams
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# Overlap compute with data transfer
with torch.cuda.stream(stream1):
    output1 = model(batch1)

with torch.cuda.stream(stream2):
    # Prefetch next batch while computing on batch1
    batch2 = batch2.to('cuda', non_blocking=True)

# Synchronize — wait for both streams
torch.cuda.synchronize()
```

## CUDA Events


## Memory Hierarchy

- Registers (fastest, per thread)
- Shared Memory (fast, per block)
- L1 Cache
- L2 Cache
- Global Memory (VRAM, slower)

### Registers
- Fastest, private to thread.

### Shared Memory
- Fast, shared among threads in a block.
- Used for - Matrix multiplication tiling, Attention kernels

### Global Memory
VRAM.
- Large but slower.
- Used for:
  - Model weights
  - Activations
  - KV cache

## Pinned Memory (Page-Locked Memory)
By default, host memory is pageable — the OS can move it.
Pinned memory is locked in RAM, enabling faster Host↔Device transfers via DMA.
```
- Pageable host memory:  CPU → staging buffer → GPU    (slow, 2 copies)
- Pinned host memory:    CPU → GPU directly via DMA    (fast, 1 copy)
```
## Unified Memory
Single memory address space accessible by both CPU and GPU.
GPU migrates pages on demand — no explicit cudaMemcpy needed.

```
float* data;
cudaMallocManaged(&data, N * sizeof(float));   // unified allocation

// CPU can write
for (int i = 0; i < N; i++) data[i] = i;

// GPU can read — hardware handles migration
kernel<<<grid, block>>>(data, N);

cudaDeviceSynchronize();
printf("%f\n", data[0]);   // CPU reads back — no explicit copy
```

### GDDR vs HBM
- GDDR (Graphics DDR): Used in consumer GPUs, optimized for graphics workloads.
- HBM (High Bandwidth Memory): Used in data center GPUs, optimized for high bandwidth

| Memory Type       | Speed       | Size (approximate)          | Scope                  | Notes                                      |
|-------------------|-------------|-----------------------------|------------------------|--------------------------------------------|
| **Registers**     | Fastest     | ~256 KB / SM                | Per thread             | Private to each individual thread          |
| **Shared Memory** | Very fast   | 16–100 KB / SM              | Per block              | Programmer-controlled, L1-like speed       |
| **L1 Cache**      | Very fast   | ~100 KB / SM                | Per SM (automatic)     | Transparent caching for global/local data  |
| **L2 Cache**      | Fast        | ~40 MB (varies by GPU)      | Whole GPU              | Shared across all SMs                      |
| **Global Memory** | Slow        | 8–80 GB                     | Whole GPU (VRAM)       | HBM2 / HBM3 on modern high-end GPUs        |
| **Constant Memory**| Fast*      | 64 KB                       | Read-only, broadcast   | Excellent for constants broadcast to warps |
| **Texture Memory**| Fast*       | Varies (backed by global)   | Read-only, spatial     | Optimized for 2D/3D data with filtering    |

### Tensor Cores
Executes matrix multiply-accumulate (MMA) in one instruction: D = A×B + C

They accelerate:
- Matrix multiplications
- FP16 / BF16
- Mixed precision
- INT8
- LLM training relies heavily on Tensor Cores.

## CUDA Software Stack
```
PyTorch / TensorFlow
      ↓
CUDA Runtime API
      ↓
CUDA Driver
      ↓
GPU Hardware
```

## CUDA Prefixes and Suffixes
- CUDA code often uses specific prefixes and suffixes to indicate the type of function or variable:
- `__global__`: Indicates a kernel function that runs on the device and is called from the host.
- `__device__`: Indicates a function that runs on the device and can only be called from other device functions or kernels.
- `__host__`: Indicates a function that runs on the host (CPU) and can only be called from the host.
- `__shared__`: Indicates a variable that resides in shared memory, accessible by all threads in a block.
- `__constant__`: Indicates a variable that resides in constant memory, which is read-only and optimized for broadcast to all threads.
- `__restrict__`: Indicates that a pointer is not aliased, allowing for potential optimizations by the compiler.

## NVidia GPUs
- Turing (2018): RTX 20 series, Tensor Cores, RT Cores.
- Ampere (2020): RTX 30 series, improved Tensor Cores, better performance.
- Hopper (2022): H100, optimized for LLM training, new Tensor Cores, NVLink 4.0, PCIe Gen5.
- Ada Lovelace (2022): RTX 40 series, improved performance, better power efficiency.
- Blackwell (2024): H200, further optimized for LLM training, new architecture features, improved performance and efficiency.

## MIG — Multi-Instance GPU

Ampere (A100) and later — partition a single GPU into isolated instances.
Each MIG instance has its own SM slice, memory, and bandwidth — hardware isolation.

```
# Enable MIG mode
nvidia-smi -i 0 -mig 1

# Create 7 equal instances (1g.10gb profile)
nvidia-smi mig -cgi 1g.10gb,1g.10gb,1g.10gb,1g.10gb,1g.10gb,1g.10gb,1g.10gb -C

# List instances
nvidia-smi mig -lgi
```
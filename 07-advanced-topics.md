# Chapter 7: Advanced Topics

## 7.1 Parallel Processing

### 7.1.1 Flynn's Taxonomy

Classification of parallel computer architectures.

```
┌─────────────────────────────────────────────────┐
│         Flynn's Taxonomy (1966)                 │
│                                                 │
│              Instruction Stream                 │
│           Single    │    Multiple               │
│        ─────────────┼─────────────              │
│         │           │             │             │
│  Single │   SISD    │    MISD     │             │
│    │    │           │             │             │
│  Data   ├───────────┼─────────────┤             │
│    │    │           │             │             │
│  Stream │   SIMD    │    MIMD     │             │
│         │           │             │             │
│ Multiple│           │             │             │
└─────────────────────────────────────────────────┘

SISD (Single Instruction, Single Data):
┌──────────────────────────┐
│   Traditional CPU        │
│                          │
│   Control Unit           │
│        │                 │
│        ▼                 │
│   Processing Unit        │
│        │                 │
│        ▼                 │
│   Data Stream            │
│                          │
│  Example: Classic CPUs   │
└──────────────────────────┘

SIMD (Single Instruction, Multiple Data):
┌──────────────────────────────────────┐
│     Control Unit (Single)            │
│            │                         │
│     ┌──────┼──────┬──────┬──────┐    │
│     ▼      ▼      ▼      ▼      ▼    │
│   [PU1] [PU2] [PU3] [PU4] [PU5]      │
│     │      │      │      │      │    │
│     ▼      ▼      ▼      ▼      ▼    │
│   [D1]  [D2]  [D3]  [D4]  [D5]       │
│                                      │
│  Same operation on different data    │
│  Examples: GPUs, Vector processors   │
│            SSE, AVX                  │
└──────────────────────────────────────┘

Example: Vector Addition
  Scalar (SISD):
    for (i=0; i<4; i++)
        C[i] = A[i] + B[i];
    
  SIMD:
    C[0:3] = A[0:3] + B[0:3];  // One instruction!

MISD (Multiple Instruction, Single Data):
┌──────────────────────────────────────┐
│   [CU1] [CU2] [CU3]                  │
│     │     │     │                    │
│     ▼     ▼     ▼                    │
│   [PU1] [PU2] [PU3]                  │
│     │     │     │                    │
│     └─────┼─────┘                    │
│           ▼                          │
│      Data Stream                     │
│                                      │
│  Rare in practice                    │
│  Example: Fault-tolerant systems     │
└──────────────────────────────────────┘

MIMD (Multiple Instruction, Multiple Data):
┌──────────────────────────────────────┐
│  [CU1] [CU2] [CU3] [CU4]             │
│    │     │     │     │               │
│    ▼     ▼     ▼     ▼               │
│  [PU1] [PU2] [PU3] [PU4]             │
│    │     │     │     │               │
│    ▼     ▼     ▼     ▼               │
│  [D1]  [D2]  [D3]  [D4]              │
│                                      │
│  Different operations, different data│
│  Examples: Multicore CPUs            │
│            Distributed systems       │
└──────────────────────────────────────┘
```

### 7.1.2 Parallel Programming Models

```
Shared Memory:
┌────────────────────────────────────┐
│  Thread 1  Thread 2  Thread 3      │
│     │         │         │          │
│     └─────────┼─────────┘          │
│               ▼                    │
│        Shared Memory               │
│     ┌────┬────┬────┬────┐          │
│     │ A  │ B  │ C  │ D  │          │
│     └────┴────┴────┴────┘          │
│                                    │
│  Synchronization needed!           │
│  - Locks, Semaphores, Barriers     │
└────────────────────────────────────┘

Example: OpenMP
  #pragma omp parallel for
  for (i = 0; i < N; i++) {
      C[i] = A[i] + B[i];
  }

Message Passing:
┌──────────┐      ┌──────────┐
│Process 1 │      │Process 2 │
│  ┌────┐  │      │  ┌────┐  │
│  │Mem │  │      │  │Mem │  │
│  └────┘  │      │  └────┘  │
└─────┬────┘      └────┬─────┘
      │  Send/Recv    │
      └───────────────┘
       (Network)

Example: MPI
  if (rank == 0) {
      MPI_Send(data, dest=1);
  } else {
      MPI_Recv(data, source=0);
  }

Data Parallel:
  Same operation on partitioned data
  
  Array[1000] partitioned:
  Core 0: Process [0:249]
  Core 1: Process [250:499]
  Core 2: Process [500:749]
  Core 3: Process [750:999]

Task Parallel:
  Different operations in parallel
  
  Task 1: Render frame
  Task 2: Process audio
  Task 3: Handle input
  Task 4: Update physics
  All run simultaneously
```

### 7.1.3 Amdahl's Law

Limits of parallel speedup.

```
┌─────────────────────────────────────────┐
│         Amdahl's Law                    │
│                                         │
│  Speedup = 1 / (S + P/N)                │
│                                         │
│  Where:                                 │
│    S = Serial fraction (cannot parallel)│
│    P = Parallel fraction                │
│    N = Number of processors             │
│    S + P = 1                            │
└─────────────────────────────────────────┘

Example:
  Program: 90% parallelizable (P=0.9, S=0.1)
  
  1 CPU:   Speedup = 1.0
  2 CPUs:  Speedup = 1 / (0.1 + 0.9/2)  = 1.82
  4 CPUs:  Speedup = 1 / (0.1 + 0.9/4)  = 3.08
  8 CPUs:  Speedup = 1 / (0.1 + 0.9/8)  = 4.71
  ∞ CPUs:  Speedup = 1 / 0.1            = 10.0

Maximum speedup limited by serial fraction!

Visualization:
Serial Part (10%)    │█
Parallel Part (90%)  │█████████

With 4 CPUs:
Serial Part (10%)    │█░░░░
Parallel Part (90%)  │██▌░░  (each CPU gets 22.5%)

Time:
Original: ████████████████████ (100%)
4 CPUs:   ███▌ (32.5%)
Speedup = 100/32.5 = 3.08

Impact of Serial Fraction:
┌──────┬────────────────────────────┐
│  S   │ Max Speedup (∞ processors) │
├──────┼────────────────────────────┤
│  5%  │       20×                  │
│ 10%  │       10×                  │
│ 25%  │        4×                  │
│ 50%  │        2×                  │
└──────┴────────────────────────────┘

Key Insight:
  Even 5% serial code limits speedup to 20×
  No matter how many processors!
```

## 7.2 Multicore Processors

### 7.2.1 Multicore Architecture

```
Dual-Core Processor:
┌─────────────────────────────────────┐
│                                     │
│  ┌──────────┐      ┌──────────┐     │
│  │  Core 0  │      │  Core 1  │     │
│  │          │      │          │     │
│  │  ┌────┐  │      │  ┌────┐  │     │
│  │  │L1I │  │      │  │L1I │  │     │
│  │  └────┘  │      │  └────┘  │     │
│  │  ┌────┐  │      │  ┌────┐  │     │
│  │  │L1D │  │      │  │L1D │  │     │
│  │  └────┘  │      │  └────┘  │     │
│  │  ┌────┐  │      │  ┌────┐  │     │
│  │  │ L2 │  │      │  │ L2 │  │     │
│  │  └────┘  │      │  └────┘  │     │
│  └─────┬────┘      └─────┬────┘     │
│        │                 │          │
│        └────────┬────────┘          │
│                 ▼                   │
│         ┌───────────────┐           │
│         │   Shared L3   │           │
│         │     Cache     │           │
│         └───────┬───────┘           │
│                 ▼                   │
│         ┌───────────────┐           │
│         │Memory Control │           │
│         └───────┬───────┘           │
└─────────────────┼───────────────────┘
                  ▼
             Main Memory

Intel Core i7 (8-core example):
┌────────────────────────────────────┐
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐        │
│ │C0  │ │C1  │ │C2  │ │C3  │        │
│ │L1I │ │L1I │ │L1I │ │L1I │        │
│ │L1D │ │L1D │ │L1D │ │L1D │        │
│ │L2  │ │L2  │ │L2  │ │L2  │        │
│ └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘        │
│    └──────┴──────┴──────┘          │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐        │
│ │C4  │ │C5  │ │C6  │ │C7  │        │
│ │L1I │ │L1I │ │L1I │ │L1I │        │
│ │L1D │ │L1D │ │L1D │ │L1D │        │
│ │L2  │ │L2  │ │L2  │ │L2  │        │
│ └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘        │
│    └──────┴──────┴──────┘          │
│              ▼                     │
│        Shared L3 Cache             │
│         (16-20 MB)                 │
└────────────────────────────────────┘

Typical Sizes:
  L1I: 32 KB per core
  L1D: 32 KB per core
  L2:  256 KB per core
  L3:  2-20 MB shared
```

### 7.2.2 Cache Coherence

Problem: Multiple caches can have different values for same address.

```
Initial State:
┌──────┐     ┌──────┐     ┌────────┐
│Core 0│     │Core 1│     │ Memory │
│ L1   │     │ L1   │     │  X=0   │
└──────┘     └──────┘     └────────┘

Core 0 reads X:
┌──────┐     ┌──────┐     ┌────────┐
│Core 0│     │Core 1│     │ Memory │
│X=0   │     │      │     │  X=0   │
└──────┘     └──────┘     └────────┘

Core 1 reads X:
┌──────┐     ┌──────┐     ┌────────┐
│Core 0│     │Core 1│     │ Memory │
│X=0   │     │ X=0  │     │  X=0   │
└──────┘     └──────┘     └────────┘

Core 0 writes X=1:
┌──────┐     ┌──────┐     ┌────────┐
│Core 0│     │Core 1│     │ Memory │
│X=1   │     │ X=0  │ ←── INCONSISTENT!
└──────┘     └──────┘     └────────┘

Solution: Cache Coherence Protocols
```

**MESI Protocol** (Most Common)

```
States:
  M - Modified:   Dirty, only in this cache
  E - Exclusive:  Clean, only in this cache
  S - Shared:     Clean, may be in other caches
  I - Invalid:    Not valid

State Transitions:
        ┌──────────┐
        │ Invalid  │
        └────┬─────┘
             │ Read miss
        ┌────▼─────┐
        │Exclusive │◄─────┐
        └────┬─────┘      │
             │ Write      │ Others read
             │            │
        ┌────▼─────┐      │
        │ Modified │      │
        └────┬─────┘      │
             │ Evict/     │
             │ Other read │
        ┌────▼─────┐      │
        │  Shared  │──────┘
        └──────────┘

Example:
Initial: All cores have I (Invalid)

Core 0 reads X:
  Core 0: I → E (Exclusive)
  Core 1: I

Core 1 reads X:
  Core 0: E → S (Shared)
  Core 1: I → S (Shared)

Core 0 writes X:
  Core 0: S → M (Modified)
  Core 1: S → I (Invalid)
  (Core 1's copy invalidated!)

Core 1 reads X:
  Core 0: M → S (write back to memory)
  Core 1: I → S (read from memory)

Bus Snooping:
┌──────────────────────────────────┐
│           Shared Bus             │
│   (All caches monitor)           │
├────────┬──────────┬──────────────┤
│        │          │              │
▼        ▼          ▼              ▼
Core 0   Core 1   Core 2    Memory
Cache    Cache    Cache
[X=S]    [X=S]    [X=I]

Core 0 writes X → Broadcast on bus
All caches see it and invalidate

Directory-Based (Scalable):
┌─────────────────────────────────┐
│     Directory at Memory         │
│  ┌────┬─────────────────┐       │
│  │Addr│ Present in cores│       │
│  ├────┼─────────────────┤       │
│  │ X  │  0, 1, 2        │       │
│  │ Y  │  1, 3           │       │
│  └────┴─────────────────┘       │
└─────────────────────────────────┘

On write to X:
  1. Check directory
  2. Send invalidate to cores 0,1,2
  3. Wait for acknowledgments
  4. Allow write
```

### 7.2.3 Memory Consistency Models

```
Sequential Consistency:
  All operations appear to execute in some
  sequential order, same for all processors

Example:
  Initially: A=0, B=0
  
  Core 0:          Core 1:
  A = 1            B = 1
  print B          print A
  
  Possible outputs: (0,0), (0,1), (1,0), (1,1)
  NOT possible: Both print 0 simultaneously
  (Sequential consistency guaranteed)

Relaxed Consistency:
  Allow reordering for performance
  Require explicit synchronization
  
  Core 0:          Core 1:
  A = 1            B = 1
  FENCE            FENCE
  print B          print A
  
  FENCE ensures previous writes visible

Memory Barrier Types:
┌──────────────────────────────────┐
│ Load-Load:   Load; LL; Load      │
│ Load-Store:  Load; LS; Store     │
│ Store-Store: Store; SS; Store    │
│ Store-Load:  Store; SL; Load     │
│ Full:        All combinations    │
└──────────────────────────────────┘
```

## 7.3 GPU Architecture

Graphics Processing Units: Massively parallel processors.

```
CPU vs GPU Philosophy:

CPU: Few powerful cores
┌─────┬─────┬─────┬─────┐
│     │     │     │     │
│Core │Core │Core │Core │
│  0  │  1  │  2  │  3  │
│     │     │     │     │
│Complex OOO execution  │
│Branch prediction      │
│Large caches           │
└─────┴─────┴─────┴─────┘

GPU: Many simple cores
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐
│││││││││││││││││││││││
│ Hundreds/Thousands  │
│ of simple cores     │
│                     │
│ SIMD execution      │
│ No branch predict   │
│ Small caches        │
└┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┘

NVIDIA GPU Architecture (Simplified):
┌─────────────────────────────────────────┐
│              GPU Chip                   │
│                                         │
│  ┌────────────┐     ┌────────────┐      │
│  │    SM 0    │     │    SM 1    │      │
│  │(Streaming  │ ... │(Streaming  │      │
│  │Multiproc.) │     │Multiproc.) │      │
│  └─────┬──────┘     └─────┬──────┘      │
│        │                  │             │
│  ┌─────┴──────────────────┴──────┐      │
│  │        Global Memory           │     │
│  │        (VRAM: 8-24 GB)         │     │
│  └────────────────────────────────┘     │
└─────────────────────────────────────────┘

Streaming Multiprocessor (SM):
┌───────────────────────────────────┐
│           SM                      │
│                                   │
│  ┌──────────────────────┐         │
│  │  Instruction Cache   │         │
│  └──────────┬───────────┘         │
│             │                     │
│    ┌────────▼────────┐            │
│    │  Warp Scheduler │            │
│    └────────┬────────┘            │
│             │                     │
│    ┌────────▼────────────┐        │
│    │   CUDA Cores (32)   │        │
│    │  ┌─┬─┬─┬─┬─┬─┬─┬─┐  │        │
│    │  │││││││││││││││││  │        │
│    │  └─┴─┴─┴─┴─┴─┴─┴─┘  │        │
│    └─────────────────────┘        │
│                                   │
│    ┌──────────────────┐           │
│    │  Shared Memory   │           │
│    │  (48-164 KB)     │           │
│    └──────────────────┘           │
│                                   │
│    ┌──────────────────┐           │
│    │  Register File   │           │
│    │  (64K registers) │           │
│    └──────────────────┘           │
└───────────────────────────────────┘

Execution Model (SIMT):
Single Instruction, Multiple Threads

Warp: Group of 32 threads
  All execute same instruction
  Different data
  
  Warp 0: T0  T1  T2  ... T31
          ↓   ↓   ↓       ↓
          [Same instruction]
          ↓   ↓   ↓       ↓
          D0  D1  D2  ... D31

Thread Hierarchy:
┌────────────────────────────────┐
│           Grid                 │
│  ┌──────┐  ┌──────┐  ┌──────┐  │
│  │Block │  │Block │  │Block │  │
│  │  0   │  │  1   │  │  2   │  │
│  │      │  │      │  │      │  │
│  │┌─┬─┐ │  │      │  │      │  │
│  ││T│T│ │  │      │  │      │  │
│  ││0│1│ │  │      │  │      │  │
│  │├─┼─┤ │  │      │  │      │  │
│  ││T│T│ │  │      │  │      │  │
│  ││2│3│ │  │      │  │      │  │
│  │└─┴─┘ │  │      │  │      │  │
│  └──────┘  └──────┘  └──────┘  │
└────────────────────────────────┘

CUDA Code Example:
__global__ void vectorAdd(float* A, float* B, 
                          float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Launch with 256 threads per block
vectorAdd<<<(N+255)/256, 256>>>(A, B, C, N);

Memory Hierarchy:
┌──────────────┬──────────┬─────────┬──────┐
│   Memory     │   Size   │  Speed  │Scope │
├──────────────┼──────────┼─────────┼──────┤
│ Registers    │ 64K×32b  │ Fastest │Thread│
│ Shared Mem   │ 48-164KB │  Fast   │Block │
│ L1 Cache     │ 16-128KB │  Fast   │  SM  │
│ L2 Cache     │ 0.5-6 MB │  Good   │ GPU  │
│ Global Mem   │ 8-24 GB  │  Slow   │ GPU  │
│ CPU Memory   │ 16-128GB │ Slowest │ CPU  │
└──────────────┴──────────┴─────────┴──────┘

Bandwidth (Typical):
  Register:      8-12 TB/s
  Shared:        ~2 TB/s
  Global:        200-900 GB/s
  CPU-GPU PCIe:  16-32 GB/s
```

### 7.3.1 GPU Performance Considerations

```
Occupancy:
  Active warps / Max warps per SM
  
  Higher occupancy → Better latency hiding
  
  Factors limiting occupancy:
  - Registers per thread
  - Shared memory per block
  - Threads per block
  - Blocks per SM

Memory Coalescing:
  Consecutive threads access consecutive addresses
  
  Good (Coalesced):
  T0: A[0]
  T1: A[1]
  T2: A[2]
  T3: A[3]
  → Single memory transaction
  
  Bad (Uncoalesced):
  T0: A[0]
  T1: A[100]
  T2: A[200]
  T3: A[300]
  → Multiple memory transactions

Branch Divergence:
  Threads in warp take different paths
  
  if (threadIdx.x < 16) {
      // Path A
  } else {
      // Path B
  }
  
  Execution:
  1. Execute Path A (threads 0-15 active)
  2. Execute Path B (threads 16-31 active)
  
  Both paths executed sequentially!
  Wasted cycles on inactive threads

Example: Matrix Multiplication
  C = A × B
  
  Sequential (CPU):
  for (i = 0; i < N; i++)
      for (j = 0; j < N; j++)
          for (k = 0; k < N; k++)
              C[i][j] += A[i][k] * B[k][j];
  
  Parallel (GPU):
  Each thread computes one C[i][j]
  N² threads in parallel!
  
  Speedup: 10-100× for large matrices
```

## 7.4 Modern Processor Features

### 7.4.1 Advanced Vector Extensions (AVX)

```
x86 SIMD Evolution:
MMX   (1997): 64-bit,  8 registers
SSE   (1999): 128-bit, 8 registers (16 in x64)
AVX   (2011): 256-bit, 16 registers
AVX-512(2016): 512-bit, 32 registers

AVX-512 Register:
┌───────────────────────────────────────────┐
│           ZMM0 (512 bits)                 │
├───────────────────────────────────────────┤
│  [0] [1] [2] [3] [4] [5] [6] [7]         │
│  [8] [9] [10] [11] [12] [13] [14] [15]   │
└───────────────────────────────────────────┘
   16 × 32-bit floats or 8 × 64-bit doubles

Example: Array Addition (8 floats)
Scalar:
  for (i = 0; i < 8; i++)
      C[i] = A[i] + B[i];
  = 8 instructions

AVX:
  __m256 a = _mm256_load_ps(&A[0]);
  __m256 b = _mm256_load_ps(&B[0]);
  __m256 c = _mm256_add_ps(a, b);
  _mm256_store_ps(&C[0], c);
  = 4 instructions (8 ops in parallel!)

AVX-512 with FMA (Fused Multiply-Add):
  c = a × b + c
  One instruction, 16 operations!
  
  Theoretical peak:
  4 GHz × 16 ops × 2 FMA units = 128 GFLOPS/core
```

### 7.4.2 Hardware Transactional Memory (HTM)

```
Traditional Locking:
  lock(mutex);
  balance += amount;
  unlock(mutex);
  
  Problems:
  - Coarse-grained: slow
  - Fine-grained: complex
  - Deadlock possible

Transactional Memory:
  XBEGIN
    balance += amount;  // Atomic region
  XEND
  
  Hardware tracks reads/writes
  Commits if no conflicts
  Aborts and retries if conflict

Intel TSX (Transactional Synchronization Extensions):
┌──────────────────────────────────────┐
│ Thread 1:          Thread 2:         │
│ XBEGIN             XBEGIN            │
│   read A             read A          │
│   write A            write A         │
│ XEND               XEND              │
│                                      │
│ Conflict detected!                   │
│ One transaction aborts, retries      │
└──────────────────────────────────────┘

Advantages:
  + Programmer convenience
  + Good for low contention
  + No deadlock

Disadvantages:
  - Limited capacity
  - Nested transactions tricky
  - Debugging harder
```

### 7.4.3 Security Features

```
Address Space Layout Randomization (ASLR):
  Randomize memory addresses
  Makes exploits harder
  
  Without ASLR:
  Stack:  0x7fff0000 (predictable)
  Heap:   0x00600000
  Code:   0x00400000
  
  With ASLR:
  Stack:  0x7f1a2000 (random)
  Heap:   0x1b8e3000
  Code:   0x56782000

NX Bit (No-Execute):
  Mark pages as non-executable
  Data pages can't contain code
  
  Page Table Entry:
  ┌────┬────┬────┬────┬────┬────┬───┐
  │Addr│ R  │ W  │ X  │ U  │... │NX │
  └────┴────┴────┴────┴────┴────┴───┘
                               ↑
                        Execute Disable

Intel SGX (Software Guard Extensions):
  Secure enclaves for sensitive code
  
  ┌────────────────────────────┐
  │    Untrusted Memory        │
  │  ┌──────────────────────┐  │
  │  │   Secure Enclave     │  │
  │  │  (Encrypted, HW      │  │
  │  │   protected)         │  │
  │  └──────────────────────┘  │
  │                            │
  │  OS and apps can't access  │
  └────────────────────────────┘

Spectre/Meltdown Mitigations:
  - Speculation barriers
  - IBRS/STIBP (Indirect Branch)
  - Page Table Isolation (PTI)
  - Microcode updates
  - Flush caches on context switch
```

### 7.4.4 Power Management

```
Dynamic Voltage and Frequency Scaling (DVFS):
  Adjust voltage and frequency based on load
  
  ┌────────────────────────────┐
  │ High Performance Mode:     │
  │   Voltage: 1.2V            │
  │   Frequency: 4.0 GHz       │
  │   Power: High              │
  └────────────────────────────┘
  
  ┌────────────────────────────┐
  │ Power Saving Mode:         │
  │   Voltage: 0.8V            │
  │   Frequency: 1.5 GHz       │
  │   Power: Low (35% of max)  │
  └────────────────────────────┘
  
  Power ∝ V² × f

Clock Gating:
  Disable clock to unused units
  
  ┌──────────────┐
  │    Clock     │
  └──────┬───────┘
         │
    ┌────▼────┐
    │  Gate   │ ← Enable signal
    └────┬────┘
         │
    ┌────▼────┐
    │  Unit   │
    └─────────┘
  
  If unit idle → gate clock
  Saves dynamic power

Power Gating:
  Completely power off units
  
  ┌──────────────┐
  │   Power      │
  └──────┬───────┘
         │
    ┌────▼────┐
    │ Switch  │ ← Control
    └────┬────┘
         │
    ┌────▼────┐
    │  Unit   │
    └─────────┘
  
  Saves static (leakage) power
  Longer wake-up time

C-States (CPU Sleep States):
  C0: Active
  C1: Halt (stop clock)
  C2: Stop clock, flush cache
  C3: Stop clock, flush TLB
  C6: Power gate core
  C7: Power gate core + LLC
  C8-C10: Deeper sleep states
  
  Deeper = More power saved, longer wake-up

Turbo Boost:
  Temporarily exceed base frequency
  When thermal/power budget allows
  
  Base:  3.0 GHz (all cores)
  Turbo: 4.5 GHz (1-2 cores)
  
  ┌─────────────────────────────┐
  │ 4 cores active: 3.2 GHz     │
  │ 2 cores active: 3.8 GHz     │
  │ 1 core active:  4.5 GHz     │
  └─────────────────────────────┘
```

## 7.5 Emerging Technologies

### 7.5.1 3D Stacking

```
Vertical integration of dies

Traditional (2D):
┌──────────────────────────────┐
│         CPU Die              │
│  ┌────┐  ┌────┐  ┌────┐      │
│  │Core│  │Core│  │Core│      │
│  └────┘  └────┘  └────┘      │
└──────────┬───────────────────┘
           │
    ┌──────▼───────────────────┐
    │     Memory (Separate)    │
    └──────────────────────────┘

3D Stacked:
┌──────────────────────────────┐
│      Memory Die (Layer 3)    │
├──────────────────────────────┤
│      Cache Die (Layer 2)     │
├──────────────────────────────┤
│      CPU Die (Layer 1)       │
│  ┌────┐  ┌────┐  ┌────┐      │
│  │Core│  │Core│  │Core│      │
│  └────┘  └────┘  └────┘      │
└──────────────────────────────┘
        │
Through-Silicon Vias (TSVs)

Benefits:
  + Much shorter interconnects
  + Higher bandwidth
  + Lower latency
  + Lower power
  + Smaller footprint

Challenges:
  - Thermal management
  - Manufacturing complexity
  - Yield
  - Cost

Example: AMD 3D V-Cache
  64 MB L3 cache stacked on CPU
  15% gaming performance boost
```

### 7.5.2 Quantum Computing

```
Quantum Bit (Qubit):
  Classical bit: 0 or 1
  Qubit: |0⟩, |1⟩, or superposition α|0⟩ + β|1⟩
  
  Superposition:
  ┌────────────────┐
  │   │0⟩ + │1⟩    │ Both states simultaneously!
  └────────────────┘

Entanglement:
  Two qubits linked
  Measuring one affects other
  
  │00⟩ + │11⟩
  (Both 0 or both 1, correlated)

Quantum Gates:
  Similar to classical logic gates
  but operate on superpositions
  
  Hadamard Gate (H):
  │0⟩ → (│0⟩ + │1⟩)/√2  (equal superposition)
  │1⟩ → (│0⟩ - │1⟩)/√2

Quantum Circuit Example:
  ──H───●───M  (Qubit 0)
        │
  ──────⊕───M  (Qubit 1)
  
  H = Hadamard (create superposition)
  ● = Control
  ⊕ = CNOT (conditional flip)
  M = Measurement

Quantum Advantage:
  Certain problems exponentially faster
  
  Examples:
  - Factoring (Shor's algorithm)
  - Database search (Grover's algorithm)
  - Simulation of quantum systems
  
  N-qubit system: 2^N states
  50 qubits: 2^50 ≈ 10^15 states
  (Classical computer struggles)

Current Limitations:
  - Decoherence (qubits fragile)
  - Error rates high
  - Scaling difficult
  - Requires extreme cooling
  - Limited applications (so far)

IBM Quantum Computer:
  ~100 qubits
  Superconducting circuits
  <15 mK temperature
```

### 7.5.3 Neuromorphic Computing

```
Brain-inspired architecture

Neuron Model:
     Inputs           Output
  ──┬──►┌────────┐
  ──┤   │ Neuron │──► Spike
  ──┤   │        │
  ──┴──►└────────┘
  (Weighted)  (Threshold)

Spiking Neural Network:
  Communication via spikes (events)
  Asynchronous, event-driven
  Low power
  
  ┌───┐    ┌───┐    ┌───┐
  │ N1│───►│ N2│───►│ N3│
  └───┘    └───┘    └───┘
    │        │        │
  Spike   Spike   Spike
    │        │        │
  Time─────────────────►

Intel Loihi:
  130,000 neurons
  130M synapses
  Event-driven
  On-chip learning
  
  Energy: 1000× better than GPU
  For certain tasks (pattern recognition)

Applications:
  - Pattern recognition
  - Sensor processing
  - Robotics
  - Always-on inference
```

### 7.5.4 Processing-in-Memory (PIM)

```
Compute where data resides

Traditional:
┌──────┐              ┌────────┐
│ CPU  │◄────────────►│ Memory │
└──────┘              └────────┘
   ↑                       ↑
  Compute            Data movement
                      (bottleneck!)

Processing-in-Memory:
┌─────────────────────────┐
│       Memory            │
│  ┌──────┐  ┌──────┐     │
│  │ Data │  │Logic │     │
│  └──────┘  └───┬──┘     │
│                │        │
│          Compute here!  │
└─────────────────────────┘

Benefits:
  + Eliminate data movement
  + Higher bandwidth
  + Lower energy
  + Better for data-intensive tasks

Example: Samsung HBM-PIM
  High Bandwidth Memory
  + Processing units
  
  Matrix operations in memory
  2× performance
  70% less energy

Challenges:
  - Limited compute capability
  - Programming model
  - Integration with CPU
```

---

**Key Takeaways:**
1. Parallel processing essential for performance scaling
2. Multicore and GPU provide different parallelism models
3. Cache coherence crucial for shared memory systems
4. Modern features: SIMD, security, power management
5. Emerging: 3D stacking, quantum, neuromorphic, PIM
6. Future: More specialization, heterogeneous computing
7. Amdahl's Law limits parallel speedup

**Conclusion:**
Computer architecture continues to evolve with:
- More cores and specialized units
- Better energy efficiency
- Novel computing paradigms
- Tighter integration of components

The field balances:
- Performance vs. Power
- Generality vs. Specialization
- Complexity vs. Programmability

For more detailed coverage of heterogeneous and multi-core systems, see [Chapter 8: Heterogeneous and Multi-Core Systems](./08-heterogeneous-computing.md).

---

## Further Reading

**Classic Textbooks:**
- "Computer Architecture: A Quantitative Approach" - Hennessy & Patterson
- "Computer Organization and Design" - Patterson & Hennessy
- "Modern Processor Design" - Shen & Lipasti

**Online Resources:**
- IEEE Micro Magazine
- ACM SIGARCH
- Hot Chips Conference
- ISCA (International Symposium on Computer Architecture)

**Hands-on:**
- CPU simulators (gem5, Sniper)
- FPGA programming
- Assembly language programming
- Performance profiling tools

**Keep Learning:**
Computer architecture is a dynamic field. Stay updated with:
- New processor announcements
- Research papers
- Open-source hardware (RISC-V)
- Industry blogs and forums


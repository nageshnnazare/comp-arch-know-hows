# Chapter 8: Heterogeneous and Multi-Core Systems

## 8.1 Heterogeneous Computing Overview

Modern systems combine different types of processors for optimal performance and efficiency.

```
┌───────────────────────────────────────────────────┐
│         Heterogeneous Computing System            │
│                                                   │
│  ┌──────────────────────────────────────────┐     │
│  │         Host CPU (General Purpose)       │     │
│  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐      │     │
│  │  │P-Core  │P-Core  │E-Core  │E-Core      │     │
│  │  └─────┘  └─────┘  └─────┘  └─────┘      │     │
│  └────────────────┬─────────────────────────┘     │
│                   │                               │
│         ┌─────────┴──────────┬──────────┐         │
│         │                    │          │         │
│         ▼                    ▼          ▼         │
│    ┌─────────┐         ┌─────────┐  ┌──────┐      │
│    │   GPU   │         │   DSP   │  │ NPU  │      │
│    │(Graphics│         │(Signal) │  │(AI)  │      │
│    │ & GPGPU)│         └─────────┘  └──────┘      │
│    └─────────┘                                    │
│                                                   │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│    │  Video  │  │ Crypto  │  │  FPGA   │          │
│    │Encoder  │  │ Engine  │  │(Reconfig)          │
│    └─────────┘  └─────────┘  └─────────┘          │
│                                                   │
│         Shared Memory / Interconnect              │
└───────────────────────────────────────────────────┘

Why Heterogeneous?

Homogeneous (All same cores):
  ┌────┬────┬────┬────┐
  │CPU │CPU │CPU │CPU │
  └────┴────┴────┴────┘
  + Simple programming
  + Uniform performance
  - Not energy efficient
  - Not optimal for all workloads

Heterogeneous (Different accelerators):
  ┌────┬────┬────┬────┐
  │CPU │GPU │NPU │DSP │
  └────┴────┴────┴────┘
  + Energy efficient (right tool for job)
  + Higher peak performance
  + Specialized optimizations
  - Complex programming
  - Data movement overhead
  - Synchronization challenges

Performance vs Power:
              Performance
                  ↑
     GPU ●        │
                  │
     CPU ●        │      ● FPGA
                  │
         ● NPU    │  ● ASIC
                  │
                  └───────────► Power
```

## 8.2 NUMA (Non-Uniform Memory Access)

Multi-socket systems with distributed memory.

```
Traditional UMA (Uniform Memory Access):
┌──────┬──────┬──────┬──────┐
│Core 0│Core 1│Core 2│Core 3│
└───┬──┴───┬──┴───┬──┴───┬──┘
    └──────┴──────┴──────┘
            │
      ┌─────▼─────┐
      │  Memory   │
      └───────────┘
All cores same distance to memory
Latency: Uniform (~100 ns)

NUMA (Non-Uniform Memory Access):
┌──────────────────────────────────────┐
│           Node 0                     │
│  ┌────┬────┬────┬────┐               │
│  │C0  │C1  │C2  │C3  │               │
│  └─┬──┴─┬──┴─┬──┴─┬──┘               │
│    └────┴────┴────┘                  │
│          │                           │
│    ┌─────▼─────┐                     │
│    │  Memory 0 │ ← Local memory      │
│    │  (64 GB)  │   (fast: ~80ns)     │
│    └─────┬─────┘                     │
└──────────┼───────────────────────────┘
           │ Interconnect (QPI/UPI)
           │
┌──────────┼───────────────────────────┐
│    ┌─────▼─────┐                     │
│    │  Memory 1 │ ← Remote memory     │
│    │  (64 GB)  │   (slow: ~140ns)    │
│    └─────┬─────┘                     │
│          │                           │
│    ┌────┬┴───┬────┬────┐             │
│    │C4  │C5  │C6  │C7  │             │
│    └────┴────┴────┴────┘             │
│           Node 1                     │
└──────────────────────────────────────┘

Memory Latency Comparison:
┌──────────────┬──────────┬────────┐
│   Access     │ Latency  │ B/W    │
├──────────────┼──────────┼────────┤
│ Local Memory │  80 ns   │ 100%   │
│ Remote Memory│ 140 ns   │  60%   │
│ Ratio        │  1.75×   │ 0.6×   │
└──────────────┴──────────┴────────┘

4-Socket NUMA System:
┌─────────┐      ┌─────────┐
│ Node 0  │◄────►│ Node 1  │
│ 16 cores│      │16 cores │
│ 128 GB  │      │128 GB   │
└────┬────┘      └────┬────┘
     │                │
     └────────┬───────┘
              │
     ┌────────┼───────┐
     │                │
┌────▼────┐      ┌────▼────┐
│ Node 2  │◄────►│ Node 3  │
│ 16 cores│      │16 cores │
│ 128 GB  │      │128 GB   │
└─────────┘      └─────────┘

Total: 64 cores, 512 GB
Mesh interconnect topology

NUMA Distance Matrix:
From│ To: Node0  Node1  Node2  Node3
────┼─────────────────────────────────
  0 │     10     21     21     32
  1 │     21     10     32     21
  2 │     21     32     10     21
  3 │     32     21     21     10

(Lower = faster)

NUMA-Aware Programming:

Bad: Remote memory access
┌──────────┐           ┌──────────┐
│  Node 0  │           │  Node 1  │
│          │           │          │
│ Process  ├──────────►│ Memory   │
│          │  Slow!    │          │
└──────────┘           └──────────┘

Good: Local memory access
┌──────────┐
│  Node 0  │
│          │
│ Process  │◄─┐
│          │  │ Fast!
│ Memory   ├──┘
└──────────┘

Linux NUMA API:
  numactl --cpunodebind=0 --membind=0 ./app
  
  # Pin to node 0 CPUs and memory
  
  numa_alloc_onnode(size, node);
  # Allocate memory on specific node

NUMA Effects on Performance:
┌──────────────────┬──────────┬──────────┐
│   Workload       │ Non-NUMA │   NUMA   │
│                  │  Aware   │  Aware   │
├──────────────────┼──────────┼──────────┤
│ Memory Intensive │  100%    │  140%    │
│ CPU Bound        │  100%    │  110%    │
│ Mixed            │  100%    │  125%    │
└──────────────────┴──────────┴──────────┘

NUMA optimization critical for:
- Databases
- Virtual machines
- High-performance computing
- Large memory footprint apps
```

## 8.3 Heterogeneous ISA Systems

### 8.3.1 ARM big.LITTLE Architecture

Different core types optimized for performance vs efficiency.

```
ARM big.LITTLE:

┌───────────────────────────────────────┐
│            SoC                        │
│                                       │
│  Big Cores (Cortex-A7x)               │
│  ┌──────┐  ┌──────┐  ┌──────┐         │
│  │ Core │  │ Core │  │ Core │         │
│  │  0   │  │  1   │  │  2   │         │
│  │      │  │      │  │      │         │
│  │ OOO  │  │ OOO  │  │ OOO  │         │
│  │ 3GHz │  │ 3GHz │  │ 3GHz │         │
│  └──────┘  └──────┘  └──────┘         │
│    High Performance, High Power       │
│                                       │
│  LITTLE Cores (Cortex-A5x)            │
│  ┌──────┐  ┌──────┐  ┌──────┐         │
│  │ Core │  │ Core │  │ Core │         │
│  │  3   │  │  4   │  │  5   │         │
│  │      │  │      │  │      │         │
│  │In-Ord│  │In-Ord│  │In-Ord│         │
│  │1.8GHz│  │1.8GHz│  │1.8GHz│         │
│  └──────┘  └──────┘  └──────┘         │
│    Lower Performance, Low Power       │
│                                       │
│         Shared L3 Cache               │
│         Coherency (CCI)               │
└───────────────────────────────────────┘

Comparison:
┌────────────┬─────────┬──────────┐
│ Feature    │   Big   │  LITTLE  │
├────────────┼─────────┼──────────┤
│ Perf/MHz   │  High   │   Low    │
│ Power      │  High   │   Low    │
│ Area       │  Large  │  Small   │
│ Pipeline   │  Deep   │ Shallow  │
│ Width      │   4     │    2     │
│ OOO Exec   │  Yes    │   No     │
└────────────┴─────────┴──────────┘

Migration Modes:

1. Cluster Migration:
   All tasks migrate together
   
   Light Load:        Heavy Load:
   [LITTLE] Active    [Big] Active
   [Big] Idle         [LITTLE] Idle

2. CPU Migration (HMP):
   Individual threads migrate
   
   Task 1 ──► [Big Core 0]
   Task 2 ──► [LITTLE Core 3]
   Task 3 ──► [Big Core 1]
   Task 4 ──► [LITTLE Core 4]
   
   Scheduler decides based on:
   - CPU utilization
   - Task priority
   - Power budget
   - Thermal state

3. Global Task Scheduling (GTS):
   All cores in single pool
   
   ┌────────────────────────────┐
   │  Available Cores:          │
   │  Big: 0, 1, 2              │
   │  LITTLE: 3, 4, 5           │
   │                            │
   │  Scheduler picks best core │
   │  for each task             │
   └────────────────────────────┘

Energy Efficiency:
             Performance
                  ↑
                  │    ● Big (3GHz)
                  │
                  │  ● Big (2GHz)
                  │
                  │● LITTLE (1.8GHz)
                  │
                  └────────────► Power

Energy-Performance Curve:
  Big cores: High perf, high power
  LITTLE: Good perf/watt at low load
  
  Optimal: Use LITTLE until needed,
           then migrate to Big

Real-world Example (Smartphone):
  Browsing: LITTLE cores (1-2W)
  Gaming:   Big cores (5-8W)
  Standby:  Ultra-low power mode
  
  Battery life improvement: 50-70%
```

### 8.3.2 Intel Hybrid Architecture (Alder Lake)

P-cores (Performance) and E-cores (Efficiency).

```
Intel 12th Gen (Alder Lake):

┌─────────────────────────────────────────┐
│              CPU Die                    │
│                                         │
│  P-Cores (Golden Cove)                  │
│  ┌────────┐  ┌────────┐                 │
│  │ P-Core │  │ P-Core │                 │
│  │   0    │  │   1    │  ┌───────┐      │
│  │        │  │        │  │ HT 0  │      │
│  │  OOO   │  │  OOO   │  │ HT 1  │      │
│  │  L1/L2 │  │  L1/L2 │  └───────┘      │
│  └────────┘  └────────┘   2 threads/core│
│    ... (up to 8 P-cores)                │
│                                         │
│  E-Cores (Gracemont)                    │
│  ┌──────┬──────┬──────┬──────┐          │
│  │E-Core│E-Core│E-Core│E-Core│          │
│  │  0   │  1   │  2   │  3   │          │
│  │      │      │      │      │          │
│  │ OOO  │ OOO  │ OOO  │ OOO  │          │
│  │ L2 Shared across 4 E-cores│          │
│  └──────┴──────┴──────┴──────┘          │
│  E-cores in clusters of 4               │
│    ... (up to 16 E-cores)               │
│                                         │
│  ┌───────────────────────────────┐      │
│  │      Shared L3 Cache          │      │
│  │         (30 MB)               │      │
│  └───────────────────────────────┘      │
└─────────────────────────────────────────┘

Configuration Example (i9-12900K):
  8 P-cores (16 threads with HT)
  8 E-cores (8 threads, no HT)
  Total: 24 threads

Core Characteristics:
┌──────────────┬─────────┬──────────┐
│ Feature      │ P-Core  │  E-Core  │
├──────────────┼─────────┼──────────┤
│ Peak Freq    │ 5.2 GHz │ 3.9 GHz  │
│ IPC          │  High   │  Medium  │
│ Hyper-Thread │  Yes    │   No     │
│ AVX-512      │  No*    │   No     │
│ L2 Cache     │ 1.25 MB │2MB/4cores│
│ Area         │  Large  │  Small   │
│ Power        │  ~30W   │  ~4W     │
└──────────────┴─────────┴──────────┘

Thread Director (Hardware):
┌────────────────────────────────────┐
│    Hardware Feedback Interface     │
│                                    │
│  Monitors per-thread:              │
│  - IPC (Instructions per cycle)    │
│  - Memory bandwidth usage          │
│  - Instruction mix                 │
│  - Branch mispredictions           │
│                                    │
│  Telemetry ──► OS Scheduler        │
└────────────────────────────────────┘

OS Scheduler Decision:
  IF (thread.priority == HIGH &&
      thread.IPC > threshold)
      Schedule on P-core
  ELSE IF (thread.background ||
           thread.IPC < threshold)
      Schedule on E-core
  END IF

Performance Scaling:
Single-threaded:
  P-core: 100% (baseline)
  E-core: 70%

Multi-threaded (all cores):
  8P+8E: ~180% vs 8P alone
  (Not 200% due to E-core efficiency)

Power Efficiency:
  Lightly threaded: E-cores 40% less power
  Background tasks: E-cores handle, P-cores sleep
  
Example Workloads:
┌────────────────────┬──────────────┐
│   Workload         │  Core Type   │
├────────────────────┼──────────────┤
│ Game main thread   │   P-Core     │
│ Game audio/physics │   E-Core     │
│ Video encoding     │   P-Core     │
│ Background indexing│   E-Core     │
│ Browser tab 1      │   P-Core     │
│ Browser tabs 2-10  │   E-Core     │
│ Antivirus scan     │   E-Core     │
└────────────────────┴──────────────┘

Benefits:
+ Higher multi-threaded perf (more cores)
+ Better power efficiency
+ More responsive (background on E-cores)
+ Smaller die than all P-cores

Challenges:
- OS scheduler complexity
- Thread affinity issues
- Legacy software unaware
- Debugging more complex
```

## 8.4 HSA (Heterogeneous System Architecture)

Unified programming model for CPU+GPU.

```
Traditional CPU+GPU:
┌──────────┐              ┌──────────┐
│   CPU    │              │   GPU    │
│          │              │          │
│ System   │              │  VRAM    │
│ Memory   │              │(Separate)│
└────┬─────┘              └────┬─────┘
     │                         │
     └──────────┬──────────────┘
                │
          PCIe (slow!)
          
Problems:
- Separate memory spaces
- Explicit data copy required
- CPU and GPU can't share pointers
- Latency overhead (PCIe)

HSA Architecture:
┌──────────────────────────────────────┐
│     Unified Memory Address Space     │
│                                      │
│  ┌──────────┐      ┌──────────┐      │
│  │   CPU    │      │   GPU    │      │
│  │          │      │          │      │
│  │ Coherent │◄────►│ Coherent │      │
│  │  Cache   │      │  Cache   │      │
│  └────┬─────┘      └────┬─────┘      │
│       │                 │            │
│       └────────┬────────┘            │
│                ▼                     │
│        ┌───────────────┐             │
│        │ Shared Memory │             │
│        │  (Coherent)   │             │
│        └───────────────┘             │
└──────────────────────────────────────┘

Key Features:

1. Unified Address Space:
   void* ptr = malloc(size);
   
   // CPU can use it
   cpu_process(ptr);
   
   // GPU can use same pointer!
   gpu_kernel<<<...>>>(ptr);
   
   // No explicit copy needed

2. Cache Coherency:
   CPU writes ──► Cache ──► Memory
                    │
   GPU reads ◄──────┘
   (Automatic coherence!)

3. User-Level Queuing:
   ┌──────────────────────────────┐
   │  Application (User Space)    │
   │                              │
   │  Dispatch packet to GPU ────►│
   │  (No kernel involvement!)    │
   └──────────────────────────────┘
   
   Lower latency dispatch

4. Preemption:
   GPU can be interrupted
   Better responsiveness
   
   High-priority task ──► Preempt GPU
                         Save context
                         Run new task
                         Restore context

HSA Memory Model:
┌──────────────────────────────────────┐
│         Memory Regions               │
│                                      │
│ ┌────────────────────────────────┐   │
│ │   Global Memory                │   │
│ │   (Coherent across all agents) │   │
│ └────────────────────────────────┘   │
│                                      │
│ ┌────────────────────────────────┐   │
│ │   Group Memory                 │   │
│ │   (Shared within workgroup)    │   │
│ └────────────────────────────────┘   │
│                                      │
│ ┌────────────────────────────────┐   │
│ │   Private Memory               │   │
│ │   (Per work-item)              │   │
│ └────────────────────────────────┘   │
└──────────────────────────────────────┘

Programming Model (OpenCL 2.0+ / ROCm):

C++ AMP Style:
  array_view<float> data(size, ptr);
  
  parallel_for_each(extent, [=](index<1> i) {
      data[i] = data[i] * 2.0f;
  });
  
  // Automatically runs on GPU
  // No explicit copy

AMD APU Example:
┌────────────────────────────────────────┐
│            AMD Ryzen APU               │
│                                        │
│  ┌──────────────┐   ┌──────────────┐   │
│  │  Zen Cores   │   │  Radeon GPU  │   │
│  │  (4-8 cores) │   │  (RDNA)      │   │
│  └──────┬───────┘   └──────┬───────┘   │ 
│         │                  │           │
│         └────────┬─────────┘           │
│                  ▼                     │
│         ┌─────────────────┐            │
│         │  Infinity Fabric │           │
│         │  (Interconnect)  │           │
│         └────────┬─────────┘           │
│                  ▼                     │
│         ┌─────────────────┐            │
│         │  DDR4/DDR5 RAM  │            │
│         │  (Shared)       │            │
│         └─────────────────┘            │
└────────────────────────────────────────┘

Benefits:
+ Simplified programming
+ Lower latency (no PCIe)
+ Better memory utilization
+ Fine-grained sharing

Use Cases:
- Machine learning inference
- Image/video processing
- Scientific computing
- Real-time graphics
```

## 8.5 Domain-Specific Accelerators

### 8.5.1 TPU (Tensor Processing Unit)

Google's AI accelerator.

```
TPU Architecture (v4):

┌─────────────────────────────────────────┐
│              TPU Chip                   │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │    Matrix Multiply Unit (MXU)     │  │
│  │                                   │  │
│  │    Systolic Array                 │  │
│  │    128 × 128 MAC units            │  │
│  │                                   │  │
│  │    Peak: 275 TFLOPS (BF16)        │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌────────────┐     ┌────────────┐      │
│  │   Vector   │     │   Scalar   │      │
│  │    Unit    │     │    Unit    │      │
│  │ (Activation│     │  (Control) │      │
│  │  functions)│     │            │      │
│  └────────────┘     └────────────┘      │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │    High Bandwidth Memory (HBM)    │  │
│  │         32 GB @ 1.2 TB/s          │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘

Systolic Array:
  Data flows through processing elements
  
  Weights ──►
  ┌───┬───┬───┬───┐
  │MAC│MAC│MAC│MAC│
  ├───┼───┼───┼───┤  Inputs
  │MAC│MAC│MAC│MAC│    │
  ├───┼───┼───┼───┤    ▼
  │MAC│MAC│MAC│MAC│
  ├───┼───┼───┼───┤
  │MAC│MAC│MAC│MAC│
  └───┴───┴───┴───┘
    │
    ▼
  Results

  Each MAC (Multiply-Accumulate):
    out = a × weight + in
    
  Highly efficient for matrix multiplication
  (Neural network core operation)

Matrix Multiplication:
  C[128×128] = A[128×128] × B[128×128]
  
  CPU: Thousands of cycles
  GPU: Hundreds of cycles
  TPU: 128 cycles (one element per cycle!)

Comparison (BF16 Matrix Multiply):
┌──────────┬───────────┬──────────┬───────┐
│ Device   │  TFLOPS   │ Memory   │ Power │
├──────────┼───────────┼──────────┼───────┤
│ CPU      │    ~1     │ 256 GB/s │ 200W  │
│ GPU      │   ~100    │   1 TB/s │ 300W  │
│ TPU v4   │   275     │ 1.2 TB/s │ 175W  │
└──────────┴───────────┴──────────┴───────┘

TPU is ~275× faster than CPU for AI!

Optimized for:
- Matrix multiplication
- Convolution
- Batch normalization
- ReLU activation

Not good for:
- General purpose code
- Branching
- Random memory access

TPU Pod (Multi-chip):
┌────┬────┬────┬────┐
│TPU │TPU │TPU │TPU │
├────┼────┼────┼────┤
│TPU │TPU │TPU │TPU │
├────┼────┼────┼────┤
│TPU │TPU │TPU │TPU │
├────┼────┼────┼────┤
│TPU │TPU │TPU │TPU │
└────┴────┴────┴────┘
   3D Torus Network
   
v4 Pod: 4096 TPUs
Peak: 1.1 Exaflops!
```

### 8.5.2 NPU (Neural Processing Unit)

Mobile AI accelerators.

```
NPU Integration (Smartphone SoC):

┌─────────────────────────────────────────┐
│            Mobile SoC                   │
│                                         │
│  ┌──────────┐        ┌──────────┐       │
│  │   CPU    │        │   GPU    │       │
│  │ (4+4)    │        │  (Mali)  │       │
│  │ Cores    │        │          │       │
│  └──────────┘        └──────────┘       │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         NPU / AI Engine           │  │
│  │                                   │  │
│  │  - Convolution engines            │  │
│  │  - Pooling units                  │  │
│  │  - Activation functions           │  │
│  │  - Quantized inference (INT8)     │  │
│  │                                   │  │
│  │  Performance: 5-30 TOPS           │  │
│  │  Power: 0.5-2W                    │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌──────────┐  ┌──────────┐             │
│  │   ISP    │  │   DSP    │             │
│  │ (Camera) │  │ (Audio)  │             │
│  └──────────┘  └──────────┘             │
└─────────────────────────────────────────┘

NPU Design:

┌─────────────────────────────────────┐
│      Convolution Accelerator        │
│                                     │
│  Input Feature Maps ──►             │
│  ┌─────────────────────────────┐    │
│  │  Convolution Engine         │    │
│  │                             │    │
│  │  ┌─────────────┐            │    │
│  │  │Weight Cache │            │    │
│  │  └─────────────┘            │    │
│  │                             │    │
│  │  MAC Array (128x)           │    │
│  │  ┌──┬──┬──┬──┬──┐           │    │
│  │  │  │  │  │  │  │ ...       │    │
│  │  └──┴──┴──┴──┴──┘           │    │
│  └─────────────────────────────┘    │
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────┐    │
│  │  Activation (ReLU, etc.)    │    │
│  └─────────────────────────────┘    │
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────┐    │
│  │  Pooling (Max, Avg)         │    │
│  └─────────────────────────────┘    │
│              │                      │
│              ▼                      │
│   Output Feature Maps               │
└─────────────────────────────────────┘

Quantization (INT8 vs FP32):

FP32 (32 bits):
  Weight: [-0.1234567] (4 bytes)
  
INT8 (8 bits):
  Weight: [-12] (1 byte)
  Scale factor: 0.01
  
  Benefits:
  + 4× memory reduction
  + 4× bandwidth reduction
  + Faster computation
  + Lower power
  
  Accuracy loss: ~1-2%

Performance Comparison (Image Classification):
┌──────────┬──────────┬───────┬────────┐
│ Device   │ Latency  │ Power │ Frames │
│          │  (ms)    │  (W)  │  /sec  │
├──────────┼──────────┼───────┼────────┤
│ CPU only │   150    │  2.5  │    7   │
│ GPU      │    50    │  1.5  │   20   │
│ NPU      │    10    │  0.5  │  100   │
└──────────┴──────────┴───────┴────────┘

NPU is 15× faster, 5× more efficient!

Use Cases:
- Face unlock
- Photo enhancement
- Voice recognition
- AR effects
- Object detection
- Translation

Apple Neural Engine:
  16 cores
  11 TOPS (A15)
  15.8 TOPS (A16)
  17 TOPS (A17 Pro)
  
Google Tensor:
  TPU-based NPU
  Edge TPU technology
  
Qualcomm Hexagon:
  AI accelerator + DSP
  Tensor acceleration
```

### 8.5.3 Video Encoding/Decoding Accelerators

```
Fixed-Function Video Engines:

┌─────────────────────────────────────────┐
│          Video Encode/Decode            │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │      Video Decoder                │  │
│  │                                   │  │
│  │  - H.264/AVC                      │  │
│  │  - H.265/HEVC                     │  │
│  │  - VP9                            │  │
│  │  - AV1                            │  │
│  │                                   │  │
│  │  Hardware decode stages:          │  │
│  │  ┌──────────────────────────┐     │  │
│  │  │ Entropy Decode           │     │  │
│  │  └───────┬──────────────────┘     │  │
│  │          ▼                        │  │
│  │  ┌──────────────────────────┐     │  │
│  │  │ Inverse Transform        │     │  │
│  │  └───────┬──────────────────┘     │  │
│  │          ▼                        │  │
│  │  ┌──────────────────────────┐     │  │
│  │  │ Motion Compensation      │     │  │
│  │  └───────┬──────────────────┘     │  │
│  │          ▼                        │  │
│  │  ┌──────────────────────────┐     │  │
│  │  │ Deblocking Filter        │     │  │
│  │  └───────┬──────────────────┘     │  │
│  │          ▼                        │  │
│  │     Decoded Frame                 │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │      Video Encoder                │  │
│  │  (Similar pipeline in reverse)    │  │
│  │  - Rate control                   │  │
│  │  - Motion estimation              │  │
│  │  - Transform & quantization       │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘

Performance (4K 60fps HEVC decode):
┌──────────────┬──────────┬───────┬────────┐
│   Method     │   CPU    │ Power │Quality │
│              │   Load   │  (W)  │        │
├──────────────┼──────────┼───────┼────────┤
│ Software     │   100%   │  20W  │  Good  │
│ GPU Compute  │    40%   │  10W  │  Good  │
│ Fixed HW     │     5%   │  1W   │Perfect │
└──────────────┴──────────┴───────┴────────┘

Hardware: 20× more efficient!

Intel Quick Sync:
  Integrated in GPU
  Up to 8K encoding
  Multi-stream support
  
NVIDIA NVENC/NVDEC:
  Dedicated encoder/decoder
  8× 4K streams simultaneously
  
Apple VideoToolbox:
  ProRes accelerator
  H.264, HEVC, ProRes
  Media engine in M-series
```

## 8.6 System-on-Chip (SoC) Integration

Modern mobile and embedded systems.

```
Typical Smartphone SoC:

┌─────────────────────────────────────────────┐
│              Application Processor          │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │         CPU Cluster                 │    │
│  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐        │    │
│  │  │Big │ │Big │ │Mid │ │Lit │        │    │
│  │  └────┘ └────┘ └────┘ └────┘        │    │
│  │         ┌───────────┐               │    │
│  │         │ L3 Cache  │               │    │
│  │         └───────────┘               │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │         GPU (Graphics)              │    │
│  │  - Shader cores                     │    │
│  │  - Texture units                    │    │
│  │  - Rasterizer                       │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   NPU    │  │   DSP    │  │   ISP    │   │
│  │ (Neural) │  │ (Audio)  │  │ (Camera) │   │
│  └──────────┘  └──────────┘  └──────────┘   │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   VPU    │  │ Security │  │  Modem   │   │
│  │ (Video)  │  │ (Crypto) │  │   5G     │   │
│  └──────────┘  └──────────┘  └──────────┘   │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │     System Cache / Interconnect     │    │
│  │        (NoC - Network on Chip)      │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  ┌────────┐ ┌────────┐ ┌────────┐           │
│  │ LPDDR5 │ │  UFS   │ │  I/O   │           │
│  │  RAM   │ │Storage │ │ (USB,  │           │
│  │  Ctrl  │ │  Ctrl  │ │ PCIe)  │           │
│  └────────┘ └────────┘ └────────┘           │
└─────────────────────────────────────────────┘

Power Domains:
┌────────────────────────────────────┐
│  Always On:                        │
│  - Low-power core                  │
│  - Sensor hub                      │
│  - RTC                             │
│  Power: ~1 mW                      │
├────────────────────────────────────┤
│  Active Display:                   │
│  - LITTLE cores                    │
│  - Display controller              │
│  - GPU (low power)                 │
│  Power: ~500 mW                    │
├────────────────────────────────────┤
│  Active Performance:               │
│  - All cores                       │
│  - GPU full                        │
│  - NPU                             │
│  - Modem                           │
│  Power: ~5-8 W                     │
└────────────────────────────────────┘

Task → Accelerator Mapping:
┌─────────────────────┬──────────────┐
│      Task           │ Accelerator  │
├─────────────────────┼──────────────┤
│ UI rendering        │  GPU + CPU   │
│ Photo capture       │  ISP → NPU   │
│ Video playback      │  VPU         │
│ Voice assistant     │  DSP + NPU   │
│ Face unlock         │  NPU         │
│ 5G data             │  Modem       │
│ Encryption          │  Crypto      │
│ Background tasks    │  LITTLE cores│
└─────────────────────┴──────────────┘

Apple M2 (Desktop-class SoC):
┌──────────────────────────────────────┐
│  8 CPU cores (4P + 4E)               │
│  10 GPU cores                        │
│  16-core Neural Engine (15.8 TOPS)   │
│  Video encode/decode engines         │
│  Display engine (supports 2× 6K)     │
│  Unified Memory (up to 24 GB)        │
│  100 GB/s bandwidth                  │
│                                      │
│  Integration = Efficiency            │
└──────────────────────────────────────┘
```

## 8.7 Interconnects for Heterogeneous Systems

```
Chip-to-Chip Interconnects:

1. Intel QPI/UPI (CPU-CPU):
┌────────┐    QPI/UPI     ┌────────┐
│  CPU   │◄─────────────► │  CPU   │
│Socket 0│   16-20 GT/s   │Socket 1│
└────────┘   64-bit wide  └────────┘

Bandwidth: ~40 GB/s per link
Latency: ~100 ns

2. AMD Infinity Fabric:
┌────────┐              ┌────────┐
│Chiplet │◄────────────►│Chiplet │
│  (CCD) │   32 GB/s    │  (CCD) │
└────────┘              └────────┘
     │                       │
     └──────────┬────────────┘
                ▼
          ┌──────────┐
          │   I/O    │
          │   Die    │
          └──────────┘

Scalable, modular design

3. NVIDIA NVLink (GPU-GPU):
┌────────┐    NVLink      ┌────────┐
│  GPU   │ ◄─────────────►│  GPU   │
│   0    │  600 GB/s      │   1    │
└────────┘  (12 links)    └────────┘

Much faster than PCIe!

4. PCIe (CPU-GPU):
┌────────┐    PCIe 5.0    ┌────────┐
│  CPU   │ ◄─────────────►│  GPU   │
│        │  x16: 64GB/s   │        │
└────────┘                └────────┘

Standard but slower

5. CXL (Compute Express Link):
┌────────┐      CXL       ┌────────┐
│  CPU   │ ◄─────────────►│ Accel. │
│        │    PCIe PHY    │  GPU/  │
│        │   + coherency  │  FPGA  │
└────────┘                └────────┘

Cache-coherent accelerators!

Network-on-Chip (NoC):
┌────────────────────────────────────┐
│        Mesh Topology               │
│                                    │
│  CPU ───┬─── GPU ───┬─── NPU       │
│         │           │              │
│  L3  ───┼─── MEM ───┼─── I/O       │
│         │           │              │
│  DSP ───┴─── VPU ───┴─── PCIE      │
│                                    │
│  Packet-switched network           │
│  Quality of Service (QoS)          │
│  Multiple concurrent transfers     │
└────────────────────────────────────┘

Coherency Protocols:

1. MOESI (AMD):
   M - Modified
   O - Owned  (shared but modified)
   E - Exclusive
   S - Shared
   I - Invalid

2. MESIF (Intel):
   F - Forward (one cache forwards to others)

3. CHI (ARM):
   Coherent Hub Interface
   Scalable to many nodes
```

## 8.8 Programming Heterogeneous Systems

```
Challenge: Multiple programming models

┌──────────────────────────────────────┐
│         Traditional Approach         │
│                                      │
│  CPU code:    C/C++                  │
│  GPU code:    CUDA/OpenCL            │
│  DSP code:    C + intrinsics         │
│  NPU code:    TensorFlow/PyTorch     │
│                                      │
│  Problem: Different tools, APIs,     │
│           debugging, profiling       │
└──────────────────────────────────────┘

Unified Programming Models:

1. OpenCL:
   // Same code, different devices
   cl_device_id cpu, gpu, fpga;
   
   clGetDeviceIDs(CPU_TYPE, &cpu);
   clGetDeviceIDs(GPU_TYPE, &gpu);
   clGetDeviceIDs(ACCEL_TYPE, &fpga);
   
   kernel void process(__global float* data) {
       int i = get_global_id(0);
       data[i] = compute(data[i]);
   }
   
   // Can run on any device!

2. SYCL (Modern C++):
   queue q(gpu_selector{});
   
   q.parallel_for(range<1>(N), [=](id<1> i) {
       result[i] = input[i] * 2.0f;
   });
   
   // Single-source C++
   // CPU and GPU in same file

3. oneAPI (Intel):
   // Unified across CPU, GPU, FPGA
   sycl::queue q(sycl::gpu_selector{});
   
   q.submit([&](sycl::handler& h) {
       h.parallel_for(N, [=](auto i) {
           // kernel code
       });
   });

4. Heterogeneous Task Graphs:
   ┌─────────────────────────────┐
   │  Task 1 (CPU)               │
   └──────┬──────────────────────┘
          │
     ┌────┴────┐
     ▼         ▼
   ┌────┐   ┌──────┐
   │GPU │   │ NPU  │  (Parallel)
   └──┬─┘   └───┬──┘
      │         │
      └────┬────┘
           ▼
   ┌─────────────┐
   │  Task 4 (CPU)│
   └─────────────┘
   
   Framework schedules optimally

Offload Patterns:

1. Offload & Wait:
   CPU: ──────┐
              │ GPU Kernel ──┐
              └──────┬───────┘
                     │ (Wait)
              ┌──────┘
              └────────────────►

2. Offload & Continue:
   CPU: ──────┐
              │ ──────────────►
              │
              └─► GPU Kernel ──►
              
3. Pipeline:
   CPU:  [Prep1][Prep2][Prep3][Prep4]
              │    │     │     │
   GPU:      [K1] [K2]  [K3]  [K4]
              │    │     │     │
   CPU:     [Post1][Post2][Post3]

Auto-tuning and Scheduling:
┌──────────────────────────────────┐
│    Runtime Scheduler             │
│                                  │
│  Profile workload:               │
│  - Data size                     │
│  - Computation complexity        │
│  - Memory patterns               │
│                                  │
│  Decide:                         │
│  IF (small data)                 │
│      Run on CPU                  │
│  ELSE IF (parallel)              │
│      Run on GPU                  │
│  ELSE IF (sequential ML)         │
│      Run on NPU                  │
│  END IF                          │
└──────────────────────────────────┘

Performance Monitoring:
┌──────────┬───────┬───────┬───────┐
│Component │Utiliz.│ Power │ Temp  │
├──────────┼───────┼───────┼───────┤
│ CPU      │  45%  │  15W  │  65°C │
│ GPU      │  80%  │  25W  │  75°C │
│ NPU      │  90%  │   3W  │  55°C │
│ Memory   │  60%  │   2W  │  50°C │
└──────────┴───────┴───────┴───────┘

Optimize based on metrics!
```

## 8.9 Future Trends

```
1. Chiplet-based Heterogeneous Systems:
┌──────────────────────────────────────┐
│  Mix and match chiplets:             │
│                                      │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐         │
│  │CPU │ │GPU │ │NPU │ │FPGA│         │
│  └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘         │
│     └──────┴──────┴──────┘           │
│            │                         │
│      ┌─────▼─────┐                   │
│      │ Interposer│                   │
│      │  (Silicon)│                   │
│      └───────────┘                   │
│                                      │
│  Benefits:                           │
│  - Modular (mix vendors)             │
│  - Better yield                      │
│  - Customizable configs              │
└──────────────────────────────────────┘

2. In-Memory Computing:
   Compute in DRAM/ReRAM
   Eliminate data movement
   1000× more efficient for AI

3. Photonics:
   Optical interconnects
   100× bandwidth
   Lower latency

4. Quantum Co-processors:
   Classical + Quantum hybrid
   Quantum for specific problems

5. Neuromorphic + Traditional:
   Spiking + Von Neumann
   Ultra-low power AI

The future is heterogeneous!
Different problems need different solutions
No one-size-fits-all architecture
```

---

**Key Takeaways:**
1. Heterogeneous systems combine specialized accelerators
2. NUMA important for multi-socket scaling
3. big.LITTLE and hybrid architectures improve efficiency
4. Domain-specific accelerators (TPU, NPU) provide huge gains
5. Interconnects and coherency are critical
6. Unified programming models simplify development
7. SoCs integrate diverse functionality on one chip
8. Future: More specialization, chiplets, novel computing

**Previous:** [Advanced Topics](./07-advanced-topics.md) | **Next:** [Performance Analysis](./09-performance-analysis.md) | **Index:** [README](./README.md)


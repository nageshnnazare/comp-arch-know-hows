# Chapter 9: Performance Analysis & Optimization

## 9.1 Performance Metrics Overview

Understanding and measuring performance is critical for system design and optimization.

```
┌─────────────────────────────────────────────────┐
│        Performance Measurement Pyramid          │
│                                                 │
│              ┌─────────────┐                    │
│              │   Latency   │ ← Response Time    │
│              └──────┬──────┘                    │
│                     │                           │
│              ┌──────▼──────┐                    │
│              │ Throughput  │ ← Work/Time        │
│              └──────┬──────┘                    │
│                     │                           │
│              ┌──────▼──────┐                    │
│              │   Bandwidth │ ← Data/Time        │
│              └──────┬──────┘                    │
│                     │                           │
│              ┌──────▼──────┐                    │
│              │   Efficiency│ ← Performance/Watt │
│              └─────────────┘                    │
└─────────────────────────────────────────────────┘

Key Metrics:

1. Latency:
   Time to complete a single operation
   Units: seconds, nanoseconds
   Example: Memory access = 100 ns

2. Throughput:
   Operations per unit time
   Units: ops/sec, instructions/sec
   Example: 1 billion instructions/sec

3. Bandwidth:
   Data transferred per unit time
   Units: bytes/sec, GB/s
   Example: Memory bandwidth = 100 GB/s

4. Utilization:
   Percentage of resource used
   Units: percentage (0-100%)
   Example: CPU utilization = 75%

5. Energy Efficiency:
   Work per unit of energy
   Units: GFLOPS/Watt, Instructions/Joule
   Example: 50 GFLOPS/Watt

Relationship:
┌────────────────────────────────────────┐
│  Throughput = Operations / Time        │
│  Bandwidth  = Data / Time              │
│  Latency    = Time / Operation         │
│  Utilization = Busy_Time / Total_Time  │
│  Efficiency = Performance / Power      │
└────────────────────────────────────────┘

Trade-offs:
  ↑ Throughput         ↑ Latency (often)
  ↑ Performance    →   ↑ Power
  ↑ Complexity     →   ↓ Reliability
```

## 9.2 CPU Performance Analysis

### 9.2.1 Basic CPU Performance Equation

```
CPU Time = Instruction Count × CPI × Clock Cycle Time

Or equivalently:

CPU Time = (Instructions / Program) × 
           (Cycles / Instruction) × 
           (Seconds / Cycle)

Components:

1. Instruction Count (IC):
   - Determined by: ISA, compiler, algorithm
   - Example: 1 million instructions

2. Cycles Per Instruction (CPI):
   - Determined by: CPU microarchitecture
   - Affected by: hazards, cache misses, branch mispredictions
   - Example: CPI = 1.5

3. Clock Cycle Time:
   - Determined by: Technology, design
   - Inverse of clock frequency
   - Example: 3 GHz = 0.333 ns/cycle

Example Calculation:
┌────────────────────────────────────────┐
│ Program A:                             │
│   Instructions: 10 million             │
│   CPI: 2.0                             │
│   Clock: 3 GHz (0.333 ns/cycle)        │
│                                        │
│ CPU Time = 10M × 2.0 × 0.333ns         │
│          = 6.66 ms                     │
│                                        │
│ MIPS = (10M / 6.66ms) / 1M             │
│      = 1500 MIPS                       │
└────────────────────────────────────────┘

Alternative Form (using frequency):
CPU Time = IC × CPI / Clock_Frequency

MIPS (Million Instructions Per Second):
MIPS = Clock_Frequency (MHz) / CPI

Note: MIPS can be misleading!
  - Different ISAs have different instructions
  - Same MIPS doesn't mean same performance
  - Use execution time for fair comparison
```

### 9.2.2 Detailed CPI Breakdown

```
CPI = CPI_ideal + CPI_stalls

CPI_stalls = CPI_structural + CPI_data + CPI_control

Example Breakdown:
┌────────────────────────┬──────────┬──────────┐
│     Component          │  Cycles  │    %     │
├────────────────────────┼──────────┼──────────┤
│ Base CPI (ideal)       │   1.0    │   50%    │
│ Structural hazards     │   0.1    │    5%    │
│ Data hazards           │   0.4    │   20%    │
│ Control hazards        │   0.3    │   15%    │
│ Cache misses           │   0.2    │   10%    │
├────────────────────────┼──────────┼──────────┤
│ Total CPI              │   2.0    │  100%    │
└────────────────────────┴──────────┴──────────┘

Detailed CPI Calculation:

CPI = Σ (CPI_i × Frequency_i)

Where:
  CPI_i = CPI for instruction type i
  Frequency_i = Fraction of instructions of type i

Example:
┌──────────────┬──────┬───────────┬──────────┐
│ Instruction  │ CPI  │ Frequency │ Contrib. │
├──────────────┼──────┼───────────┼──────────┤
│ ALU          │ 1.0  │   45%     │   0.45   │
│ Load         │ 2.0  │   20%     │   0.40   │
│ Store        │ 2.0  │   10%     │   0.20   │
│ Branch       │ 1.5  │   15%     │   0.23   │
│ Other        │ 1.0  │   10%     │   0.10   │
├──────────────┼──────┼───────────┼──────────┤
│ Average      │      │   100%    │   1.38   │
└──────────────┴──────┴───────────┴──────────┘

Weighted CPI = 1.38
```

### 9.2.3 Speedup and Improvement

```
Amdahl's Law (revisited):

Speedup_overall = 1 / ((1 - P) + P/S)

Where:
  P = Fraction of program affected
  S = Speedup of that fraction

Example: Optimize floating-point operations
┌──────────────────────────────────────┐
│ Original program:                    │
│   FP operations: 40% of time         │
│   Other: 60% of time                 │
│                                      │
│ FP unit 2× faster:                   │
│   P = 0.4, S = 2                     │
│                                      │
│ Speedup = 1 / (0.6 + 0.4/2)          │
│         = 1 / (0.6 + 0.2)            │
│         = 1 / 0.8                    │
│         = 1.25×                      │
│                                      │
│ Only 25% improvement overall!        │
└──────────────────────────────────────┘

Visualization:
Original: [███████ Other 60% ███████][████ FP 40% ████]
After:    [███████ Other 60% ███████][██ FP 20% ██]
          └─────────────────────────────────┘
                    80% of original time
          Speedup = 100/80 = 1.25×

Key Insight:
  "Make the common case fast"
  Optimize what takes most time!

Multiple Improvements:

Time_new = Time_old × Π (1 / Speedup_i)

Example:
  Original: 100 seconds
  Cache improvement: 1.2× speedup
  Branch prediction: 1.1× speedup
  
  Time_new = 100 / (1.2 × 1.1)
           = 100 / 1.32
           = 75.76 seconds
  
  Overall speedup: 1.32×
```

## 9.3 Memory System Performance

### 9.3.1 Cache Performance

```
Average Memory Access Time (AMAT):

AMAT = Hit_Time + Miss_Rate × Miss_Penalty

Example: L1 Cache
┌──────────────────────────────────────┐
│ L1 hit time: 1 cycle                 │
│ L1 miss rate: 5%                     │
│ L2 access time: 10 cycles            │
│                                      │
│ AMAT = 1 + 0.05 × 10                 │
│      = 1.5 cycles                    │
└──────────────────────────────────────┘

Multi-Level Cache:

AMAT = Hit_L1 + MR_L1 × (Hit_L2 + MR_L2 × Hit_L3 + 
       MR_L3 × Mem_Time)

Example: 3-level cache
┌────────┬──────────┬────────┬─────────┐
│ Level  │ Hit Time │ Miss % │ Access  │
├────────┼──────────┼────────┼─────────┤
│ L1     │   1 ns   │   5%   │   1 ns  │
│ L2     │  10 ns   │  20%   │  10 ns  │
│ L3     │  30 ns   │  10%   │  30 ns  │
│ Memory │ 100 ns   │   -    │ 100 ns  │
└────────┴──────────┴────────┴─────────┘

AMAT = 1 + 0.05 × (10 + 0.20 × (30 + 0.10 × 100))
     = 1 + 0.05 × (10 + 0.20 × 40)
     = 1 + 0.05 × (10 + 8)
     = 1 + 0.05 × 18
     = 1 + 0.9
     = 1.9 ns average

Effective Bandwidth:

BW_eff = BW_peak × Hit_Rate

Example:
  Peak bandwidth: 100 GB/s
  Cache hit rate: 95%
  
  Effective BW = 100 × 0.95 = 95 GB/s

Cache Miss Classification:
┌──────────────┬─────────────────────────┐
│  Miss Type   │       Cause             │
├──────────────┼─────────────────────────┤
│ Compulsory   │ First access (cold)     │
│ Capacity     │ Cache too small         │
│ Conflict     │ Mapping collision       │
│ Coherence    │ Invalidation (MP)       │
└──────────────┴─────────────────────────┘

Miss Rate Reduction:

1. Increase cache size:
   32 KB → 64 KB: ~30% miss rate reduction
   (Diminishing returns)

2. Increase associativity:
   Direct → 2-way: ~15% reduction
   2-way → 4-way: ~10% reduction
   4-way → 8-way: ~5% reduction

3. Larger block size:
   32B → 64B: Better spatial locality
   But: More conflict misses possible

4. Prefetching:
   Hardware or software
   Reduce compulsory misses
```

### 9.3.2 TLB Performance

```
TLB Miss Penalty:

TLB_Miss_Penalty = Page_Table_Levels × Memory_Access_Time

Example: 4-level page table
┌──────────────────────────────────────┐
│ Memory access time: 100 ns           │
│ Page table levels: 4                 │
│                                      │
│ TLB miss penalty = 4 × 100 = 400 ns  │
│                                      │
│ TLB hit time: 1 ns                   │
│ TLB miss rate: 1%                    │
│                                      │
│ Effective access = 1 + 0.01 × 400    │
│                  = 5 ns              │
│                                      │
│ With TLB: 5 ns                       │
│ Without TLB: 400 ns                  │
│ Speedup: 80×                         │
└──────────────────────────────────────┘

Combined Cache + TLB:

Total_Time = TLB_Time + Cache_Time + Memory_Time

Scenarios:
┌──────────────┬──────────┬───────────┐
│   Scenario   │   Time   │  Comment  │
├──────────────┼──────────┼───────────┤
│ TLB hit,     │   2 ns   │   Best    │
│ Cache hit    │          │           │
│              │          │           │
│ TLB hit,     │  102 ns  │   Common  │
│ Cache miss   │          │           │
│              │          │           │
│ TLB miss,    │  502 ns  │   Rare    │
│ Cache hit    │          │           │
│              │          │           │
│ TLB miss,    │  602 ns  │   Worst   │
│ Cache miss   │          │           │
└──────────────┴──────────┴───────────┘

Page Size Impact:

Large pages:
  + Fewer TLB misses
  + Fewer page table entries
  - Internal fragmentation
  - Longer page fault handling

Small pages:
  + Less fragmentation
  + Finer-grained protection
  - More TLB misses
  - Larger page tables

Typical: 4 KB default, 2 MB/1 GB huge pages
```

## 9.4 Benchmarking

### 9.4.1 Benchmark Types

```
┌─────────────────────────────────────────────┐
│          Benchmark Categories               │
│                                             │
│  1. Microbenchmarks                         │
│     - Test specific operations              │
│     - Example: Memory latency               │
│     + Fast, focused                         │
│     - May not represent real workload       │
│                                             │
│  2. Kernels                                 │
│     - Core computational loops              │
│     - Example: Matrix multiply              │
│     + Portable across systems               │
│     - Limited scope                         │
│                                             │
│  3. Synthetic Benchmarks                    │
│     - Representative mix of operations      │
│     - Example: Dhrystone, Whetstone         │
│     + Repeatable                            │
│     - May be "gamed" by vendors             │
│                                             │
│  4. Application Benchmarks                  │
│     - Real programs                         │
│     - Example: SPEC CPU, real apps          │
│     + Most realistic                        │
│     - Complex, time-consuming               │
└─────────────────────────────────────────────┘

Major Benchmark Suites:

SPEC CPU (Standard Performance Evaluation Corp):
┌────────────────────────────────────────┐
│ SPEC CPU 2017                          │
│                                        │
│ SPECint (Integer):                     │
│   - Compilers                          │
│   - Interpreters                       │
│   - Compression                        │
│   10+ benchmarks                       │
│                                        │
│ SPECfp (Floating-Point):               │
│   - Scientific computing               │
│   - Physics simulation                 │
│   - Weather modeling                   │
│   13+ benchmarks                       │
│                                        │
│ Metrics:                               │
│   - SPECrate: Throughput               │
│   - SPECspeed: Single-task time        │
└────────────────────────────────────────┘

GeekBench:
┌────────────────────────────────────────┐
│ Cross-platform benchmark               │
│                                        │
│ Single-Core Score                      │
│ Multi-Core Score                       │
│                                        │
│ Tests:                                 │
│ - Integer                              │
│ - Floating-point                       │
│ - Crypto                               │
│ - Memory                               │
└────────────────────────────────────────┘

STREAM (Memory Bandwidth):
┌────────────────────────────────────────┐
│ Four kernel operations:                │
│                                        │
│ Copy:   a[i] = b[i]                    │
│ Scale:  a[i] = q × b[i]                │
│ Add:    a[i] = b[i] + c[i]             │
│ Triad:  a[i] = b[i] + q × c[i]         │
│                                        │
│ Reports: GB/s for each                 │
└────────────────────────────────────────┘

Linpack (Floating-Point):
┌────────────────────────────────────────┐
│ Solve dense linear system              │
│ Ax = b                                 │
│                                        │
│ Used for Top500 supercomputers         │
│                                        │
│ Metric: GFLOPS (billion FLOPS)         │
└────────────────────────────────────────┘

Graphics: 3DMark, Unigine
AI/ML: MLPerf
Mobile: AnTuTu, GFXBench
```

### 9.4.2 Interpreting Results

```
Reporting Performance:

1. Arithmetic Mean (Don't use for ratios!):
   Mean = Σ(Time_i) / n

2. Harmonic Mean (Use for rates):
   HM = n / Σ(1/Rate_i)

3. Geometric Mean (Best for ratios):
   GM = (Π Time_i)^(1/n)

Example: Benchmark Results
┌──────────┬────────┬────────┬────────┐
│Benchmark │System A│System B│ Ratio  │
├──────────┼────────┼────────┼────────┤
│   Test 1 │  10s   │  20s   │  2.0×  │
│   Test 2 │  20s   │  10s   │  0.5×  │
│   Test 3 │  30s   │  30s   │  1.0×  │
└──────────┴────────┴────────┴────────┘

Arithmetic mean of ratios: 1.17× (WRONG!)
Geometric mean: ³√(2.0 × 0.5 × 1.0) = 1.0× (CORRECT)

Statistical Significance:

Run multiple iterations:
┌─────────────────────────────────────┐
│ Run 1: 10.2 seconds                 │
│ Run 2: 10.5 seconds                 │
│ Run 3: 10.1 seconds                 │
│ Run 4: 10.3 seconds                 │
│ Run 5: 10.4 seconds                 │
│                                     │
│ Mean: 10.3 seconds                  │
│ StdDev: 0.16 seconds                │
│ Variance: 1.6%                      │
│                                     │
│ Report: 10.3 ± 0.2 seconds          │
└─────────────────────────────────────┘

Confidence Intervals:
  95% confidence: Mean ± 1.96 × (StdDev/√n)

Comparison Checklist:
☑ Same compiler, flags
☑ Same input data
☑ Warm cache (or cold, consistently)
☑ Multiple runs
☑ Same background load
☑ Same power settings
☑ Measure wall-clock time

Common Pitfalls:
✗ Comparing different ISAs directly
✗ Using peak theoretical performance
✗ Ignoring variance
✗ Cherry-picking best results
✗ Comparing single runs
```

## 9.5 Profiling and Analysis Tools

### 9.5.1 Hardware Performance Counters

```
Modern CPUs have hardware counters for events:

┌─────────────────────────────────────────┐
│    Hardware Performance Counters        │
│                                         │
│  Counter 0: Instructions retired        │
│  Counter 1: Cycles elapsed              │
│  Counter 2: Cache misses (L1D)          │
│  Counter 3: Branch mispredictions       │
│  Counter 4: TLB misses                  │
│  Counter 5: Stall cycles                │
│  ...                                    │
│                                         │
│  Typically 4-8 programmable counters    │
│  + Fixed counters                       │
└─────────────────────────────────────────┘

Intel: Performance Monitoring Unit (PMU)
ARM: Performance Monitor Unit
AMD: Performance Monitoring Counters

Linux perf tool:
  perf stat ./program
  
  Output:
  ┌─────────────────────────────────────┐
  │ 1,234,567,890 cycles                │
  │   987,654,321 instructions          │
  │          0.80 IPC                   │
  │    12,345,678 cache-misses          │
  │   456,789,012 cache-references      │
  │          2.7% cache-miss-rate       │
  │     5,432,109 branch-misses         │
  │   123,456,789 branches              │
  │          4.4% branch-miss-rate      │
  └─────────────────────────────────────┘

Derived Metrics:
  IPC = Instructions / Cycles
  CPI = Cycles / Instructions
  Cache miss rate = Misses / References
  Branch accuracy = (Branches - Misses) / Branches

Event-Based Sampling:
  Sample every N events
  Example: Every 1 million cycles, record PC
  
  Output: Hotspot profile
  ┌──────────────────────┬────────┐
  │      Function        │   %    │
  ├──────────────────────┼────────┤
  │ matrix_multiply      │  45%   │
  │ fft_transform        │  23%   │
  │ sort_array           │  12%   │
  │ ...                  │  ...   │
  └──────────────────────┴────────┘
```

### 9.5.2 Profiling Techniques

```
1. Time-Based Profiling:

   Sample program counter periodically
   
   gprof (GNU Profiler):
   ┌──────────────────────────────────────┐
   │ Flat profile:                        │
   │                                      │
   │  %time   cumulative   self          │
   │          seconds     seconds  calls │
   │ 35.00      0.70       0.70   10000  │
   │          matrix_mult                │
   │ 25.00      1.20       0.50    5000  │
   │          fourier_transform          │
   └──────────────────────────────────────┘

2. Instrumentation:

   Insert timing code
   
   void foo() {
       start_timer();
       // function body
       end_timer();
   }
   
   Overhead: 5-10% typical

3. Event Tracing:

   Record events with timestamps
   
   Timeline:
   ┌────────────────────────────────────┐
   │ Thread 1: [████][  ][████][██]    │
   │ Thread 2: [  ][████████][    ]    │
   │ Thread 3: [██████][  ][██████]    │
   │ Time:     0ms  10ms  20ms  30ms    │
   └────────────────────────────────────┘

4. Hardware Tracing:

   Intel PT (Processor Trace)
   ARM CoreSight
   
   Captures:
   - Branch taken/not-taken
   - Indirect branch targets
   - Timestamps
   
   Post-process to reconstruct execution

Profiling Tools:

Linux:
  - perf: Hardware counters, sampling
  - gprof: Call graph profiling
  - valgrind --tool=callgrind: Detailed
  - oprofile: System-wide profiling

Windows:
  - Visual Studio Profiler
  - Windows Performance Analyzer
  - Intel VTune

Cross-platform:
  - Intel VTune Profiler
  - AMD μProf
  - Arm Streamline
```

### 9.5.3 Bottleneck Identification

```
Top-Down Analysis (Intel):

Level 1: Where are cycles spent?
┌────────────────────────────────────┐
│ Frontend Bound:        25%         │
│   (Instruction fetch/decode)       │
│                                    │
│ Backend Bound:         40%         │
│   (Execution units stalled)        │
│                                    │
│ Bad Speculation:       20%         │
│   (Branch mispredicts)             │
│                                    │
│ Retiring:              15%         │
│   (Useful work)                    │
└────────────────────────────────────┘

Only 15% useful work!

Level 2: Backend Bound breakdown
┌────────────────────────────────────┐
│ Core Bound:            15%         │
│   (Execution units)                │
│                                    │
│ Memory Bound:          25%         │
│   (Cache/memory latency)           │
└────────────────────────────────────┘

Level 3: Memory Bound details
┌────────────────────────────────────┐
│ L1 Bound:               5%         │
│ L2 Bound:               5%         │
│ L3 Bound:               5%         │
│ DRAM Bound:            10%         │
└────────────────────────────────────┘

Conclusion: Optimize DRAM access!

Roofline Model:

Performance vs Operational Intensity

Performance
(GFLOPS)
    ↑
    │         _______________  Peak Performance
    │        /
    │       /
    │      /  Memory Bound
    │     /
    │    /
    └───┴────────────────────────► 
        Operational Intensity
        (FLOPS/Byte)

If below line: Memory bottleneck
If on line: Compute bottleneck

Example Analysis:
┌────────────────────────────────────┐
│ Matrix Multiply:                   │
│   Ops: 2N³ FLOPs                   │
│   Data: 3N² bytes                  │
│   Intensity: 2N³/(3N²) = 2N/3      │
│                                    │
│   For N=1000: 667 FLOPS/byte       │
│   → Compute bound ✓                │
│                                    │
│   For N=10: 6.7 FLOPS/byte         │
│   → Memory bound ✗                 │
└────────────────────────────────────┘

Common Bottlenecks:

1. Memory Bandwidth:
   Symptom: High cache miss rate
   Solution: Improve locality, prefetch

2. Memory Latency:
   Symptom: Long stall cycles
   Solution: Increase parallelism, hide latency

3. Branch Mispredictions:
   Symptom: High bad speculation
   Solution: Reduce branches, improve predictability

4. Port Contention:
   Symptom: Execution unit stalls
   Solution: Better instruction mix

5. False Sharing:
   Symptom: High coherence traffic
   Solution: Align data, pad structures
```

## 9.6 Optimization Techniques

### 9.6.1 Compiler Optimizations

```
Optimization Levels:

-O0: No optimization
  - Fast compilation
  - Easy debugging
  - Slow execution

-O1: Basic optimization
  - Some improvements
  - Still debuggable

-O2: Moderate optimization (default)
  - Good performance
  - Reasonable compile time

-O3: Aggressive optimization
  - Best performance (usually)
  - Longer compile time
  - May increase code size

-Os: Optimize for size
  - Smaller binaries
  - Good for embedded

Common Optimizations:

1. Constant Folding:
   Before: x = 2 * 3 * y;
   After:  x = 6 * y;

2. Dead Code Elimination:
   Before: x = 5; x = 10; return x;
   After:  x = 10; return x;

3. Common Subexpression:
   Before: a = b * c + d;
           e = b * c + f;
   After:  temp = b * c;
           a = temp + d;
           e = temp + f;

4. Loop Unrolling:
   Before:
   for (i = 0; i < 100; i++)
       a[i] = b[i] + c[i];
   
   After:
   for (i = 0; i < 100; i += 4) {
       a[i]   = b[i]   + c[i];
       a[i+1] = b[i+1] + c[i+1];
       a[i+2] = b[i+2] + c[i+2];
       a[i+3] = b[i+3] + c[i+3];
   }
   
   Benefits:
   + Fewer loop overhead instructions
   + More instruction-level parallelism
   + Better scheduling opportunities

5. Loop Invariant Code Motion:
   Before:
   for (i = 0; i < n; i++)
       a[i] = x * y + b[i];
   
   After:
   temp = x * y;
   for (i = 0; i < n; i++)
       a[i] = temp + b[i];

6. Inlining:
   Before:
   int square(int x) { return x * x; }
   y = square(5);
   
   After:
   y = 5 * 5;
   
   Trade-off: Code size vs function call overhead

7. Vectorization (Auto-SIMD):
   Before:
   for (i = 0; i < n; i++)
       c[i] = a[i] + b[i];
   
   After (AVX2):
   for (i = 0; i < n; i += 8)
       _mm256_store(c+i, 
           _mm256_add(
               _mm256_load(a+i),
               _mm256_load(b+i)));
   
   8× faster with AVX2!

Profile-Guided Optimization (PGO):
┌────────────────────────────────────┐
│ 1. Compile with instrumentation    │
│    gcc -fprofile-generate          │
│                                    │
│ 2. Run with typical workload       │
│    ./program < input.txt           │
│    (Generates profile data)        │
│                                    │
│ 3. Recompile with profile          │
│    gcc -fprofile-use               │
│                                    │
│ Benefits:                          │
│   - Better inlining decisions      │
│   - Optimized branch layout        │
│   - Dead code removal              │
│   - 10-30% improvement typical     │
└────────────────────────────────────┘
```

### 9.6.2 Algorithm-Level Optimization

```
Choose the Right Algorithm:

Example: Sorting
┌──────────────────┬──────────────┬─────────┐
│   Algorithm      │ Complexity   │ Use Case│
├──────────────────┼──────────────┼─────────┤
│ Bubble Sort      │ O(n²)        │ Never   │
│ Quick Sort       │ O(n log n)   │ General │
│ Merge Sort       │ O(n log n)   │ Stable  │
│ Radix Sort       │ O(kn)        │ Integers│
│ Counting Sort    │ O(n+k)       │ Small k │
└──────────────────┴──────────────┴─────────┘

Example: 1M elements
  Bubble: 1M² = 1 trillion ops
  Quick:  1M × log(1M) = 20M ops
  Speedup: 50,000×!

Data Structure Choice:

Search for element:
┌────────────────┬────────────┬──────────┐
│ Data Structure │  Search    │  Insert  │
├────────────────┼────────────┼──────────┤
│ Array          │   O(n)     │   O(n)   │
│ Sorted Array   │  O(log n)  │   O(n)   │
│ Linked List    │   O(n)     │   O(1)   │
│ Hash Table     │   O(1)     │   O(1)   │
│ Binary Tree    │  O(log n)  │  O(log n)│
└────────────────┴────────────┴──────────┘

For frequent lookups: Hash table!

Cache-Aware Algorithms:

Matrix Multiply (naive):
for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
        for (k = 0; k < n; k++)
            C[i][j] += A[i][k] * B[k][j];
                                    ↑
                       Poor spatial locality!

Blocked (Tiled) Version:
for (ii = 0; ii < n; ii += B)
    for (jj = 0; jj < n; jj += B)
        for (kk = 0; kk < n; kk += B)
            for (i = ii; i < ii+B; i++)
                for (j = jj; j < jj+B; j++)
                    for (k = kk; k < kk+B; k++)
                        C[i][j] += A[i][k] * B[k][j];

Block size B chosen to fit in cache
Speedup: 2-5× typical

Parallelization:

Amdahl's Law Analysis:
┌────────────────────────────────────┐
│ Profile shows:                     │
│   Loop 1: 60% of time              │
│   Loop 2: 30% of time              │
│   Rest:   10% of time              │
│                                    │
│ Parallelize loops:                 │
│   P = 0.9 (90% parallelizable)     │
│   N = 16 cores                     │
│                                    │
│ Speedup = 1/(0.1 + 0.9/16)         │
│         = 1/0.156                  │
│         = 6.4×                     │
│                                    │
│ On 16 cores: only 6.4× speedup!    │
│ 10% serial code is bottleneck      │
└────────────────────────────────────┘
```

### 9.6.3 Memory Optimization

```
1. Improve Spatial Locality:

   Bad:
   struct {
       char flag1;    // 1 byte
       int data1;     // 4 bytes (padded)
       char flag2;    // 1 byte
       int data2;     // 4 bytes (padded)
   };  // Total: 16 bytes
   
   Good:
   struct {
       int data1;     // 4 bytes
       int data2;     // 4 bytes
       char flag1;    // 1 byte
       char flag2;    // 1 byte
   };  // Total: 12 bytes
   
   25% memory savings + better cache usage!

2. Improve Temporal Locality:

   Bad:
   // Phase 1: Process all A
   for (i = 0; i < n; i++)
       process_A(data[i].A);
   
   // Phase 2: Process all B (A evicted from cache)
   for (i = 0; i < n; i++)
       process_B(data[i].B);
   
   Good:
   // Process A and B together
   for (i = 0; i < n; i++) {
       process_A(data[i].A);
       process_B(data[i].B);
   }

3. Reduce Cache Conflicts:

   Power-of-2 array sizes can cause conflicts
   
   // May conflict in direct-mapped cache
   float a[1024], b[1024], c[1024];
   
   // Better: Pad to avoid stride conflicts
   float a[1024], pad1[16];
   float b[1024], pad2[16];
   float c[1024];

4. Prefetching:

   Software prefetch:
   for (i = 0; i < n; i++) {
       __builtin_prefetch(&data[i+8]);
       process(data[i]);
   }
   
   Prefetch 8 iterations ahead
   Hides memory latency

5. Alignment:

   // Misaligned: multiple cache lines
   char buffer[100];
   int* p = (int*)(&buffer[1]);
   
   // Aligned: single cache line
   alignas(64) int data[16];

6. False Sharing Avoidance:

   Bad:
   struct {
       int counter1;  // Thread 1 uses
       int counter2;  // Thread 2 uses
   } shared;
   
   Both in same cache line → ping-pong!
   
   Good:
   struct {
       alignas(64) int counter1;
       alignas(64) int counter2;
   } shared;
   
   Separate cache lines → no false sharing

Performance Impact:
┌────────────────────┬──────────┬──────────┐
│   Technique        │ Speedup  │ Effort   │
├────────────────────┼──────────┼──────────┤
│ Data alignment     │ 1.1-1.5× │   Low    │
│ Locality improve   │ 1.5-3×   │  Medium  │
│ Blocking/tiling    │ 2-5×     │  Medium  │
│ Prefetching        │ 1.2-2×   │   Low    │
│ False share avoid  │ 2-10×    │   Low    │
└────────────────────┴──────────┴──────────┘
```

## 9.7 Case Studies

### 9.7.1 Matrix Multiplication Optimization

```
Problem: Multiply two N×N matrices

Naive Implementation:
┌────────────────────────────────────┐
│ for (i = 0; i < n; i++)            │
│   for (j = 0; j < n; j++)          │
│     for (k = 0; k < n; k++)        │
│       C[i][j] += A[i][k] * B[k][j];│
│                                    │
│ N=1000: 2 billion ops              │
│ Time: 8.5 seconds                  │
│ Performance: 0.24 GFLOPS           │
└────────────────────────────────────┘

Optimization 1: Transpose B
┌────────────────────────────────────┐
│ // Transpose B for better locality │
│ for (i = 0; i < n; i++)            │
│   for (j = 0; j < n; j++)          │
│     BT[j][i] = B[i][j];            │
│                                    │
│ for (i = 0; i < n; i++)            │
│   for (j = 0; j < n; j++)          │
│     for (k = 0; k < n; k++)        │
│       C[i][j] += A[i][k] * BT[j][k];│
│                                    │
│ Time: 2.1 seconds                  │
│ Speedup: 4×                        │
│ Performance: 0.95 GFLOPS           │
└────────────────────────────────────┘

Optimization 2: Blocking/Tiling
┌────────────────────────────────────┐
│ #define BLOCK 64                   │
│                                    │
│ for (ii = 0; ii < n; ii += BLOCK) │
│   for (jj = 0; jj < n; jj += BLOCK)│
│     for (kk = 0; kk < n; kk += BLOCK)│
│       for (i = ii; i < ii+BLOCK; i++)│
│         for (j = jj; j < jj+BLOCK; j++)│
│           for (k = kk; k < kk+BLOCK; k++)│
│             C[i][j] += A[i][k]*B[k][j];│
│                                    │
│ Time: 0.8 seconds                  │
│ Speedup: 10.6× vs naive            │
│ Performance: 2.5 GFLOPS            │
└────────────────────────────────────┘

Optimization 3: SIMD (AVX2)
┌────────────────────────────────────┐
│ // Process 8 floats at once        │
│ for (i = 0; i < n; i++)            │
│   for (j = 0; j < n; j += 8) {     │
│     __m256 sum = _mm256_setzero(); │
│     for (k = 0; k < n; k++)        │
│       sum = _mm256_fmadd(          │
│           _mm256_set1(A[i][k]),    │
│           _mm256_load(&B[k][j]),   │
│           sum);                    │
│     _mm256_store(&C[i][j], sum);   │
│   }                                │
│                                    │
│ Time: 0.15 seconds                 │
│ Speedup: 56× vs naive              │
│ Performance: 13.3 GFLOPS           │
└────────────────────────────────────┘

Optimization 4: Parallel (OpenMP)
┌────────────────────────────────────┐
│ #pragma omp parallel for           │
│ for (i = 0; i < n; i++)            │
│   // ... blocked SIMD code         │
│                                    │
│ 8 cores × 13.3 GFLOPS/core         │
│                                    │
│ Time: 0.025 seconds                │
│ Speedup: 340× vs naive             │
│ Performance: 80 GFLOPS             │
└────────────────────────────────────┘

Final: BLAS Library (Intel MKL)
┌────────────────────────────────────┐
│ cblas_sgemm(..., A, B, C);         │
│                                    │
│ Highly optimized:                  │
│ - Multi-level blocking             │
│ - AVX-512 instructions             │
│ - Cache prefetching                │
│ - Assembly optimization            │
│                                    │
│ Time: 0.008 seconds                │
│ Speedup: 1062× vs naive            │
│ Performance: 250 GFLOPS            │
│                                    │
│ Near peak theoretical performance! │
└────────────────────────────────────┘

Summary:
┌────────────────┬──────────┬──────────┐
│  Version       │   Time   │ Speedup  │
├────────────────┼──────────┼──────────┤
│ Naive          │  8.50s   │   1.0×   │
│ Transpose      │  2.10s   │   4.0×   │
│ Blocked        │  0.80s   │  10.6×   │
│ SIMD           │  0.15s   │  56.7×   │
│ Parallel       │  0.025s  │ 340×     │
│ MKL (optimal)  │  0.008s  │ 1062×    │
└────────────────┴──────────┴──────────┘

Lessons:
1. Algorithm matters most (transpose)
2. Cache locality crucial (blocking)
3. Use SIMD when possible
4. Parallelize
5. Use optimized libraries!
```

### 9.7.2 Image Processing Optimization

```
Problem: Apply filter to 4K image (3840×2160 pixels)

Naive Convolution (3×3 filter):
┌────────────────────────────────────┐
│ for (y = 1; y < height-1; y++)     │
│   for (x = 1; x < width-1; x++) {  │
│     sum = 0;                       │
│     for (fy = -1; fy <= 1; fy++)   │
│       for (fx = -1; fx <= 1; fx++) │
│         sum += image[y+fy][x+fx] * │
│                filter[fy+1][fx+1]; │
│     output[y][x] = sum;            │
│   }                                │
│                                    │
│ 8.3M pixels × 9 ops = 75M ops      │
│ Time: 250 ms                       │
└────────────────────────────────────┘

Optimization 1: Separable Filter
┌────────────────────────────────────┐
│ // Many filters are separable:    │
│ // 2D = 1D horizontal × 1D vertical│
│                                    │
│ // Horizontal pass                 │
│ for (y = 0; y < height; y++)       │
│   for (x = 1; x < width-1; x++)    │
│     temp[y][x] = image[y][x-1] +   │
│                  2*image[y][x] +   │
│                  image[y][x+1];    │
│                                    │
│ // Vertical pass                   │
│ for (y = 1; y < height-1; y++)     │
│   for (x = 0; x < width; x++)      │
│     output[y][x] = temp[y-1][x] +  │
│                    2*temp[y][x] +  │
│                    temp[y+1][x];   │
│                                    │
│ 8.3M × 3 × 2 = 50M ops (vs 75M)    │
│ Time: 85 ms                        │
│ Speedup: 2.9×                      │
└────────────────────────────────────┘

Optimization 2: SIMD (Process 8 pixels)
┌────────────────────────────────────┐
│ for (y = 0; y < height; y++)       │
│   for (x = 0; x < width; x += 8) { │
│     __m256i prev = load(x-1);      │
│     __m256i curr = load(x);        │
│     __m256i next = load(x+1);      │
│     __m256i result =               │
│         add(add(prev, slli(curr,1)),│
│             next);                 │
│     store(x, result);              │
│   }                                │
│                                    │
│ Time: 12 ms                        │
│ Speedup: 20.8×                     │
└────────────────────────────────────┘

Optimization 3: Multi-threading
┌────────────────────────────────────┐
│ #pragma omp parallel for           │
│ for (y = 0; y < height; y++) {     │
│   // SIMD code for row y           │
│ }                                  │
│                                    │
│ 8 threads                          │
│ Time: 2 ms                         │
│ Speedup: 125×                      │
└────────────────────────────────────┘

Optimization 4: GPU (CUDA)
┌────────────────────────────────────┐
│ __global__ void filter(            │
│     uint8_t* in, uint8_t* out) {   │
│   int x = blockIdx.x*blockDim.x +  │
│           threadIdx.x;             │
│   int y = blockIdx.y*blockDim.y +  │
│           threadIdx.y;             │
│                                    │
│   // Load to shared memory         │
│   // Compute filter                │
│   // Write result                  │
│ }                                  │
│                                    │
│ 8.3M pixels in parallel            │
│ Time: 0.3 ms                       │
│ Speedup: 833×                      │
└────────────────────────────────────┘

Real-world: Use NPU
┌────────────────────────────────────┐
│ // Mobile NPU optimized            │
│ npu_convolution_2d(                │
│     input, filter, output);        │
│                                    │
│ Hardware acceleration              │
│ Time: 0.15 ms                      │
│ Speedup: 1666×                     │
│ Power: 50 mW (vs 5W on CPU)        │
└────────────────────────────────────┘
```

## 9.8 Performance Checklist

```
Quick Optimization Checklist:

Algorithm Level:
☐ Correct algorithm complexity?
☐ Better algorithm available?
☐ Avoid redundant computation?
☐ Appropriate data structures?

Memory:
☐ Good spatial locality?
☐ Good temporal locality?
☐ Minimize cache misses?
☐ Data aligned?
☐ Avoid false sharing?
☐ Use prefetching?

CPU:
☐ Compiler optimizations enabled?
☐ Hot loops identified?
☐ Branches predictable?
☐ ILP opportunities?
☐ Use SIMD?
☐ Avoid division (expensive)?

Parallelization:
☐ Amdahl's Law analysis done?
☐ Parallelize hot sections?
☐ Minimize synchronization?
☐ Load balanced?
☐ NUMA-aware?

Tools:
☐ Profiled with perf/VTune?
☐ Examined assembly?
☐ Benchmarked properly?
☐ Multiple test cases?
☐ Validated correctness?

Modern HW:
☐ Consider GPU?
☐ Consider accelerators (NPU/TPU)?
☐ Heterogeneous execution?
☐ Power-aware?
```

---

**Key Takeaways:**
1. Measure before optimizing (profile!)
2. Focus on bottlenecks (Amdahl's Law)
3. Algorithm choice is most important
4. Memory hierarchy awareness crucial
5. Use appropriate tools (profilers, counters)
6. Consider modern hardware (SIMD, GPU, accelerators)
7. Validate correctness after optimization
8. Document performance improvements

**Previous:** [Heterogeneous Computing](./08-heterogeneous-computing.md) | **Next:** [Quick Reference](./appendix-reference.md)


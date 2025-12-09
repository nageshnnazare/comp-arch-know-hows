# Appendix A: Quick Reference & Glossary

## A.1 Key Formulas

### Performance Metrics

```
CPU Time = IC × CPI × Clock_Cycle_Time

Where:
  IC  = Instruction Count
  CPI = Cycles Per Instruction
  Clock_Cycle_Time = 1 / Frequency

MIPS = Clock_Frequency (MHz) / CPI

IPC (Instructions Per Cycle) = 1 / CPI

Speedup = Time_old / Time_new

Efficiency = Speedup / Number_of_Processors
```

### Amdahl's Law

```
Speedup = 1 / ((1 - P) + P/S)

Where:
  P = Fraction enhanced
  S = Speedup of enhanced portion

Maximum Speedup = 1 / (1 - P)
(When S → ∞)
```

### Memory Hierarchy

```
AMAT (Average Memory Access Time):
AMAT = Hit_Time + Miss_Rate × Miss_Penalty

Multi-level:
AMAT = T_L1 + MR_L1 × (T_L2 + MR_L2 × (T_L3 + MR_L3 × T_mem))

Effective Bandwidth:
BW_eff = BW_peak × Hit_Rate

Miss Rate = Misses / Total_Accesses

Hit Rate = 1 - Miss_Rate
```

### Cache Size

```
Cache_Size = Number_of_Sets × Associativity × Block_Size

Number of Bits for Address:
  Tag bits    = Address_bits - Index_bits - Offset_bits
  Index bits  = log₂(Number_of_Sets)
  Offset bits = log₂(Block_Size)

Example: 32-bit address, 32 KB cache, 64B blocks, 4-way
  Sets = 32KB / (4 × 64B) = 128
  Index = log₂(128) = 7 bits
  Offset = log₂(64) = 6 bits
  Tag = 32 - 7 - 6 = 19 bits
```

### Virtual Memory

```
Page Table Size = (Virtual_Address_Space / Page_Size) × Entry_Size

Example: 32-bit VA, 4KB pages, 4B entries
  Size = (2³² / 2¹²) × 4 = 2²⁰ × 4 = 4 MB

TLB Effective Access Time:
T_eff = T_TLB + (1 - Hit_TLB) × T_PageTable + T_memory
```

### Power

```
Dynamic Power = α × C × V² × f

Where:
  α = Activity factor
  C = Capacitance
  V = Voltage
  f = Frequency

Total Power = Dynamic_Power + Static_Power

Energy = Power × Time

Energy Efficiency = Operations / Energy (ops/Joule)
```

### Bandwidth & Latency

```
Bandwidth = Data_Size / Transfer_Time

Latency = Time_to_First_Byte

Throughput = Items / Time

Little's Law:
Throughput = Concurrency / Latency
```

## A.2 Number Conversions

### Quick Conversion Table

```
┌─────────┬──────────┬───────┬──────────────┐
│ Decimal │  Binary  │  Hex  │    Octal     │
├─────────┼──────────┼───────┼──────────────┤
│    0    │   0000   │   0   │      0       │
│    1    │   0001   │   1   │      1       │
│    2    │   0010   │   2   │      2       │
│    3    │   0011   │   3   │      3       │
│    4    │   0100   │   4   │      4       │
│    5    │   0101   │   5   │      5       │
│    6    │   0110   │   6   │      6       │
│    7    │   0111   │   7   │      7       │
│    8    │   1000   │   8   │     10       │
│    9    │   1001   │   9   │     11       │
│   10    │   1010   │   A   │     12       │
│   11    │   1011   │   B   │     13       │
│   12    │   1100   │   C   │     14       │
│   13    │   1101   │   D   │     15       │
│   14    │   1110   │   E   │     16       │
│   15    │   1111   │   F   │     17       │
└─────────┴──────────┴───────┴──────────────┘

Powers of 2:
2⁰  = 1          2¹⁰ = 1,024        (1 KB)
2¹  = 2          2²⁰ = 1,048,576    (1 MB)
2²  = 4          2³⁰ = 1,073,741,824 (1 GB)
2³  = 8          2⁴⁰ = ~1 trillion   (1 TB)
2⁴  = 16
2⁵  = 32
2⁶  = 64
2⁷  = 128
2⁸  = 256
2⁹  = 512
```

### Two's Complement (8-bit)

```
┌──────────┬──────────┬──────────┐
│ Decimal  │  Binary  │   Hex    │
├──────────┼──────────┼──────────┤
│   127    │ 01111111 │   0x7F   │ ← Max positive
│   ...    │   ...    │   ...    │
│     2    │ 00000010 │   0x02   │
│     1    │ 00000001 │   0x01   │
│     0    │ 00000000 │   0x00   │
│    -1    │ 11111111 │   0xFF   │
│    -2    │ 11111110 │   0xFE   │
│   ...    │   ...    │   ...    │
│  -128    │ 10000000 │   0x80   │ ← Min negative
└──────────┴──────────┴──────────┘

Range: -2ⁿ⁻¹ to 2ⁿ⁻¹ - 1
8-bit: -128 to 127
16-bit: -32,768 to 32,767
32-bit: -2,147,483,648 to 2,147,483,647
```

## A.3 Memory & Storage Units

### Size Conversions

```
Binary (IEC standard):
1 KiB = 2¹⁰ bytes  = 1,024 bytes
1 MiB = 2²⁰ bytes  = 1,048,576 bytes
1 GiB = 2³⁰ bytes  = 1,073,741,824 bytes
1 TiB = 2⁴⁰ bytes  = 1,099,511,627,776 bytes
1 PiB = 2⁵⁰ bytes

Decimal (SI standard, used in storage):
1 KB = 10³ bytes   = 1,000 bytes
1 MB = 10⁶ bytes   = 1,000,000 bytes
1 GB = 10⁹ bytes   = 1,000,000,000 bytes
1 TB = 10¹² bytes  = 1,000,000,000,000 bytes
1 PB = 10¹⁵ bytes

Note: Hard drive manufacturers use decimal,
      operating systems use binary!
      
      500 GB HDD = 500,000,000,000 bytes
                 = ~465.66 GiB
```

### Typical Latencies (Order of Magnitude)

```
┌──────────────────────────┬────────────┬──────────┐
│       Operation          │  Latency   │  Cycles  │
│                          │            │ @3GHz    │
├──────────────────────────┼────────────┼──────────┤
│ Register access          │   0.3 ns   │    1     │
│ L1 cache hit             │   1 ns     │    3     │
│ L2 cache hit             │   3 ns     │    10    │
│ L3 cache hit             │  10 ns     │    30    │
│ Main memory (DRAM)       │  100 ns    │   300    │
│ NVMe SSD read (random)   │  100 µs    │ 300,000  │
│ SATA SSD read            │  150 µs    │ 450,000  │
│ HDD seek + rotate        │  10 ms     │ 30M      │
│ Network: same datacenter │  500 µs    │  1.5M    │
│ Network: coast-to-coast  │  50 ms     │ 150M     │
└──────────────────────────┴────────────┴──────────┘

Visualization (log scale):
Register   ●
L1         ●
L2          ●
L3           ●●
RAM            ●●●●
SSD                      ●●●●●●●
HDD                                  ●●●●●●●●●●●●●
Network                                      ●●●●●●●●●●
```

### Bandwidth Reference

```
┌────────────────────────┬──────────────┬───────────┐
│      Technology        │  Bandwidth   │  Latency  │
├────────────────────────┼──────────────┼───────────┤
│ DDR4-3200 (dual ch)    │   51 GB/s    │   ~80ns   │
│ DDR5-4800 (dual ch)    │   77 GB/s    │   ~70ns   │
│ PCIe 3.0 x16           │   32 GB/s    │   ~1µs    │
│ PCIe 4.0 x16           │   64 GB/s    │   ~1µs    │
│ PCIe 5.0 x16           │  128 GB/s    │   ~1µs    │
│ NVMe SSD (PCIe 4.0)    │  7-8 GB/s    │  ~100µs   │
│ SATA III SSD           │  0.6 GB/s    │  ~150µs   │
│ USB 3.2 Gen 2          │  1.25 GB/s   │   ~ms     │
│ 10 Gigabit Ethernet    │  1.25 GB/s   │   ~ms     │
│ Thunderbolt 4          │  5 GB/s      │   ~µs     │
│ GPU HBM2               │  900 GB/s    │   ~100ns  │
│ GPU HBM3               │ 2.4 TB/s     │   ~100ns  │
└────────────────────────┴──────────────┴───────────┘
```

## A.4 Instruction Set Reference

### Common x86-64 Instructions

```
Data Movement:
  MOV  dest, src     ; Move
  PUSH src           ; Push onto stack
  POP  dest          ; Pop from stack
  LEA  dest, src     ; Load effective address
  XCHG a, b          ; Exchange

Arithmetic:
  ADD  dest, src     ; dest = dest + src
  SUB  dest, src     ; dest = dest - src
  MUL  src           ; Unsigned multiply
  IMUL src           ; Signed multiply
  DIV  src           ; Unsigned divide
  IDIV src           ; Signed divide
  INC  dest          ; Increment
  DEC  dest          ; Decrement
  NEG  dest          ; Negate

Logical:
  AND  dest, src     ; Bitwise AND
  OR   dest, src     ; Bitwise OR
  XOR  dest, src     ; Bitwise XOR
  NOT  dest          ; Bitwise NOT
  SHL  dest, count   ; Shift left
  SHR  dest, count   ; Shift right
  SAR  dest, count   ; Arithmetic shift right
  ROL  dest, count   ; Rotate left
  ROR  dest, count   ; Rotate right

Comparison:
  CMP  op1, op2      ; Compare (subtract, set flags)
  TEST op1, op2      ; Bitwise test (AND, set flags)

Control Flow:
  JMP  target        ; Unconditional jump
  JE   target        ; Jump if equal (ZF=1)
  JNE  target        ; Jump if not equal (ZF=0)
  JG   target        ; Jump if greater (signed)
  JL   target        ; Jump if less (signed)
  JA   target        ; Jump if above (unsigned)
  JB   target        ; Jump if below (unsigned)
  CALL target        ; Call function
  RET                ; Return from function

SIMD (AVX2):
  VMOVAPS  xmm, mem  ; Move aligned packed single
  VADDPS   xmm, xmm  ; Add packed single
  VMULPS   xmm, xmm  ; Multiply packed single
  VFMADD   xmm, xmm  ; Fused multiply-add
```

### ARM Instructions (Common)

```
Data Processing:
  ADD  Rd, Rn, Rm    ; Rd = Rn + Rm
  SUB  Rd, Rn, Rm    ; Rd = Rn - Rm
  MUL  Rd, Rn, Rm    ; Rd = Rn × Rm
  AND  Rd, Rn, Rm    ; Rd = Rn & Rm
  ORR  Rd, Rn, Rm    ; Rd = Rn | Rm
  EOR  Rd, Rn, Rm    ; Rd = Rn ^ Rm
  MOV  Rd, operand   ; Move
  MVN  Rd, operand   ; Move NOT

Memory:
  LDR  Rt, [Rn]      ; Load register
  STR  Rt, [Rn]      ; Store register
  LDRB Rt, [Rn]      ; Load byte
  STRB Rt, [Rn]      ; Store byte
  LDM  Rn, {reglist} ; Load multiple
  STM  Rn, {reglist} ; Store multiple

Branch:
  B    label         ; Branch
  BL   label         ; Branch with link (call)
  BX   Rm            ; Branch and exchange
  BEQ  label         ; Branch if equal
  BNE  label         ; Branch if not equal
  BGT  label         ; Branch if greater
  BLT  label         ; Branch if less

Compare:
  CMP  Rn, operand   ; Compare
  CMN  Rn, operand   ; Compare negative
  TST  Rn, operand   ; Test bits
```

## A.5 Common Acronyms

```
┌──────────┬───────────────────────────────────────┐
│ Acronym  │             Full Name                 │
├──────────┼───────────────────────────────────────┤
│ ALU      │ Arithmetic Logic Unit                 │
│ AMAT     │ Average Memory Access Time            │
│ APU      │ Accelerated Processing Unit           │
│ ASIC     │ Application-Specific Integrated Circ. │
│ AVX      │ Advanced Vector Extensions            │
│ BHT      │ Branch History Table                  │
│ BIOS     │ Basic Input/Output System             │
│ BTB      │ Branch Target Buffer                  │
│ CAM      │ Content Addressable Memory            │
│ CAS      │ Column Address Strobe                 │
│ CISC     │ Complex Instruction Set Computer      │
│ CPI      │ Cycles Per Instruction                │
│ CPU      │ Central Processing Unit               │
│ CU       │ Control Unit                          │
│ DMA      │ Direct Memory Access                  │
│ DRAM     │ Dynamic Random Access Memory          │
│ DSP      │ Digital Signal Processor              │
│ ECC      │ Error Correcting Code                 │
│ EPIC     │ Explicitly Parallel Instruction Comp. │
│ FLOPS    │ Floating-Point Operations Per Second  │
│ FMA      │ Fused Multiply-Add                    │
│ FPGA     │ Field-Programmable Gate Array         │
│ FPU      │ Floating-Point Unit                   │
│ FSM      │ Finite State Machine                  │
│ GPU      │ Graphics Processing Unit              │
│ HBM      │ High Bandwidth Memory                 │
│ HDD      │ Hard Disk Drive                       │
│ HSA      │ Heterogeneous System Architecture     │
│ HTM      │ Hardware Transactional Memory         │
│ IC       │ Instruction Count                     │
│ ILP      │ Instruction-Level Parallelism         │
│ IPC      │ Instructions Per Cycle                │
│ IR       │ Instruction Register                  │
│ ISA      │ Instruction Set Architecture          │
│ ISR      │ Interrupt Service Routine             │
│ LRU      │ Least Recently Used                   │
│ LSB      │ Least Significant Bit                 │
│ MAR      │ Memory Address Register               │
│ MDR      │ Memory Data Register                  │
│ MESI     │ Modified-Exclusive-Shared-Invalid     │
│ MIMD     │ Multiple Instruction, Multiple Data   │
│ MIPS     │ Million Instructions Per Second       │
│ MMU      │ Memory Management Unit                │
│ MOESI    │ Modified-Owned-Exclusive-Shared-Inv.  │
│ MSB      │ Most Significant Bit                  │
│ MTBF     │ Mean Time Between Failures            │
│ NoC      │ Network-on-Chip                       │
│ NPU      │ Neural Processing Unit                │
│ NUMA     │ Non-Uniform Memory Access             │
│ OOO      │ Out-of-Order (execution)              │
│ OS       │ Operating System                      │
│ PC       │ Program Counter                       │
│ PCIe     │ Peripheral Component Interconnect Exp.│
│ PGO      │ Profile-Guided Optimization           │
│ PIM      │ Processing-In-Memory                  │
│ PSW      │ Program Status Word                   │
│ RAID     │ Redundant Array of Independent Disks  │
│ RAM      │ Random Access Memory                  │
│ RAS      │ Row Address Strobe                    │
│ RISC     │ Reduced Instruction Set Computer      │
│ ROB      │ Reorder Buffer                        │
│ ROM      │ Read-Only Memory                      │
│ RTL      │ Register Transfer Level               │
│ SIMD     │ Single Instruction, Multiple Data     │
│ SIMT     │ Single Instruction, Multiple Threads  │
│ SMT      │ Simultaneous Multithreading           │
│ SoC      │ System-on-Chip                        │
│ SP       │ Stack Pointer                         │
│ SPEC     │ Standard Performance Evaluation Corp. │
│ SRAM     │ Static Random Access Memory           │
│ SSD      │ Solid State Drive                     │
│ TDP      │ Thermal Design Power                  │
│ TLB      │ Translation Lookaside Buffer          │
│ TPU      │ Tensor Processing Unit                │
│ VLIW     │ Very Long Instruction Word            │
│ VRAM     │ Video Random Access Memory            │
└──────────┴───────────────────────────────────────┘
```

## A.6 Typical CPU Specifications

### Desktop Processors (2023-2024)

```
Intel Core i9-13900K:
┌───────────────────────────────────────┐
│ Cores:        8P + 16E = 24 cores     │
│ Threads:      32 (with HT)            │
│ Base Clock:   3.0 GHz (P) / 2.2 (E)   │
│ Boost Clock:  5.8 GHz (P) / 4.3 (E)   │
│ L3 Cache:     36 MB                   │
│ TDP:          125W base, 253W turbo   │
│ Memory:       DDR5-5600, DDR4-3200    │
│ PCIe:         5.0 (16 lanes)          │
└───────────────────────────────────────┘

AMD Ryzen 9 7950X:
┌───────────────────────────────────────┐
│ Cores:        16 (Zen 4)              │
│ Threads:      32                      │
│ Base Clock:   4.5 GHz                 │
│ Boost Clock:  5.7 GHz                 │
│ L3 Cache:     64 MB                   │
│ TDP:          170W                    │
│ Memory:       DDR5-5200               │
│ PCIe:         5.0 (28 lanes)          │
└───────────────────────────────────────┘

Apple M2 Max:
┌───────────────────────────────────────┐
│ Cores:        8P + 4E = 12 cores      │
│ GPU Cores:    38                      │
│ Neural Eng:   16 cores (15.8 TOPS)    │
│ Memory:       Up to 96 GB unified     │
│ Bandwidth:    400 GB/s                │
│ Process:      5nm                     │
│ TDP:          ~40-50W                 │
└───────────────────────────────────────┘
```

### Server Processors

```
Intel Xeon Platinum 8480+:
┌───────────────────────────────────────┐
│ Cores:        56                      │
│ Threads:      112                     │
│ Base Clock:   2.0 GHz                 │
│ Boost Clock:  3.8 GHz                 │
│ L3 Cache:     105 MB                  │
│ TDP:          350W                    │
│ Memory:       DDR5-4800, 8 channels   │
│ Sockets:      Up to 8-socket systems  │
└───────────────────────────────────────┘

AMD EPYC 9654:
┌───────────────────────────────────────┐
│ Cores:        96 (Zen 4)              │
│ Threads:      192                     │
│ Base Clock:   2.4 GHz                 │
│ Boost Clock:  3.7 GHz                 │
│ L3 Cache:     384 MB                  │
│ TDP:          360W                    │
│ Memory:       DDR5-4800, 12 channels  │
│ PCIe:         5.0 (128 lanes)         │
└───────────────────────────────────────┘
```

### Mobile Processors

```
Apple A17 Pro:
┌───────────────────────────────────────┐
│ Cores:        2P + 4E = 6 cores       │
│ GPU:          6 cores                 │
│ Neural Eng:   16 cores (35 TOPS)      │
│ Process:      3nm                     │
│ Transistors:  19 billion              │
│ TDP:          ~5-8W                   │
└───────────────────────────────────────┘

Qualcomm Snapdragon 8 Gen 3:
┌───────────────────────────────────────┐
│ CPU:          1×3.3 + 3×3.2 + 2×3.0   │
│               + 2×2.3 GHz (8 cores)   │
│ GPU:          Adreno 750              │
│ AI Engine:    Hexagon NPU (73 TOPS)   │
│ Process:      4nm                     │
│ Modem:        5G (10 Gbps down)       │
└───────────────────────────────────────┘
```

## A.7 Cache Size Quick Reference

```
Typical Cache Sizes (2024):

Desktop/Laptop:
┌──────────┬──────────┬──────────┬──────────┐
│          │    L1    │    L2    │    L3    │
├──────────┼──────────┼──────────┼──────────┤
│ Intel    │ 32K I+D  │ 1-2 MB   │ 24-36 MB │
│          │ per core │ per core │  shared  │
├──────────┼──────────┼──────────┼──────────┤
│ AMD      │ 32K I+D  │ 1 MB     │ 32-64 MB │
│          │ per core │ per core │  shared  │
├──────────┼──────────┼──────────┼──────────┤
│ Apple M  │ 128-192K │ 12-16 MB │ Unified  │
│          │ per core │  shared  │   SLC    │
└──────────┴──────────┴──────────┴──────────┘

Server:
┌──────────┬──────────┬──────────┬──────────┐
│          │    L1    │    L2    │    L3    │
├──────────┼──────────┼──────────┼──────────┤
│ Xeon     │ 32K I+D  │ 1-2 MB   │ 105 MB   │
│          │ per core │ per core │  shared  │
├──────────┼──────────┼──────────┼──────────┤
│ EPYC     │ 32K I+D  │ 1 MB     │ 384 MB   │
│          │ per core │ per core │  shared  │
└──────────┴──────────┴──────────┴──────────┘

Mobile:
┌──────────┬──────────┬──────────┬──────────┐
│          │    L1    │    L2    │    L3    │
├──────────┼──────────┼──────────┼──────────┤
│ ARM High │ 64K I+D  │ 512K-1MB │ 4-8 MB   │
│ Perf     │ per core │ per core │  shared  │
├──────────┼──────────┼──────────┼──────────┤
│ ARM Eff  │ 32K I+D  │ 128-256K │ shared   │
│          │ per core │ per core │   L3     │
└──────────┴──────────┴──────────┴──────────┘

General Trends:
- L1: 32-192 KB per core (I+D split)
- L2: 256 KB - 2 MB per core
- L3: 1-4 MB per core (shared)
- Larger caches = lower miss rate but higher latency
```

## A.8 Benchmark Scores Reference

### SPEC CPU 2017 (Approximate)

```
Single-Thread Performance (SPECrate):
┌──────────────────────────┬─────────┬────────┐
│       Processor          │ SPECint │SPECfp  │
├──────────────────────────┼─────────┼────────┤
│ Intel i9-13900K (P-core) │   ~14   │  ~17   │
│ AMD Ryzen 9 7950X        │   ~13   │  ~16   │
│ Apple M2 Max             │   ~12   │  ~15   │
│ ARM Neoverse V2          │   ~10   │  ~12   │
└──────────────────────────┴─────────┴────────┘

Multi-Thread Performance (SPECrate):
┌──────────────────────────┬─────────┬────────┐
│       Processor          │ SPECint │SPECfp  │
├──────────────────────────┼─────────┼────────┤
│ AMD EPYC 9654 (96c)      │  ~500   │  ~650  │
│ Intel Xeon 8480+ (56c)   │  ~400   │  ~500  │
│ AMD Ryzen 9 7950X (16c)  │  ~150   │  ~180  │
└──────────────────────────┴─────────┴────────┘
```

### Memory Bandwidth (STREAM)

```
┌──────────────────────────┬─────────────┐
│       System             │  Triad GB/s │
├──────────────────────────┼─────────────┤
│ DDR5-5600 (dual channel) │    75-85    │
│ DDR5-4800 (8-channel)    │   350-400   │
│ HBM2e (GPU)              │   800-900   │
│ HBM3 (GPU)               │  2000-2400  │
└──────────────────────────┴─────────────┘
```

### AI Performance

```
┌─────────────────────┬──────────┬──────────┐
│     Accelerator     │INT8 TOPS │FP16 TFLOPS│
├─────────────────────┼──────────┼──────────┤
│ NVIDIA H100         │  2000    │   1000   │
│ NVIDIA A100         │  1248    │    312   │
│ Google TPU v4       │  275 (BF16 TFLOPS)  │
│ Apple M2 Neural Eng │   15.8   │    -     │
│ Qualcomm Hexagon    │   73     │    -     │
└─────────────────────┴──────────┴──────────┘
```

## A.9 Power Consumption Reference

```
Typical Power Draw:

Desktop CPU (Load):
┌────────────────────┬──────────┬──────────┐
│     Processor      │   Idle   │   Load   │
├────────────────────┼──────────┼──────────┤
│ Intel i9-13900K    │  30-50W  │ 250-300W │
│ AMD Ryzen 9 7950X  │  40-60W  │ 200-230W │
│ Apple M2 Max       │   5-10W  │  40-60W  │
└────────────────────┴──────────┴──────────┘

Laptop CPU:
┌────────────────────┬──────────┬──────────┐
│     Processor      │   Idle   │   Load   │
├────────────────────┼──────────┼──────────┤
│ Intel i9-13980HX   │  10-20W  │  100-155W│
│ AMD Ryzen 9 7945HX │  10-20W  │  80-120W │
│ Apple M2 Pro       │   3-5W   │  25-35W  │
└────────────────────┴──────────┴──────────┘

Mobile SoC:
┌────────────────────┬──────────┬──────────┐
│        SoC         │   Idle   │   Load   │
├────────────────────┼──────────┼──────────┤
│ Apple A17 Pro      │  0.5-1W  │   6-8W   │
│ Snapdragon 8 Gen 3 │  0.3-1W  │   5-10W  │
└────────────────────┴──────────┴──────────┘

GPU:
┌────────────────────┬──────────┬──────────┐
│        GPU         │   Idle   │   Load   │
├────────────────────┼──────────┼──────────┤
│ NVIDIA RTX 4090    │  20-30W  │ 450-500W │
│ AMD RX 7900 XTX    │  15-25W  │ 350-400W │
│ Mobile RTX 4090    │  10-15W  │ 150-175W │
└────────────────────┴──────────┴──────────┘

Server:
┌────────────────────┬──────────┬──────────┐
│     Processor      │   Idle   │   Load   │
├────────────────────┼──────────┼──────────┤
│ Intel Xeon 8480+   │  50-80W  │ 300-400W │
│ AMD EPYC 9654      │  60-100W │ 320-400W │
└────────────────────┴──────────┴──────────┘

Energy Efficiency (Representative):
Desktop CPU:   ~100 GFLOPS/Watt
Mobile CPU:    ~50-80 GFLOPS/Watt
GPU:           ~100-150 GFLOPS/Watt
TPU:           ~500-1000 GFLOPS/Watt (INT8)
```

## A.10 Common Debugging Techniques

```
Performance Debugging Workflow:

1. Profile:
   $ perf record -g ./program
   $ perf report
   
2. Check hardware counters:
   $ perf stat -e cycles,instructions,cache-misses ./program
   
3. View hotspots:
   $ perf annotate function_name
   
4. Memory profiling:
   $ valgrind --tool=massif ./program
   $ ms_print massif.out
   
5. Cache analysis:
   $ valgrind --tool=cachegrind ./program
   $ cg_annotate cachegrind.out
   
6. Check compiler output:
   $ gcc -O3 -S file.c  # View assembly
   $ objdump -d binary  # Disassemble
   
7. Intel VTune (if available):
   $ vtune -collect hotspots ./program
   
8. Check NUMA effects:
   $ numactl --hardware
   $ numastat -p PID

Common Issues & Solutions:

Issue: Low IPC (< 1.0)
→ Check: Branch mispredictions, cache misses
→ Solution: Improve branch predictability, data locality

Issue: High cache miss rate
→ Check: Access patterns, data structure layout
→ Solution: Blocking, prefetching, alignment

Issue: Low CPU utilization
→ Check: Memory bandwidth saturation, I/O waits
→ Solution: Reduce memory traffic, async I/O

Issue: Poor multi-core scaling
→ Check: Synchronization overhead, false sharing
→ Solution: Reduce locks, pad data structures

Issue: Thermal throttling
→ Check: CPU temperature, sustained load
→ Solution: Better cooling, reduce peak power
```

## A.11 Quick Architecture Comparison

```
┌──────────────┬──────────┬──────────┬──────────┐
│   Feature    │  x86-64  │   ARM    │  RISC-V  │
├──────────────┼──────────┼──────────┼──────────┤
│ Type         │   CISC   │   RISC   │   RISC   │
│ Instruction  │ Variable │  Fixed   │  Fixed   │
│   Length     │  1-15B   │  2/4B    │  2/4B    │
│ Registers    │   16     │  31/32   │  31/32   │
│ Addressing   │   Many   │   Few    │   Few    │
│   Modes      │          │          │          │
│ Condition    │  Yes     │  Flags   │   No     │
│   Codes      │          │ optional │  flags   │
│ Memory       │   Any    │Load/Store│Load/Store│
│   Access     │          │          │          │
│ SIMD         │ SSE/AVX  │  NEON    │   V ext  │
│ Dominant     │ Desktop/ │  Mobile/ │ Emerging │
│   Market     │  Server  │ Embedded │          │
└──────────────┴──────────┴──────────┴──────────┘
```

---

**How to Use This Reference:**
1. Quick lookup for formulas and conversions
2. Verify typical values for sanity checking
3. Reference for common acronyms
4. Baseline for performance comparison
5. Debugging starting point

**Tip:** Bookmark this page for quick access during development and analysis!

**Previous:** [Performance Analysis](./09-performance-analysis.md) | **Index:** [README](./README.md)


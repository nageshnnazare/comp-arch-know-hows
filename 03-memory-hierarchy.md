# Chapter 3: Memory Hierarchy

## 3.1 Memory Hierarchy Overview

The memory hierarchy is organized by speed, cost, and capacity.

```
┌──────────────────────────────────────────────────────┐
│              Memory Hierarchy Pyramid                │
│                                                      │
│                    ┌──────────┐                      │
│                    │Registers │ ← Fastest            │
│                    └────┬─────┘   Smallest           │
│                         │         Most Expensive     │
│                    ┌────▼─────┐                      │
│                    │ L1 Cache │                      │
│                    └────┬─────┘                      │
│                         │                            │
│                  ┌──────▼────────┐                   │
│                  │   L2 Cache    │                   │
│                  └──────┬────────┘                   │
│                         │                            │
│              ┌──────────▼──────────┐                 │
│              │     L3 Cache        │                 │
│              └──────────┬──────────┘                 │
│                         │                            │
│          ┌──────────────▼──────────────┐             │
│          │       Main Memory (RAM)     │             │
│          └──────────────┬──────────────┘             │
│                         │                            │
│      ┌──────────────────▼──────────────────┐         │
│      │    Secondary Storage (SSD/HDD)      │         │
│      └──────────────────┬──────────────────┘         │
│                         │                            │
│   ┌─────────────────────▼─────────────────────┐      │
│   │   Tertiary Storage (Tape, Cloud)          │      │
│   └───────────────────────────────────────────┘      │
│                                               ↓      │
│                                           Slowest    │
│                                           Largest    │
│                                           Cheapest   │
└──────────────────────────────────────────────────────┘

Memory Characteristics:

┌─────────────┬──────────┬───────────┬────────────┬──────────┐
│   Level     │   Size   │   Speed   │ Cost/Byte  │ Location │
├─────────────┼──────────┼───────────┼────────────┼──────────┤
│ Registers   │  <1 KB   │  0.25 ns  │  Highest   │  On-CPU  │
│ L1 Cache    │  32-128KB│  1 ns     │    ↑       │  On-CPU  │
│ L2 Cache    │  256KB-  │  3-10 ns  │    │       │  On-CPU  │
│             │  512KB   │           │    │       │          │
│ L3 Cache    │  2-64 MB │  10-20 ns │    │       │  On-CPU  │
│ Main Memory │  4-128GB │  50-100ns │    │       │  Board   │
│ SSD         │  128GB-  │  50-150μs │    │       │  Board   │
│             │  4TB     │           │    │       │          │
│ HDD         │  500GB-  │  5-10 ms  │    ↓       │  Board   │
│             │  20TB    │           │  Lowest    │          │
└─────────────┴──────────┴───────────┴────────────┴──────────┘

Note: 1 ms = 1,000 μs = 1,000,000 ns
```

## 3.2 Principle of Locality

Memory hierarchy exploits two types of locality:

### 3.2.1 Temporal Locality
```
If a memory location is accessed, it's likely to be 
accessed again in the near future.

Example: Loop variables
┌─────────────────────┐
│ for (i = 0; i < 10; │
│      i++) {         │
│     sum += i;       │ ← 'i' and 'sum' accessed
│ }                   │   repeatedly
└─────────────────────┘

Access Pattern (variable 'i'):
Time: t1  t2  t3  t4  t5  t6  t7  t8
      ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
      i   i   i   i   i   i   i   i
```

### 3.2.2 Spatial Locality
```
If a memory location is accessed, nearby locations 
are likely to be accessed soon.

Example: Array access
┌─────────────────────┐
│ int arr[100];       │
│ for (i = 0; i < 100;│
│      i++) {         │
│     sum += arr[i];  │ ← Sequential access
│ }                   │
└─────────────────────┘

Memory Layout:
┌─────┬────┬────┬────┬────┬────┬────┐
│arr[0]│[1]│[2]│[3]│[4]│[5]│[6]│... │
└─────┴────┴────┴────┴────┴────┴────┘
  ↑    ↑    ↑    ↑    ↑
  Accessed in sequence
```

## 3.3 Cache Memory

Cache is a small, fast memory between CPU and main memory.

### 3.3.1 Cache Organization

```
CPU Request:
┌────────────────────────────────────┐
│         Memory Address             │
├──────────┬───────────┬─────────────┤
│   Tag    │   Index   │   Offset    │
└──────────┴───────────┴─────────────┘
    ↓            ↓           ↓
    │            │           │
    │      Select Set  Select Byte
    │            │      in Block
    │            ↓
    │     ┌──────────────┐
    │     │ Cache Line 0 │
    │     ├──────────────┤
    │     │ Cache Line 1 │
    │     ├──────────────┤
    │     │     ...      │
    │     ├──────────────┤
    │     │ Cache Line n │
    │     └──────────────┘
    │            │
    ▼            ▼
Compare Tags → Hit/Miss
```

### 3.3.2 Cache Line Structure

```
┌──────┬─────┬────────────────────────────────┐
│Valid │ Tag │         Data Block             │
│ Bit  │     │    (typically 64 bytes)        │
└──────┴─────┴────────────────────────────────┘
   1b     n bits         512 bits

Components:
- Valid Bit: Indicates if cache line contains valid data
- Tag: Identifies which memory block is stored
- Data Block: Actual data (cache line/block)
- Dirty Bit (for write-back): Indicates if modified
```

### 3.3.3 Cache Mapping Techniques

#### Direct Mapped Cache
```
Each memory block maps to exactly ONE cache line.

Cache Index = (Memory Address / Block Size) % Number of Cache Lines

Example: 8-line cache, 4-byte blocks
┌───────────┐     ┌──────────────┐
│ Mem Block │────►│ Cache Line   │
│   0, 8, 16│────►│      0       │
│   1, 9, 17│────►│      1       │
│   2,10, 18│────►│      2       │
│   3,11, 19│────►│      3       │
│   4,12, 20│────►│      4       │
│   5,13, 21│────►│      5       │
│   6,14, 22│────►│      6       │
│   7,15, 23│────►│      7       │
└───────────┘     └──────────────┘

Address Format (32-bit):
┌──────────────┬───────┬────────┐
│   Tag (22)   │Index  │Offset  │
│              │ (3)   │ (2)    │
└──────────────┴───────┴────────┘

Advantages:
  - Simple, fast
  - Low cost
  
Disadvantages:
  - High conflict misses
  - Poor utilization if blocks conflict
```

#### Fully Associative Cache
```
Any memory block can go in ANY cache line.

┌───────────┐     ┌──────────────┐
│ Mem Block │     │ Cache Line   │
│  Any      │────►│      Any     │
│  Block    │────►│     Line     │
└───────────┘     └──────────────┘

All lines checked in parallel (CAM - Content Addressable Memory)

Address Format:
┌──────────────────────┬────────┐
│     Tag (30)         │Offset  │
│                      │ (2)    │
└──────────────────────┴────────┘

Comparison Network:
┌─────────────────────────────────┐
│  Tag ──┬──────┬──────┬──────┐   │
│        │      │      │      │   │
│       ▼=     ▼=     ▼=     ▼=   │
│      Line0  Line1  Line2  Line3 │
│        │      │      │      │   │
│        └──────┴──────┴──────┘   │
│               │                  │
│             Hit?                 │
└─────────────────────────────────┘

Advantages:
  - Lowest miss rate
  - Best utilization
  
Disadvantages:
  - Expensive (comparison hardware)
  - Slow (compare all tags)
  - Complex replacement policy
```

#### Set-Associative Cache
```
Compromise: N-way set associative
- Cache divided into sets
- Each memory block maps to a set
- Can go in any line within that set

Example: 2-way set associative, 4 sets
┌───────────┐     ┌─────────────────────┐
│ Mem Block │     │      Set 0          │
│  0, 4, 8  │────►│  [Line 0] [Line 1]  │
│  1, 5, 9  │────►│      Set 1          │
│  2, 6, 10 │────►│  [Line 2] [Line 3]  │
│  3, 7, 11 │────►│      Set 2          │
└───────────┘────►│  [Line 4] [Line 5]  │
                  │      Set 3          │
                  │  [Line 6] [Line 7]  │
                  └─────────────────────┘

Address Format:
┌────────────┬────────┬────────┐
│  Tag (20)  │Set Index│Offset │
│            │  (2)   │ (2)    │
└────────────┴────────┴────────┘

Set selection:
  Set = (Address / Block Size) % Number of Sets

Tag comparison within set:
┌──────────────────────────┐
│ Set Index ──► Select Set │
│                          │
│   ┌──────┐    ┌──────┐   │
│   │Tag=? │    │Tag=? │   │
│   │Valid │    │Valid │   │
│   │Data  │    │Data  │   │
│   └───┬──┘    └───┬──┘   │
│       └─────┬─────┘      │
│            Hit?          │
└──────────────────────────┘

Common configurations:
  - 2-way: good balance
  - 4-way: better miss rate
  - 8-way: diminishing returns
  - 16-way: approaching fully associative

Advantages:
  - Lower miss rate than direct
  - Less expensive than fully associative
  - Good balance

Disadvantages:
  - More complex than direct
  - Slower than direct (compare N tags)
```

### 3.3.4 Cache Replacement Policies

When cache is full, which line to replace?

```
1. Least Recently Used (LRU):
   Replace the line not used for longest time
   
   Access Sequence: A B C D A B E
   
   4-line cache:
   Step 1: [A _ _ _]
   Step 2: [A B _ _]
   Step 3: [A B C _]
   Step 4: [A B C D]
   Step 5: [A B C D] (A accessed, move to front)
   Step 6: [A B C D] (B accessed, move to front)
   Step 7: [A B E D] (Replace C - least recently used)
              ↑
   
   Implementation: Counter or age bits
   - Accurate but expensive for high associativity
   
2. First In First Out (FIFO):
   Replace oldest entry
   
   Simple queue:
   [A] ← [B] ← [C] ← [D]
    ↓                   ↑
   Out               In
   
   Next replacement: A (first in)
   
   - Simple to implement
   - Doesn't consider usage pattern
   
3. Random:
   Replace random line
   
   - Very simple (random number generator)
   - Surprisingly effective
   - Used in some modern processors
   
4. Least Frequently Used (LFU):
   Replace line with lowest access count
   
   Line    Count
   [A]       5
   [B]       2  ← Replace
   [C]       8
   [D]       3
   
   - Good for some access patterns
   - Overhead of maintaining counters

5. Not Recently Used (NRU):
   Approximate LRU with reference bit
   
   Each line has reference bit:
   - Set to 1 on access
   - Periodically cleared
   - Replace line with bit = 0
   
   - Low overhead
   - Good approximation of LRU

┌──────────┬────────────┬──────────┬──────────┐
│ Policy   │ Complexity │ Hardware │ Accuracy │
├──────────┼────────────┼──────────┼──────────┤
│ LRU      │   High     │   High   │  Best    │
│ FIFO     │   Low      │   Low    │  Poor    │
│ Random   │  Lowest    │  Lowest  │  Fair    │
│ LFU      │   High     │   High   │  Good    │
│ NRU      │   Low      │   Low    │  Good    │
└──────────┴────────────┴──────────┴──────────┘
```

### 3.3.5 Write Policies

#### Write-Through
```
Write to cache AND main memory simultaneously

CPU Write ───┬──► Cache (update)
             │
             └──► Memory (update)

Advantages:
  - Main memory always consistent
  - Simple
  - Good for multiprocessor systems

Disadvantages:
  - Slow (memory write every time)
  - High memory traffic
  
Optimization: Write Buffer
┌────┐     ┌─────────┐     ┌────────┐
│CPU │────►│  Cache  │     │ Memory │
└────┘     └────┬────┘     └────────┘
                │              ↑
                ▼              │
           ┌─────────────┐    │
           │Write Buffer │────┘
           │   (Queue)   │
           └─────────────┘
```

#### Write-Back
```
Write only to cache, update memory later

CPU Write ───► Cache (update, set dirty bit)

Memory update when:
- Cache line is replaced
- Explicit flush

┌─────────────────────────────────┐
│ Cache Line                      │
├──────┬─────┬──────┬────────────┤
│Valid │Dirty│ Tag  │    Data    │
│  1   │  1  │ 0x42 │  [Data]    │
└──────┴─────┴──────┴────────────┘
         ↑
    Indicates modified

On replacement:
  IF dirty THEN
      Write to memory
  END IF
  Load new block

Advantages:
  - Fast (no wait for memory)
  - Reduced memory traffic
  - Multiple writes to same location

Disadvantages:
  - Complex (track dirty lines)
  - Memory inconsistent
  - Need write-back on replacement
```

#### Write-Allocate vs No-Write-Allocate
```
On Write Miss:

Write-Allocate:
  1. Load block into cache
  2. Write to cache
  (Usually used with write-back)
  
  Miss ─► Load block ─► Write to cache

No-Write-Allocate:
  1. Write directly to memory
  2. Don't load into cache
  (Usually used with write-through)
  
  Miss ─► Write to memory (skip cache)

┌─────────────────┬──────────────────────┐
│ Cache Policy    │  Write Miss Policy   │
├─────────────────┼──────────────────────┤
│ Write-Through   │ No-Write-Allocate    │
│ Write-Back      │ Write-Allocate       │
└─────────────────┴──────────────────────┘
```

### 3.3.6 Cache Misses

```
Types of Cache Misses (3 C's):

1. Compulsory Miss (Cold Start):
   ┌─────────────────────────────┐
   │ First access to a block     │
   │ Cannot be avoided           │
   │ Cache initially empty       │
   └─────────────────────────────┘
   
   Example: First loop iteration
   arr[0] - MISS (first access)

2. Capacity Miss:
   ┌─────────────────────────────┐
   │ Cache too small             │
   │ Working set > cache size    │
   │ Previously accessed block   │
   │ was evicted                 │
   └─────────────────────────────┘
   
   Example: Large array scan
   Cache: 4 blocks
   Access: [A][B][C][D][E]
                       ↑
   MISS - A was evicted (capacity)

3. Conflict Miss:
   ┌─────────────────────────────┐
   │ Multiple blocks map to      │
   │ same cache location         │
   │ (Direct mapped/set assoc.)  │
   └─────────────────────────────┘
   
   Example: Direct mapped
   Access pattern: Block 0, 8, 0, 8...
   Both map to same line
   0 - MISS
   8 - MISS (evicts 0)
   0 - MISS (evicts 8) ← Conflict
   8 - MISS (evicts 0)

4. Coherence Miss (Multiprocessor):
   ┌─────────────────────────────┐
   │ Other processor modified    │
   │ Cache line invalidated      │
   └─────────────────────────────┘

Miss Rate Reduction:
- Increase cache size (capacity)
- Increase associativity (conflict)
- Larger block size (compulsory)
  (but may increase miss penalty)
- Prefetching (compulsory)
```

### 3.3.7 Multi-Level Cache

```
Modern CPUs have multiple cache levels:

┌─────────────────────────────────────────┐
│              CPU Core                   │
│  ┌────────┐              ┌────────┐     │
│  │ L1-I   │  32-64 KB    │ L1-D   │     │
│  │(Instr) │  1-2 cycles  │ (Data) │     │
│  └────┬───┘              └────┬───┘     │
│       └──────────┬────────────┘         │
└──────────────────┼──────────────────────┘
                   ▼
         ┌──────────────────┐
         │   L2 Cache       │  256-512 KB
         │   Unified        │  10-20 cycles
         └─────────┬────────┘
                   ▼
         ┌──────────────────┐
         │   L3 Cache       │  2-64 MB
         │   Shared         │  30-40 cycles
         └─────────┬────────┘
                   ▼
         ┌──────────────────┐
         │   Main Memory    │  4-128 GB
         │   DRAM           │  100-300 cycles
         └──────────────────┘

Inclusion Policies:

1. Inclusive: L2 contains copy of L1
┌────────────┐
│     L2     │
│ ┌────────┐ │
│ │   L1   │ │  L1 ⊆ L2
│ └────────┘ │
└────────────┘

2. Exclusive: No duplication
┌──────┐  ┌──────┐
│  L1  │  │  L2  │  L1 ∩ L2 = ∅
└──────┘  └──────┘

3. Non-Inclusive: No guarantee
┌──────┐
│  L1  │
└───┬──┘
    │    ┌──────┐
    └────│  L2  │  L1 may overlap L2
         └──────┘

Access Path:
┌─────────────────────────────────┐
│ CPU ──► L1 ──► L2 ──► L3 ──► RAM│
│        hit?   hit?   hit?   hit?│
└─────────────────────────────────┘

Average Access Time:
T_avg = T_L1 + MR_L1 × (T_L2 + MR_L2 × (T_L3 + MR_L3 × T_mem))

Where MR = Miss Rate
```

## 3.4 Main Memory (RAM)

### 3.4.1 DRAM (Dynamic RAM)

```
DRAM Cell Structure:
┌─────────────────────────┐
│      Bit Line           │
│         │               │
│    ┌────┴────┐          │
│    │         │          │
│  ──┤ Access  │          │
│    │Transistor          │
│    └────┬────┘          │
│         │               │
│      ┌──┴──┐            │
│      │ Cap │  ← Stores bit (charge)
│      │  C  │
│      └─────┘            │
│         │               │
│        GND              │
└─────────────────────────┘

1 transistor + 1 capacitor per bit

Characteristics:
- Needs refresh (capacitor leaks)
- Slower than SRAM
- Higher density
- Lower cost per bit
- Main memory

DRAM Organization:
Row Decoder
   │
   ▼
┌────┬────┬────┬────┐
│Cell│Cell│Cell│Cell│◄─── Row
├────┼────┼────┼────┤
│Cell│Cell│Cell│Cell│
├────┼────┼────┼────┤
│Cell│Cell│Cell│Cell│
└─┬──┴─┬──┴─┬──┴─┬──┘
  │    │    │    │
  ▼    ▼    ▼    ▼
    Column Decoder
    Sense Amplifiers
```

### 3.4.2 SRAM (Static RAM)

```
SRAM Cell Structure:
     Vdd
      │
   ┌──┴──┐
   │  │  │
  ┌┴┐ │ ┌┴┐
  │ │ │ │ │
  └┬┘ │ └┬┘
   │ ┌┴┐ │
   └─┤ ├─┘  ← Feedback (stable states)
     └┬┘
      │
  ┌───┴───┐
  │Access │
  │Gates  │
  └───┬───┘
      │
  Bit Lines

6 transistors per bit

Characteristics:
- No refresh needed
- Faster than DRAM
- Lower density
- Higher cost per bit
- Cache memory
```

### 3.4.3 Memory Organization

```
Memory Array Organization:

Address: 32 bits
Data: 64 bits (8 bytes)
Size: 4 GB (2³² bytes)

┌──────────────────────────────────┐
│     Address (32 bits)            │
├────────────────┬─────────────────┤
│  Row Address   │ Column Address  │
│    (16 bits)   │   (16 bits)     │
└────────┬───────┴────────┬────────┘
         │                │
         ▼                ▼
    ┌────────┐      ┌────────┐
    │  Row   │      │ Column │
    │Decoder │      │Decoder │
    └───┬────┘      └────┬───┘
        │                │
        ▼                ▼
   ┌─────────────────────────┐
   │    Memory Cell Array    │
   │    (2D matrix)          │
   │  Row-Column selection   │
   └────────┬────────────────┘
            │
            ▼
     ┌─────────────┐
     │Sense Amps & │
     │   I/O       │
     └──────┬──────┘
            │
            ▼
       Data (64 bits)

Memory Banks:
Multiple independent banks for parallelism

Bank 0    Bank 1    Bank 2    Bank 3
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│      │ │      │ │      │ │      │
│Array │ │Array │ │Array │ │Array │
│      │ │      │ │      │ │      │
└──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
   └────────┴────────┴────────┘
              │
         Shared Bus

Benefits:
- Parallel access
- Higher bandwidth
- Reduced conflicts
```

### 3.4.4 Memory Access Timing

```
DRAM Access Sequence:

   ┌──────┬──────┬──────┬──────┐
   │ RAS  │ CAS  │ Data │Precharge
   │      │      │      │      │
t: 0     15     30     45     60 ns

Phases:
1. RAS (Row Address Strobe):
   - Send row address
   - Activate row
   - tRCD: RAS to CAS delay (15ns)

2. CAS (Column Address Strobe):
   - Send column address
   - Select column
   - tCL: CAS latency (15ns)

3. Data Transfer:
   - Read/Write data
   - Burst transfer possible

4. Precharge:
   - Close row
   - Prepare for next access
   - tRP: Precharge time (15ns)

Total Latency: ~60ns for single access

DDR (Double Data Rate):
Data transferred on both clock edges

Clock:  ┌─┐   ┌─┐   ┌─┐   ┌─┐
        │ │   │ │   │ │   │ │
     ───┘ └───┘ └───┘ └───┘ └───

Data:   ↑ ↑   ↑ ↑   ↑ ↑   ↑ ↑
        D0D1  D2D3  D4D5  D6D7

Effective data rate = 2 × clock rate
```

## 3.5 Virtual Memory

Virtual memory provides:
1. Large address space
2. Protection
3. Sharing

```
┌──────────────────────────────────────────────┐
│         Virtual Memory System                │
│                                              │
│  ┌────────────────┐                          │
│  │Virtual Address │ (Program's view)         │
│  │  0xFFFFFFFF    │                          │
│  │      ...       │                          │
│  │  0x00000000    │                          │
│  └────────┬───────┘                          │
│           │                                  │
│           ▼                                  │
│  ┌─────────────────┐                         │
│  │      MMU        │ (Translation)           │
│  │  (Page Table)   │                         │
│  └────────┬────────┘                         │
│           │                                  │
│           ▼                                  │
│  ┌────────────────┐                          │
│  │Physical Address│ (Actual RAM)             │
│  │  0x7FFFFFFF    │ (smaller)                │
│  │      ...       │                          │
│  │  0x00000000    │                          │
│  └────────────────┘                          │
│           │                                  │
│           ▼                                  │
│  ┌────────────────┐                          │
│  │     Disk       │ (Backing store)          │
│  │  (Swap space)  │                          │
│  └────────────────┘                          │
└──────────────────────────────────────────────┘
```

### 3.5.1 Paging

```
Virtual Memory divided into fixed-size pages
Physical Memory divided into fixed-size frames

Typical page size: 4 KB (4096 bytes)

Virtual Address (32-bit, 4KB pages):
┌──────────────────────┬────────────┐
│  Page Number (20)    │ Offset(12) │
└──────────┬───────────┴────────────┘
           │               │
           ▼               │
    ┌─────────────┐        │
    │ Page Table  │        │
    │             │        │
    │  VPN → PFN  │        │
    └──────┬──────┘        │
           │               │
           ▼               │
Physical Address:          │
┌────────────────┬─────────▼──┐
│Frame Number(20)│ Offset(12) │
└────────────────┴────────────┘

Page Table Entry (PTE):
┌──────┬──────┬──────┬──────┬──────┬────────────┐
│Valid │Dirty │Access│R/W   │U/S   │Frame Number│
│ (1)  │ (1)  │ (1)  │ (1)  │ (1)  │   (20)     │
└──────┴──────┴──────┴──────┴──────┴────────────┘

Flags:
- Valid: Page in memory
- Dirty: Modified since loaded
- Access: Recently accessed
- R/W: Read-only or read-write
- U/S: User or supervisor mode

Page Table Types:

1. Single-Level Page Table:
┌────────────────┐
│Virtual Address │
├────────┬───────┤
│  VPN   │Offset │
└───┬────┴───────┘
    │
    ▼
┌─────────────┐
│ Page Table  │
│ [0] → PFN   │
│ [1] → PFN   │
│ [2] → PFN   │
│     ...     │
└─────────────┘

Problem: Large tables
  32-bit VA, 4KB pages
  = 2²⁰ entries × 4 bytes
  = 4 MB per process!

2. Multi-Level Page Table:
┌────────────────────────────┐
│     Virtual Address        │
├──────┬──────┬──────┬───────┤
│  P1  │  P2  │  P3  │Offset │
└───┬──┴───┬──┴───┬──┴───────┘
    │      │      │
    ▼      │      │
┌────────┐ │      │
│Level 1 │ │      │
│  Dir   │ │      │
└───┬────┘ │      │
    │      │      │
    ▼      ▼      │
  ┌────────┐      │
  │Level 2 │      │
  │  Dir   │      │
  └───┬────┘      │
      │           │
      ▼           ▼
    ┌────────┐
    │Level 3 │
    │  Page  │
    │  Table │
    └───┬────┘
        │
        ▼
   Physical Frame

Advantages:
- Sparse tables (allocate on demand)
- Reduced memory usage
- Scalable
```

### 3.5.2 Translation Lookaside Buffer (TLB)

```
TLB: Cache for page table entries

┌─────────────────────────────────────────────┐
│         Address Translation with TLB        │
│                                             │
│  Virtual Address                            │
│       │                                     │
│       ▼                                     │
│  ┌─────────┐                                │
│  │   TLB   │ (Small, fast cache)            │
│  │         │                                │
│  └────┬────┴───────┐                        │
│       │ Hit        │ Miss                   │
│       ▼            ▼                        │
│   Physical    ┌──────────┐                  │
│   Address     │Page Table│                  │
│               │(in Memory)                  │
│               └─────┬────┘                  │
│                     │                       │
│                     ▼                       │
│              Physical Address               │
│                     │                       │
│              Update TLB                     │
└─────────────────────┴───────────────────────┘

TLB Entry:
┌──────┬─────┬─────────────┬──────────┐
│Valid │ Tag │ Frame Number│ Flags    │
└──────┴─────┴─────────────┴──────────┘

TLB Organization (Associative):
┌────────────────────────────────┐
│  TLB (typically 64-256 entries)│
│                                │
│  VPN     PFN    Flags          │
│ ┌────┐ ┌────┐ ┌──────┐         │
│ │0x12│ │0xAB│ │V D A R│        │
│ ├────┤ ├────┤ ├──────┤         │
│ │0x34│ │0xCD│ │V D A R│        │
│ ├────┤ ├────┤ ├──────┤         │
│ │ ...│ │ ...│ │  ... │         │
│ └────┘ └────┘ └──────┘         │
│   ↑                            │
│   └── Parallel search          │
└────────────────────────────────┘

Access Time Calculation:
  t_avg = t_TLB + (1 - h_TLB) × t_page_table
  
  Where h_TLB = TLB hit rate (typically 95-99%)

Example:
  TLB access: 1 ns
  Memory access: 100 ns
  TLB hit rate: 98%
  
  t_avg = 1 + (1 - 0.98) × 100
        = 1 + 0.02 × 100
        = 1 + 2
        = 3 ns
```

### 3.5.3 Page Faults

```
Page Fault: Requested page not in memory

┌────────────────────────────────────────┐
│      Page Fault Handling               │
│                                        │
│  1. CPU accesses virtual address       │
│     │                                  │
│     ▼                                  │
│  2. TLB miss → Check page table        │
│     │                                  │
│     ▼                                  │
│  3. Valid bit = 0 → Page Fault!        │
│     │                                  │
│     ▼                                  │
│  4. Trap to OS                         │
│     │                                  │
│     ▼                                  │
│  5. Find page on disk                  │
│     │                                  │
│     ▼                                  │
│  6. Find free frame                    │
│     │ (or evict a page)                │
│     ▼                                  │
│  7. Load page from disk → memory       │
│     │ (millions of cycles!)            │
│     ▼                                  │
│  8. Update page table                  │
│     │                                  │
│     ▼                                  │
│  9. Restart instruction                │
└────────────────────────────────────────┘

Page Replacement Algorithms:

1. FIFO (First-In-First-Out):
   Memory: [A][B][C]
   Access D: Replace A (oldest)
   Result: [D][B][C]

2. LRU (Least Recently Used):
   Memory: [A][B][C]
   Access times: A=1, B=5, C=3
   Replace A (least recent)

3. LFU (Least Frequently Used):
   Memory: [A][B][C]
   Counts: A=10, B=2, C=5
   Replace B (least frequent)

4. Optimal (theoretical):
   Replace page not needed for longest time
   Not implementable (requires future knowledge)
   Used as benchmark

5. Clock (Second Chance):
   Circular list with reference bit
   
        ┌───┐
    ┌──►│ A │ R=1
    │   ├───┤
    │   │ B │ R=0 ← Replace
    │   ├───┤
    │   │ C │ R=1
    │   └───┘
    │     ↑
    └─────┘
    Hand pointer
    
   - Scan circularly
   - If R=1, set R=0 and continue
   - If R=0, replace
```

### 3.5.4 Memory Protection and Sharing

```
Protection:
┌──────────────────────────────────┐
│  Process A     Process B         │
│  ┌──────┐      ┌──────┐          │
│  │ VM   │      │ VM   │          │
│  │Space │      │Space │          │
│  └───┬──┘      └───┬──┘          │
│      │             │             │
│      ▼             ▼             │
│  ┌────────────────────┐          │
│  │   Physical Memory  │          │
│  │  ┌────┐   ┌────┐  │           │
│  │  │ A  │   │ B  │  │           │
│  │  │Pages   │Pages  │           │
│  │  └────┘   └────┘  │           │
│  └────────────────────┘          │
│                                  │
│  Page tables ensure isolation    │
└──────────────────────────────────┘

Sharing:
┌──────────────────────────────────┐
│  Process A     Process B         │
│  ┌──────┐      ┌──────┐          │
│  │ VM   │      │ VM   │          │
│  └───┬──┘      └───┬──┘          │
│      │             │             │
│      └──────┬──────┘             │
│             ▼                    │
│      ┌────────────┐              │
│      │   Shared   │              │
│      │  Library   │              │
│      │  (e.g.,    │              │
│      │   libc)    │              │
│      └────────────┘              │
│                                  │
│  Both map to same physical pages │
└──────────────────────────────────┘

Protection Bits in PTE:
- R (Read): Can read page
- W (Write): Can write page
- X (Execute): Can execute code
- U (User): User mode accessible
- S (Supervisor): Kernel mode only

Example combinations:
  R  W  X  | Usage
  1  0  0  | Read-only data
  1  1  0  | Read-write data
  1  0  1  | Code (executable)
  1  1  1  | Rare (security risk)
```

---

**Key Takeaways:**
1. Memory hierarchy trades off speed, size, and cost
2. Cache exploits locality to provide fast average access
3. Multiple cache levels balance performance and cost
4. Virtual memory provides large address space and protection
5. TLB accelerates address translation
6. Page replacement policies impact performance

**Next:** [Instruction Set Architecture](./04-instruction-sets.md)


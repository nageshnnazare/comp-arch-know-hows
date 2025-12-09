# Chapter 5: Pipelining

## 5.1 Pipeline Concept

Pipelining overlaps execution of multiple instructions to improve throughput.

```
Analogy: Laundry Pipeline
────────────────────────────────────────────
Task breakdown per load:
1. Wash   (30 min)
2. Dry    (30 min)
3. Fold   (30 min)
4. Store  (30 min)

Sequential (No Pipeline):
Load 1: [W][D][F][S]
Load 2:             [W][D][F][S]
Load 3:                         [W][D][F][S]
Time:   ────────────────────────────────────►
        0   30  60  90 120 150 180 210 240 min
        Total: 240 minutes for 3 loads

Pipelined:
Load 1: [W][D][F][S]
Load 2:    [W][D][F][S]
Load 3:       [W][D][F][S]
Time:   ────────────────────────────────────►
        0   30  60  90 120 150 min
        Total: 150 minutes for 3 loads!

Speedup: 240/150 = 1.6×
```

### 5.1.1 Instruction Pipeline

```
Classic 5-Stage RISC Pipeline:

┌─────┬────────┬─────────┬────────┬───────────┐
│ IF  │   ID   │   EX    │  MEM   │    WB     │
│Fetch│ Decode │ Execute │ Memory │Write Back │
└─────┴────────┴─────────┴────────┴───────────┘

Stage Details:

1. IF (Instruction Fetch):
   - Read instruction from memory
   - Update PC
   
2. ID (Instruction Decode):
   - Decode instruction
   - Read register operands
   - Generate control signals
   
3. EX (Execute):
   - Perform ALU operation
   - Calculate address
   - Evaluate branch condition
   
4. MEM (Memory Access):
   - Read/write data memory
   - (Only for load/store)
   
5. WB (Write Back):
   - Write result to register
   - Update register file
```

### 5.1.2 Pipeline Execution

```
Time ──────────────────────────────────────►
Cycle: 1    2    3    4    5    6    7    8

I1:   [IF][ID][EX][MEM][WB]
I2:        [IF][ID][EX][MEM][WB]
I3:             [IF][ID][EX][MEM][WB]
I4:                  [IF][ID][EX][MEM][WB]
I5:                       [IF][ID][EX][MEM][WB]

Observations:
- At cycle 5: All 5 stages busy (max throughput)
- Ideal CPI = 1 (one instruction per cycle)
- Latency per instruction: 5 cycles
- Throughput: 1 instruction/cycle (steady state)

Sequential vs Pipelined:
Sequential: 5 instr × 5 cycles = 25 cycles
Pipelined:  5 + (5-1) = 9 cycles
Speedup: 25/9 = 2.78×

Maximum speedup with N stages:
  Speedup = N (when # instructions >> N)
```

## 5.2 Pipeline Performance

```
Ideal Pipeline Performance:

CPI_ideal = 1 (one instruction per cycle)

Speedup = Pipeline_Depth / (1 + Pipeline_Stall_CPI)

Time_pipelined = (Instruction_Count + Pipeline_Depth - 1) × Cycle_Time

Example:
  1000 instructions, 5-stage pipeline
  Sequential: 1000 × 5 = 5000 cycles
  Pipelined:  1000 + 5 - 1 = 1004 cycles
  Speedup: 5000/1004 = 4.98×

Pipeline Throughput:
  Instructions per cycle at steady state = 1
  
Clock Speed Improvement:
  Sequential cycle = sum of all stage delays
  Pipelined cycle = max(stage delays) + latch overhead

Example:
  IF: 200ps, ID: 100ps, EX: 150ps, MEM: 200ps, WB: 100ps
  Latch overhead: 20ps per stage
  
  Sequential cycle = 200+100+150+200+100 = 750ps
  Pipelined cycle = max(200,100,150,200,100) + 20 = 220ps
  
  Frequency improvement: 750/220 = 3.4× faster clock
```

### 5.2.1 Pipeline Limitations

```
Real Performance:

CPI_actual = CPI_ideal + CPI_stalls

Stalls caused by:
1. Structural hazards (resource conflicts)
2. Data hazards (dependencies)
3. Control hazards (branches)

Example:
  CPI_ideal = 1
  CPI_stalls = 0.3 (30% of time stalled)
  CPI_actual = 1.3
  
  Effective speedup = 5/1.3 = 3.85×
  (vs ideal 5×)

Pipeline Efficiency:
  Efficiency = CPI_ideal / CPI_actual
             = 1 / 1.3
             = 77%
```

## 5.3 Pipeline Hazards

### 5.3.1 Structural Hazards

Resource conflict: two instructions need same hardware.

```
Problem: Single memory for instructions and data

Cycle: 1    2    3    4    5    6
I1:   [IF][ID][EX][MEM][WB]
I2:        [IF][ID][EX][MEM][WB]
I3:             [IF][ID][EX][MEM][WB]
I4:                  [IF][ID]  ← STALL!
                          ↑
                     Conflict: I1 needs MEM (data)
                               I4 needs MEM (instruction)

Solutions:

1. Separate Instruction and Data Memory:
   ┌───────────────┐
   │  Instruction  │ ← IF stage
   │    Memory     │
   └───────────────┘
   
   ┌───────────────┐
   │     Data      │ ← MEM stage
   │    Memory     │
   └───────────────┘
   (Harvard Architecture)

2. Stall Pipeline:
   Wait for resource to become available
   (Reduces performance)

3. Duplicate Resources:
   Multiple ALUs, register file ports, etc.

4. Resource Scheduling:
   Clever scheduling to avoid conflicts

Register File Ports:
   Need 2 read ports + 1 write port
   (ID reads 2 operands, WB writes result)
   
   ┌──────────────────┐
   │  Register File   │
   │                  │
   │  Read Port 1  ──►│ ID stage
   │  Read Port 2  ──►│ ID stage
   │  Write Port   ◄──│ WB stage
   └──────────────────┘
```

### 5.3.2 Data Hazards

Instruction depends on result of previous instruction.

#### Read After Write (RAW) - True Dependency

```
ADD R1, R2, R3   ; R1 = R2 + R3
SUB R4, R1, R5   ; R4 = R1 - R5 (needs R1!)
                    ↑
                  Dependency

Pipeline without forwarding:

Cycle: 1    2    3    4    5    6    7
ADD:  [IF][ID][EX][MEM][WB]
SUB:       [IF][ID][EX][MEM][WB]
                ↑      ↑
             Need R1  R1 available

Problem: SUB needs R1 in cycle 3 (EX stage)
         ADD writes R1 in cycle 5 (WB stage)
         Too late!

Without solution:
Cycle: 1    2    3    4    5    6    7    8
ADD:  [IF][ID][EX][MEM][WB]
SUB:       [IF][ID][  ][  ][EX][MEM][WB]
                    ↑
                 2-cycle stall
```

**Solution 1: Stalling (Pipeline Bubble)**

```
Hardware detects dependency and stalls

Cycle: 1    2    3    4    5    6    7    8
ADD:  [IF][ID][EX][MEM][WB]
SUB:       [IF][ID][--][--][EX][MEM][WB]
                    ↑    ↑
                 NOPs inserted

Implementation:
  IF (ID_stage_instruction needs result from EX/MEM/WB stage) THEN
      Stall IF and ID stages
      Insert bubble (NOP) in EX stage
  END IF

Performance impact: CPI increases
```

**Solution 2: Forwarding (Bypassing)**

```
Forward result from later stage to earlier stage

Cycle: 1    2    3    4    5    6    7
ADD:  [IF][ID][EX][MEM][WB]
                ↓
SUB:       [IF][ID][EX][MEM][WB]
                    ↑
          Forward from EX/MEM register

Forwarding Paths:
┌─────────────────────────────────────────┐
│        Pipeline Stages                  │
│  [IF][ID][EX][MEM][WB]                  │
│              │    │    │                │
│              │    │    └──┐             │
│              │    └────┐  │             │
│              └──┐      │  │             │
│        [IF][ID][EX][MEM][WB]            │
│                 ↑   ↑                   │
│           From EX/MEM,                  │
│           MEM/WB stages                 │
└─────────────────────────────────────────┘

Forwarding Mux at EX stage:
         From EX/MEM ──┐
         From MEM/WB ──┤
         From ID/EX  ──┴─► ALU
         
No stall needed! CPI = 1
```

**Load-Use Hazard (Unavoidable Stall)**

```
LW   R1, 0(R2)   ; Load: R1 = Memory[R2]
ADD  R3, R1, R4  ; Use R1 immediately

Cycle: 1    2    3    4    5    6    7
LW:   [IF][ID][EX][MEM][WB]
                     ↑
                  Data available
ADD:       [IF][ID][  ][EX][MEM][WB]
                ↑   ↑
             Need R1 1-cycle stall

Even with forwarding, need 1 stall cycle!

Data from memory not available until end of MEM stage
ADD needs it at start of EX stage

Solution: Compiler reordering
LW   R1, 0(R2)
ADD  R5, R6, R7  ; Independent instruction
ADD  R3, R1, R4  ; Use R1 (no stall now!)
```

#### Write After Read (WAR) - Anti-dependency

```
SUB R4, R1, R3   ; Read R1
ADD R1, R2, R3   ; Write R1

Cycle: 1    2    3    4    5    6
SUB:  [IF][ID][EX][MEM][WB]
           ↑ Read R1
ADD:       [IF][ID][EX][MEM][WB]
                         ↑ Write R1

In-order pipeline: Not a problem!
  SUB reads R1 in cycle 2 (ID)
  ADD writes R1 in cycle 6 (WB)
  Correct order maintained

Problem only in:
  - Out-of-order execution
  - Compiler optimization
```

#### Write After Write (WAW) - Output Dependency

```
ADD R1, R2, R3   ; Write R1
SUB R1, R4, R5   ; Write R1

In simple pipeline: Not a problem
  Both writes in program order

Problem in:
  - Out-of-order execution
  - Multiple functional units
  
Solution: Register renaming
```

### 5.3.3 Control Hazards

Branch instructions change program flow.

```
Problem: Don't know next instruction until branch resolves

Branch Resolution in Different Stages:

Best case (ID):
Cycle: 1    2    3    4    5    6
BR:   [IF][ID][EX][MEM][WB]
           ↑ Branch resolved here
Next:       [IF][ID][EX][MEM][WB]
            1 cycle penalty

Typical case (EX):
Cycle: 1    2    3    4    5    6    7
BR:   [IF][ID][EX][MEM][WB]
                ↑ Branch resolved
Next:            [IF][ID][EX][MEM][WB]
            2 cycle penalty

Worst case (MEM):
Cycle: 1    2    3    4    5    6    7    8
BR:   [IF][ID][EX][MEM][WB]
                     ↑ Resolved
Next:                 [IF][ID][EX][MEM][WB]
            3 cycle penalty
```

**Solution 1: Stall Until Branch Resolves**

```
Cycle: 1    2    3    4    5    6    7
BR:   [IF][ID][EX][MEM][WB]
           ↑
     [IF][ID][--][--][EX][MEM][WB]
Next (correct):      [IF][ID][EX][MEM][WB]

Simple but slow
CPI penalty: stall_cycles × branch_frequency

Example:
  Branch resolves in EX (2 cycle stall)
  Branch frequency: 20%
  CPI penalty = 2 × 0.2 = 0.4
  CPI = 1 + 0.4 = 1.4
```

**Solution 2: Predict Not Taken**

```
Assume branch not taken, continue with PC+4

If prediction correct:
  No penalty!
  
If prediction wrong:
Cycle: 1    2    3    4    5    6    7    8
BR:   [IF][ID][EX][MEM][WB]
                ↑ Wrong!
I1:        [IF][ID][EX]← Flush
I2:             [IF][ID]← Flush
Correct:             [IF][ID][EX][MEM][WB]

Must flush wrong-path instructions
Penalty only on misprediction
```

**Solution 3: Predict Taken**

```
Assume branch taken, fetch from target

Calculate target address early (in ID)
Start fetching from target

If correct: No penalty
If wrong: Flush and restart
```

**Solution 4: Dynamic Branch Prediction**

```
Use history to predict branch direction

1-bit Predictor:
┌──────────────┬────────────┐
│ Branch PC    │ Last Result│
├──────────────┼────────────┤
│  0x1000      │   Taken    │
│  0x1020      │ Not Taken  │
│  0x1040      │   Taken    │
└──────────────┴────────────┘

Problem: Loop last iteration causes 2 mispredictions
  Loop: T T T T T N
  Pred: T T T T T T ✗
  Next: N ✗ T T T T

2-bit Saturating Counter (better):
     ┌─────────────────┐
     │Strongly Taken(11)│ ←──┐
     └────┬────────────┘    │
   Taken  │  Not Taken      │ Taken
          ▼                 │
     ┌─────────────────┐    │
     │Weakly Taken (10)│ ───┘
     └────┬────────────┘
          │  Not Taken
          ▼
     ┌─────────────────┐
     │Weakly Not       │
     │Taken (01)       │
     └────┬────────────┘
   Taken  │  Not Taken
          ▼
     ┌─────────────────┐
     │Strongly Not     │
     │Taken (00)       │
     └─────────────────┘

Only mispredicts once at loop end!

Branch History Table (BHT):
┌────────┬──────────┬──────────┐
│Index   │ 2-bit    │ Target   │
│(PC)    │ Counter  │ Address  │
├────────┼──────────┼──────────┤
│  000   │    11    │  0x2000  │
│  001   │    10    │  0x3000  │
│  010   │    01    │  0x1500  │
│  ...   │   ...    │   ...    │
└────────┴──────────┴──────────┘

Typical size: 512-4096 entries
Accuracy: 90-95%

Two-Level Adaptive Predictor:
  Global history → Pattern → Prediction
  Tracks correlation between branches
  Accuracy: 95-98%
```

**Solution 5: Delayed Branch**

```
Compiler fills branch delay slot with useful instruction

Original:
  CMP R1, R2
  BEQ target
  ; delay slot
  ADD R3, R4, R5  ; At target

Reordered:
  CMP R1, R2
  ADD R3, R4, R5  ; Move to delay slot
  BEQ target      ; ADD executes regardless
  
Branch delay slot: instruction after branch always executes

Pipeline:
Cycle: 1    2    3    4    5    6
BEQ:  [IF][ID][EX][MEM][WB]
ADD:       [IF][ID][EX][MEM][WB]  ← Always executes
Next:           [IF][ID][EX][MEM][WB]

No penalty if slot filled!

Used in: MIPS, SPARC
Not used in modern CPUs (prediction is better)
```

### 5.3.4 Hazard Detection and Resolution Summary

```
┌──────────────────┬──────────────┬──────────────────┐
│  Hazard Type     │   Cause      │    Solution      │
├──────────────────┼──────────────┼──────────────────┤
│ Structural       │ Resource     │ - Separate I/D$  │
│                  │ conflict     │ - Duplicate HW   │
│                  │              │ - Stall          │
├──────────────────┼──────────────┼──────────────────┤
│ Data (RAW)       │ Dependency   │ - Forwarding     │
│                  │              │ - Stall          │
│                  │              │ - Reorder        │
├──────────────────┼──────────────┼──────────────────┤
│ Data (WAR/WAW)   │ Out-of-order │ - Register       │
│                  │              │   renaming       │
├──────────────────┼──────────────┼──────────────────┤
│ Control          │ Branches     │ - Prediction     │
│                  │              │ - Delayed branch │
│                  │              │ - Early resolve  │
└──────────────────┴──────────────┴──────────────────┘
```

## 5.4 Advanced Pipelining

### 5.4.1 Deeper Pipelines

```
Intel Pentium 4: 20-31 stages!

More stages = Higher clock frequency

Example progression:
Classic RISC: 5 stages
┌────┬────┬────┬────┬────┐
│ IF │ ID │ EX │MEM │ WB │
└────┴────┴────┴────┴────┘

Pentium Pro: 12 stages
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│F1│F2│D1│D2│E1│E2│M1│M2│W1│W2│...  │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Pentium 4: 20+ stages
┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│ Many fine-grained stages...           │
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

Benefits:
  + Higher clock frequency
  + Simple per-stage logic
  + Better timing closure

Drawbacks:
  - Higher branch misprediction penalty
  - More pipeline stages to flush
  - More complex forwarding
  - More latch overhead
  - Power consumption

Branch misprediction cost:
  5-stage:  2-3 cycles lost
  20-stage: 15-20 cycles lost!

Modern trend: Moderate depth (12-15 stages)
Balance between frequency and efficiency
```

### 5.4.2 Superscalar Processors

```
Multiple instructions issued per cycle

Dual-Issue Superscalar:
Cycle: 1       2       3       4       5
I1,I2 [IF][ID][EX][MEM][WB]
I3,I4      [IF][ID][EX][MEM][WB]
I5,I6           [IF][ID][EX][MEM][WB]

Ideal: CPI = 0.5 (IPC = 2)

Requirements:
1. Multiple instruction fetch
2. Multiple decode units
3. Multiple execution units
4. Register file with more ports
5. Hazard detection for parallel instructions

Organization:
┌─────────────────────────────────────────┐
│          Fetch Unit                     │
│      (Fetch 4-8 inst/cycle)             │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│         Decode/Dispatch                 │
│    (Issue 2-4 inst/cycle)               │
└───┬─────────────────────────────────┬───┘
    ▼                                 ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ ALU 0  │  │ ALU 1  │  │Load/Str│  │ FP Unit│
└────┬───┘  └────┬───┘  └────┬───┘  └────┬───┘
     └───────────┴───────────┴───────────┘
                     ▼
              ┌──────────────┐
              │ Write Back   │
              └──────────────┘

Example: 4-wide superscalar
  Can execute up to 4 instructions per cycle
  Requires checking dependencies between all pairs
  
  4 instructions: 6 pairs to check
  8 instructions: 28 pairs to check
  Complexity: O(n²)

Limitations:
  - True dependencies limit parallelism
  - Not always 4 independent instructions
  - Structural hazards
  
Actual IPC: 2-3 (not 4) for 4-wide superscalar
```

### 5.4.3 Out-of-Order Execution (OOO)

```
Execute instructions as soon as operands available

┌──────────────────────────────────────────┐
│    In-Order Front End                    │
│      (Fetch & Decode)                    │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│        Instruction Queue                 │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│     Reservation Stations                 │
│  (Wait for operands, issue when ready)   │
└──────────┬───────────────────────────────┘
           ▼
┌──────────────────────────────────────────┐
│   Out-of-Order Execution                 │
│  (Multiple functional units)             │
└──────────┬───────────────────────────────┘
           ▼
┌──────────────────────────────────────────┐
│      Reorder Buffer (ROB)                │
│   (Restore program order)                │
└──────────┬───────────────────────────────┘
           ▼
┌──────────────────────────────────────────┐
│    In-Order Retirement/Commit            │
└──────────────────────────────────────────┘

Example:
Program order:
  I1: LW  R1, 0(R2)     ; Load (slow)
  I2: ADD R3, R1, R4    ; Depends on I1
  I3: SUB R5, R6, R7    ; Independent!
  I4: AND R8, R5, R9    ; Depends on I3

In-order execution:
  Cycle: 1 2 3 4 5 6 7 8 9 10 11 12
  I1:    [F D E M M W]
  I2:          [F D - - E M W]
  I3:                [F D - - E M W]
  I4:                      [F D E M W]

Out-of-order execution:
  Cycle: 1 2 3 4 5 6 7 8 9 10
  I1:    [F D E M M W]
  I2:          [F D - - E M W]
  I3:          [F D E M W]      ← Executes early!
  I4:             [F D E M W]   ← Executes early!
  
Total: 10 cycles vs 12 cycles

Key Components:

1. Reservation Stations:
   ┌────┬────┬────┬────┬─────┐
   │Op  │Op1 │Op2 │Dest│Ready│
   ├────┼────┼────┼────┼─────┤
   │ADD │ V  │ V  │ R3 │ Yes │ ← Can execute
   │SUB │Tag │ V  │ R5 │ No  │ ← Wait for operand
   │MUL │ V  │ V  │ R7 │ Yes │ ← Can execute
   └────┴────┴────┴────┴─────┘

2. Reorder Buffer:
   Maintains program order for commit
   ┌─────┬────┬──────┬──────┬────────┐
   │Entry│Inst│Result│ Dest │Complete│
   ├─────┼────┼──────┼──────┼────────┤
   │  1  │ADD │  10  │  R3  │  Yes   │
   │  2  │SUB │  5   │  R5  │  Yes   │
   │  3  │MUL │  -   │  R7  │  No    │
   └─────┴────┴──────┴──────┴────────┘
   
   Commit head when complete, in order

3. Register Renaming:
   Eliminate false dependencies (WAR, WAW)
   
   Architectural registers: R0-R31
   Physical registers: P0-P127
   
   Rename Table:
   ┌──────┬──────────┐
   │Arch  │ Physical │
   ├──────┼──────────┤
   │ R1   │   P45    │
   │ R2   │   P12    │
   │ R3   │   P87    │
   │ ...  │   ...    │
   └──────┴──────────┘
   
   Example:
   ADD R1, R2, R3  → ADD P45, P12, P87
   SUB R1, R4, R5  → SUB P92, P20, P31  (new P92!)
   
   Both can execute in parallel!
   No WAW hazard
```

### 5.4.4 VLIW (Very Long Instruction Word)

```
Compiler packs multiple operations into one instruction

VLIW Instruction (e.g., 128 bits):
┌────────┬────────┬────────┬────────┐
│  ALU1  │  ALU2  │Load/Str│ Branch │
│ 32 bits│ 32 bits│ 32 bits│ 32 bits│
└────────┴────────┴────────┴────────┘

All operations execute in parallel

Example (IA-64/Itanium):
┌────────────────────────────────────────┐
│ Bundle (128 bits):                     │
│                                        │
│  [ADD R1,R2,R3][LD R4,0(R5)][BR loop] │
│        ↓              ↓           ↓    │
│      ALU1          Memory      Branch  │
└────────────────────────────────────────┘

All execute simultaneously!

Compiler's job:
  1. Find independent operations
  2. Schedule them together
  3. Fill empty slots with NOPs if needed

Example:
Original:
  ADD R1, R2, R3
  SUB R4, R5, R6
  MUL R7, R8, R9
  LD  R10, 0(R11)

VLIW (2-wide):
  [ADD R1,R2,R3][SUB R4,R5,R6]
  [MUL R7,R8,R9][LD R10,0(R11)]

2 cycles instead of 4!

Advantages:
  + Simple hardware
  + No dynamic scheduling needed
  + Lower power
  + Explicit parallelism

Disadvantages:
  - Compiler complexity
  - Code size (NOPs)
  - Binary incompatibility
  - Cache miss handling
  - Unpredictable latencies

Comparison:
┌──────────────┬───────────┬─────────┐
│   Feature    │Superscalar│  VLIW   │
├──────────────┼───────────┼─────────┤
│ Parallelism  │  Dynamic  │ Static  │
│ Scheduled by │ Hardware  │Compiler │
│ Hardware     │  Complex  │ Simple  │
│ Code size    │  Smaller  │ Larger  │
│ Power        │  Higher   │  Lower  │
│ Adaptability │   Good    │  Poor   │
└──────────────┴───────────┴─────────┘
```

### 5.4.5 Multi-threading

```
SMT (Simultaneous Multi-Threading) / Hyper-Threading

Run multiple threads on same core

Traditional Superscalar (underutilized):
Cycle: 1    2    3    4    5
ALU1: [I1][I3][I5][  ][I9]
ALU2: [I2][I4][  ][I7][  ]
Mem:  [  ][  ][I6][I8][  ]
      50% utilization ──────────►

With SMT (2 threads):
Cycle: 1    2    3    4    5
ALU1: [T1][T1][T2][T1][T2]
ALU2: [T2][T1][T1][T2][T1]
Mem:  [T1][T2][T1][T2][T1]
      90% utilization ──────────►

Architecture:
┌──────────────────────────────────┐
│  Replicated per thread:          │
│  - PC                            │
│  - Registers                     │
│  - Return stack                  │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  Shared:                         │
│  - Fetch/Decode                  │
│  - Execution units               │
│  - Caches                        │
│  - TLBs                          │
└──────────────────────────────────┘

Intel Hyper-Threading:
  2 threads per core
  2-30% performance improvement
  Minimal hardware cost

Coarse-Grained Multi-threading:
  Switch threads on long-latency events
  (cache miss, branch misprediction)

Fine-Grained Multi-threading:
  Switch threads every cycle
  (Round-robin)

Simultaneous Multi-threading (SMT):
  Issue from multiple threads same cycle
  Best utilization
  Most complex

Example utilization:
Single thread:      ████░░░░░░  40%
Coarse MT:         ████░░██░░  60%
Fine MT:           █░█░█░█░█░  80%
SMT:               ████████░░  90%
```

## 5.5 Pipeline Performance Analysis

```
Real-World Example:

Processor: 5-stage pipeline, 3 GHz
Program: 1 million instructions
Branch frequency: 20%
Load frequency: 30%

Hazards:
  - Branch misprediction: 10% (2-cycle penalty)
  - Load-use: 20% of loads (1-cycle penalty)

CPI calculation:
  CPI_ideal = 1.0
  
  Branch penalty:
    0.20 × 0.10 × 2 = 0.04
    
  Load-use penalty:
    0.30 × 0.20 × 1 = 0.06
    
  CPI_actual = 1.0 + 0.04 + 0.06 = 1.10

Execution time:
  Time = Instructions × CPI × Cycle_Time
       = 1,000,000 × 1.10 × (1/3×10⁹)
       = 0.367 ms

Without pipeline (5× CPI):
  Time = 1,000,000 × 5 × (1/3×10⁹)
       = 1.67 ms

Speedup = 1.67 / 0.367 = 4.55×

Efficiency = 4.55/5 = 91%

Amdahl's Law for Pipeline:
  Speedup = 1 / ((1-P) + P/N)
  
  Where:
    P = Fraction parallelizable
    N = Number of stages
    
  For P=0.95, N=5:
    Speedup = 1 / (0.05 + 0.95/5)
            = 4.35×
```

---

**Key Takeaways:**
1. Pipelining overlaps instruction execution for higher throughput
2. Ideal CPI = 1, but hazards reduce performance
3. Hazards: Structural, Data (RAW/WAR/WAW), Control
4. Solutions: Forwarding, stalling, prediction, reordering
5. Advanced: Superscalar, OOO execution, SMT
6. Deeper pipelines increase frequency but worsen branch penalty
7. Compiler and hardware cooperate to maximize performance

**Next:** [I/O Systems](./06-io-systems.md)


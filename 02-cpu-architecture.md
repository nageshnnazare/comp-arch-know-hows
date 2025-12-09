# Chapter 2: CPU Architecture and Organization

## 2.1 CPU Overview

The Central Processing Unit (CPU) is the brain of the computer, responsible for executing instructions and performing calculations.

```
┌─────────────────────────────────────────────────┐
│                   CPU                           │
│  ┌──────────────┐         ┌─────────────────┐   │
│  │   Control    │◄────────┤   Registers     │   │
│  │    Unit      │         │   (PC, IR, etc) │   │
│  │     (CU)     │         └─────────────────┘   │
│  └──────┬───────┘                               │
│         │                                       │
│         ▼                                       │
│  ┌──────────────┐         ┌─────────────────┐   │
│  │  Arithmetic  │◄────────┤     Cache       │   │
│  │    Logic     │         │    (L1/L2)      │   │
│  │  Unit (ALU)  │         └─────────────────┘   │
│  └──────────────┘                               │
└──────────────┬──────────────────────────────────┘
               │
               ▼
        [System Bus]
        (Data, Address, Control)
```

## 2.2 CPU Components

### 2.2.1 Arithmetic Logic Unit (ALU)

The ALU performs arithmetic and logical operations.

```
         ┌─────────────────────────────┐
Operand A│                             │
────────►│                             │
         │          ALU                │──────► Result
Operand B│                             │
────────►│                             │
         │   - Addition                │──────► Status Flags
Operation│   - Subtraction             │        (Z, C, N, V)
Control ─►│   - AND, OR, XOR, NOT      │
         │   - Shift, Rotate           │
         │   - Compare                 │
         └─────────────────────────────┘

Status Flags:
  Z (Zero):     Result is zero
  C (Carry):    Carry out from MSB
  N (Negative): Result is negative (MSB = 1)
  V (Overflow): Signed overflow occurred
  P (Parity):   Even/odd number of 1s
```

**ALU Operations:**
```
Arithmetic Operations:
  - ADD, SUB, MUL, DIV
  - INCREMENT, DECREMENT
  - NEGATE

Logical Operations:
  - AND, OR, XOR, NOT
  - NAND, NOR

Shift Operations:
  - Logical Shift Left (LSL):    0 ← [bits] ← 0
  - Logical Shift Right (LSR):   0 → [bits] → 0
  - Arithmetic Shift Right (ASR): Sign → [bits] → 0
  - Rotate Left (ROL):           ← [bits] ←┐
                                           └──┘
  - Rotate Right (ROR):          ┌→ [bits] →
                                 └───────────┘

Comparison Operations:
  - CMP (Compare)
  - TEST (Bit test)
```

**Example: 4-bit ALU Block Diagram**
```
  A3 A2 A1 A0        B3 B2 B1 B0
   │  │  │  │         │  │  │  │
   ▼  ▼  ▼  ▼         ▼  ▼  ▼  ▼
  ┌───────────────────────────────┐
  │  Cin                          │
  │   │                           │
  │   ▼                           │
  │  [Full Adder 0] ──► S0, Cout  │
  │   Cin↑                        │
  │   ▼                           │
  │  [Full Adder 1] ──► S1, Cout  │
  │   Cin↑                        │
  │   ▼                           │
  │  [Full Adder 2] ──► S2, Cout  │
  │   Cin↑                        │
  │   ▼                           │
  │  [Full Adder 3] ──► S3, Cout  │
  │                               │
  │  [Logic Unit]                 │
  │  - AND, OR, XOR               │
  │                               │
  │  [Multiplexer]                │
  │  (Select Operation)           │
  └───────────────────────────────┘
       │  │  │  │
       ▼  ▼  ▼  ▼
      S3 S2 S1 S0 (Result)
       + Status Flags
```

### 2.2.2 Control Unit (CU)

The Control Unit coordinates and controls all CPU operations.

```
         ┌────────────────────────────────┐
         │        Control Unit            │
         │                                │
Clock ──►│  ┌──────────────────┐          │
         │  │ Instruction      │          │
IR ─────►│  │ Decoder          │          │
         │  └────────┬─────────┘          │
         │           │                    │
Status ─►│  ┌────────▼─────────┐          │
Flags    │  │  Control Logic   │          │
         │  │  (FSM/Microprog) │          │
External │  └────────┬─────────┘          │
Signals─►│           │                    │
         │           ▼                    │
         │  ┌─────────────────┐           │
         │  │  Control Signal │           │
         │  │   Generation    │           │
         │  └────────┬────────┘           │
         └───────────┼────────────────────┘
                     │
                     ▼
          Control Signals to:
          - ALU operations
          - Register transfers
          - Memory read/write
          - I/O operations
          - Bus control

Control Unit Types:

1. Hardwired Control:
   ┌──────────────┐
   │  Instruction │
   │   Decoder    │
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ Combinational│
   │    Logic     │
   └──────┬───────┘
          ▼
     Control Signals
   
   Advantages: Fast
   Disadvantages: Difficult to modify

2. Microprogrammed Control:
   ┌──────────────┐
   │ Instruction  │──► Address
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ Control Store│
   │ (Microcode)  │
   └──────┬───────┘
          ▼
     Control Signals
   
   Advantages: Flexible, easy to modify
   Disadvantages: Slower
```

### 2.2.3 Registers

Registers are the fastest storage locations in the CPU.

```
Register Classification:

1. User-Visible Registers:
   ┌─────────────────────────────┐
   │  General Purpose Registers  │
   │  R0, R1, R2, ..., Rn        │
   │  (Data operations)          │
   └─────────────────────────────┘
   
   ┌─────────────────────────────┐
   │  Data Registers             │
   │  (Hold operands, results)   │
   └─────────────────────────────┘
   
   ┌─────────────────────────────┐
   │  Address Registers          │
   │  (Hold memory addresses)    │
   └─────────────────────────────┘
   
   ┌─────────────────────────────┐
   │  Index Registers            │
   │  (Array indexing)           │
   └─────────────────────────────┘
   
   ┌─────────────────────────────┐
   │  Stack Pointer (SP)         │
   │  (Points to top of stack)   │
   └─────────────────────────────┘

2. Control & Status Registers:
   ┌─────────────────────────────┐
   │  Program Counter (PC)       │
   │  (Address of next inst.)    │
   └─────────────────────────────┘
   
   ┌─────────────────────────────┐
   │  Instruction Register (IR)  │
   │  (Holds current inst.)      │
   └─────────────────────────────┘
   
   ┌─────────────────────────────┐
   │  Memory Address Register    │
   │  (MAR) - Address for memory │
   └─────────────────────────────┘
   
   ┌─────────────────────────────┐
   │  Memory Data Register (MDR) │
   │  (Data to/from memory)      │
   └─────────────────────────────┘
   
   ┌─────────────────────────────┐
   │  Program Status Word (PSW)  │
   │  (Flags, mode bits, etc.)   │
   └─────────────────────────────┘
```

**Register Organization Example (32-bit CPU):**
```
┌───────────────────────────────────┐
│ R0  │  General Purpose Register 0 │ 32 bits
├───────────────────────────────────┤
│ R1  │  General Purpose Register 1 │ 32 bits
├───────────────────────────────────┤
│ R2  │  General Purpose Register 2 │ 32 bits
├───────────────────────────────────┤
│ ... │                             │
├───────────────────────────────────┤
│ R15 │  General Purpose Register15 │ 32 bits
├───────────────────────────────────┤
│ SP  │  Stack Pointer              │ 32 bits
├───────────────────────────────────┤
│ LR  │  Link Register (ret addr)   │ 32 bits
├───────────────────────────────────┤
│ PC  │  Program Counter            │ 32 bits
├───────────────────────────────────┤
│ PSR │  Program Status Register    │ 32 bits
│     │  N Z C V ...                │
└───────────────────────────────────┘
```

## 2.3 Instruction Cycle

The instruction cycle (fetch-decode-execute cycle) is the basic operation of the CPU.

```
┌─────────────────────────────────────────────────┐
│           INSTRUCTION CYCLE                     │
│                                                 │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐  │
│  │  FETCH  │─────►│ DECODE  │─────►│ EXECUTE │  │
│  └────┬────┘      └─────────┘      └────┬────┘  │
│       │                                 │       │
│       └─────────────────────────────────┘       │
│              (Repeat for next instruction)      │
└─────────────────────────────────────────────────┘

Detailed Steps:

1. FETCH Phase:
   ┌────────────────────────────────┐
   │ a. MAR ← PC                    │
   │ b. MDR ← Memory[MAR]           │
   │ c. IR ← MDR                    │
   │ d. PC ← PC + instruction_size  │
   └────────────────────────────────┘

2. DECODE Phase:
   ┌────────────────────────────────┐
   │ a. Decode opcode in IR         │
   │ b. Identify operands           │
   │ c. Read operands if needed     │
   │ d. Generate control signals    │
   └────────────────────────────────┘

3. EXECUTE Phase:
   ┌────────────────────────────────┐
   │ a. Perform operation (ALU)     │
   │ b. Store result                │
   │ c. Update flags                │
   │ d. Handle interrupts (if any)  │
   └────────────────────────────────┘

Optional Phases:

4. MEMORY ACCESS (for load/store):
   ┌────────────────────────────────┐
   │ - Read from or write to memory │
   └────────────────────────────────┘

5. WRITE BACK:
   ┌────────────────────────────────┐
   │ - Write result to register     │
   └────────────────────────────────┘
```

**Timing Diagram:**
```
Clock  : ___┌───┐___┌───┐___┌───┐___┌───┐___┌───┐___
         
PC     : ──<100>──<104>──<104>──<104>──<104>───
                    (incremented)

MAR    : ────────<100>─────────────────────────
         
MDR    : ────────────<INST>────────────────────
         
IR     : ──────────────<INST>──────────────────
         
Phase  : [  FETCH  ][DECODE][EXECUTE][ FETCH...]
```

## 2.4 Instruction Format

Instructions are encoded in various formats depending on the architecture.

### 2.4.1 Common Instruction Formats

```
1. Three-Address Format:
┌────────┬────────┬────────┬────────┐
│ Opcode │  Src1  │  Src2  │  Dest  │
└────────┴────────┴────────┴────────┘
Example: ADD R1, R2, R3  (R1 = R2 + R3)

2. Two-Address Format:
┌────────┬────────┬────────┐
│ Opcode │  Src   │ Src/Dst│
└────────┴────────┴────────┘
Example: ADD R1, R2  (R1 = R1 + R2)

3. One-Address Format (Accumulator):
┌────────┬────────┐
│ Opcode │  Src   │
└────────┴────────┘
Example: ADD R1  (ACC = ACC + R1)

4. Zero-Address Format (Stack):
┌────────┐
│ Opcode │
└────────┘
Example: ADD  (Pop two, add, push result)
```

### 2.4.2 Instruction Components

```
Typical Instruction Format (32-bit):
┌──────┬──────┬──────┬──────┬─────────────┐
│Opcode│  Rd  │  Rs  │  Rt  │  Function   │
│ 6bit │ 5bit │ 5bit │ 5bit │   11bit     │
└──────┴──────┴──────┴──────┴─────────────┘

Fields:
- Opcode: Operation code (what to do)
- Rd: Destination register
- Rs: Source register 1
- Rt: Source register 2
- Function: Specifies variant of operation
- Immediate: Constant value (in some formats)
- Address: Memory address (in some formats)

Example Instruction Encodings:

R-Type (Register):
┌──────┬────┬────┬────┬─────┬────────┐
│OP(6) │Rs  │Rt  │Rd  │Shamt│Funct(6)│
└──────┴────┴────┴────┴─────┴────────┘
ADD R1, R2, R3

I-Type (Immediate):
┌──────┬────┬────┬──────────────────┐
│OP(6) │Rs  │Rt  │   Immediate(16)  │
└──────┴────┴────┴──────────────────┘
ADDI R1, R2, #100

J-Type (Jump):
┌──────┬──────────────────────────┐
│OP(6) │      Address(26)         │
└──────┴──────────────────────────┘
JMP 0x1000
```

## 2.5 CPU Performance Metrics

### 2.5.1 Clock Speed
```
Clock Cycle Time = 1 / Clock Frequency

Examples:
  3.0 GHz CPU: Clock cycle = 1/3×10⁹ = 0.333 ns
  2.5 GHz CPU: Clock cycle = 1/2.5×10⁹ = 0.4 ns

Clock Signal:
     ┌─┐   ┌─┐   ┌─┐   ┌─┐
     │ │   │ │   │ │   │ │
  ───┘ └───┘ └───┘ └───┘ └───
     ├─────┤
   Clock Period
```

### 2.5.2 CPU Time
```
CPU Time = Instruction Count × CPI × Clock Cycle Time

Where:
  CPI = Cycles Per Instruction

Example:
  Program: 1 million instructions
  CPI: 2.0
  Clock: 2 GHz (0.5 ns cycle)
  
  CPU Time = 10⁶ × 2 × 0.5×10⁻⁹
           = 1 ms
```

### 2.5.3 MIPS (Million Instructions Per Second)
```
MIPS = Instruction Count / (Execution Time × 10⁶)

Or:

MIPS = Clock Rate (MHz) / CPI

Example:
  3 GHz CPU with CPI = 2
  MIPS = 3000 / 2 = 1500 MIPS
```

### 2.5.4 Speedup
```
Speedup = Execution Time (old) / Execution Time (new)

Example:
  Old CPU: 10 seconds
  New CPU: 2 seconds
  Speedup = 10/2 = 5× faster
```

## 2.6 CPU Organization Models

### 2.6.1 Accumulator-Based
```
┌──────────────────┐
│   Accumulator    │ (Special register for ALU operations)
└────────┬─────────┘
         │
         ▼
    ┌────────┐
    │  ALU   │
    └────────┘

Example: ADD X
  ACC ← ACC + Memory[X]

Advantages:
  - Simple instruction format
  - Less hardware complexity

Disadvantages:
  - High memory traffic
  - Accumulator is bottleneck
```

### 2.6.2 Register-Based (General Purpose Registers)
```
┌──────┬──────┬──────┬──────┐
│  R0  │  R1  │  R2  │  ... │
└───┬──┴───┬──┴───┬──┴──────┘
    │      │      │
    └──────┼──────┘
           ▼
       ┌────────┐
       │  ALU   │
       └────────┘

Example: ADD R1, R2, R3
  R1 ← R2 + R3

Advantages:
  - Faster (fewer memory accesses)
  - More flexible
  - Better compiler optimization

Disadvantages:
  - More complex instruction encoding
  - More hardware (register file)

Modern CPUs typically have 16-32 GPRs
```

### 2.6.3 Stack-Based
```
     Stack
    ┌──────┐  ← SP (Stack Pointer)
    │  10  │  Top
    ├──────┤
    │  20  │
    ├──────┤
    │  30  │
    └──────┘

Example: ADD
  1. Pop 10
  2. Pop 20
  3. Compute 10 + 20 = 30
  4. Push 30

Advantages:
  - Very compact instructions
  - Simple to implement

Disadvantages:
  - Stack is bottleneck
  - Not efficient for all operations

Used in: JVM, PostScript, Forth
```

## 2.7 Datapath

The datapath is the collection of functional units that perform data processing.

```
┌────────────────────────────────────────────────────┐
│                    DATAPATH                        │
│                                                    │
│  ┌──────────┐                                      │
│  │   PC     │──────┐                               │
│  └────┬─────┘      │                               │
│       │            ▼                               │
│       │     ┌─────────────┐                        │
│       │     │ Instruction │                        │
│       │     │   Memory    │                        │
│       │     └──────┬──────┘                        │
│       │            │                               │
│       │            ▼                               │
│       │     ┌─────────────┐                        │
│       │     │     IR      │                        │
│       │     └──────┬──────┘                        │
│       │            │                               │
│       │            ▼                               │
│       │     ┌─────────────┐     ┌──────────┐       │
│       │     │  Decoder &  │────►│ Control  │       │
│       │     │   Control   │     │ Signals  │       │
│       │     └─────────────┘     └──────────┘       │
│       │                                            │
│       │     ┌─────────────┐                        │
│       └────►│   Adder     │ (PC + 4)               │
│             │   (+4)      │                        │
│             └──────┬──────┘                        │
│                    │                               │
│             ┌──────┴──────┐                        │
│             │             │                        │
│             ▼             ▼                        │
│      ┌───────────┐  ┌─────────┐                    │
│      │ Register  │  │         │                    │
│      │   File    │  │   ALU   │                    │
│      │ (R0-R31)  │  │         │                    │
│      └─────┬─────┘  └────┬────┘                    │
│            │             │                         │
│            ▼             ▼                         │
│      ┌──────────────────────┐                      │
│      │   Data Memory        │                      │
│      └──────────────────────┘                      │
└────────────────────────────────────────────────────┘

Signal Flow:
1. PC → Instruction Memory → IR
2. IR → Control Unit → Control Signals
3. Registers → ALU → Result
4. Result → Registers or Memory
```

## 2.8 Single-Cycle vs Multi-Cycle CPU

### 2.8.1 Single-Cycle CPU
```
Each instruction completes in ONE clock cycle

Timing:
Clock: ┌──────────────────────┐
       │   One Instruction    │
       └──────────────────────┘
       │ F │ D │ E │ M │ WB   │
       └──────────────────────┘

Advantages:
  - Simple design
  - Easy to understand

Disadvantages:
  - Clock cycle = longest instruction
  - Inefficient (fast instructions wait)
  - Large amount of hardware duplication

Example:
  If longest instruction = 800ps
  All instructions take 800ps
  Even if ADD only needs 400ps
```

### 2.8.2 Multi-Cycle CPU
```
Instructions take multiple cycles

Timing:
         Cycle 1  Cycle 2  Cycle 3  Cycle 4
LOAD:   [Fetch] [Decode] [Execute] [Memory] [Write]
ADD:    [Fetch] [Decode] [Execute] [Write]
JUMP:   [Fetch] [Decode] [Execute]

Advantages:
  - Faster clock
  - Better resource utilization
  - Different instructions take different times

Disadvantages:
  - More complex control
  - Need additional registers between stages

Example:
  Clock cycle = 200ps (shortest step)
  LOAD: 5 × 200ps = 1000ps
  ADD:  4 × 200ps = 800ps
  JUMP: 3 × 200ps = 600ps
```

## 2.9 Interrupts and Exception Handling

```
┌─────────────────────────────────────────────┐
│         Interrupt Mechanism                 │
│                                             │
│  Normal Execution:                          │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐                │
│  │ I1 │→│ I2 │→│ I3 │→│ I4 │→...            │
│  └────┘ └────┘ └────┘ └────┘                │
│                   ↓ Interrupt!              │
│                   ▼                         │
│  ┌────────────────────────────┐             │
│  │   Save PC and Status       │             │
│  └────────────┬───────────────┘             │
│               ▼                             │
│  ┌────────────────────────────┐             │
│  │   Load Interrupt Vector    │             │
│  └────────────┬───────────────┘             │
│               ▼                             │
│  ┌────┐ ┌────┐ ┌────┐                       │
│  │ISR1│→│ISR2│→│IRET│                       │
│  └────┘ └────┘ └──┬─┘                       │
│                   │                         │
│                   ▼ Return                  │
│  ┌────┐ ┌────┐ ┌────┐                       │
│  │ I4 │→│ I5 │→│ I6 │→...                   │
│  └────┘ └────┘ └────┘                       │
└─────────────────────────────────────────────┘

Interrupt Types:

1. Hardware Interrupts:
   - Maskable (can be disabled)
   - Non-maskable (cannot be disabled)
   - Sources: I/O devices, timers

2. Software Interrupts:
   - System calls
   - Traps
   - Explicitly triggered by software

3. Exceptions:
   - Divide by zero
   - Invalid opcode
   - Page fault
   - Overflow

Interrupt Priority:
┌─────────────────────┬──────────┐
│      Type           │ Priority │
├─────────────────────┼──────────┤
│ Reset               │ Highest  │
│ Machine Check       │    ↑     │
│ NMI                 │    │     │
│ Hardware Interrupts │    │     │
│ Software Interrupts │    ↓     │
│ Traps               │ Lowest   │
└─────────────────────┴──────────┘

Interrupt Vector Table:
Memory Address    Handler Address
┌────────────┬─────────────────┐
│ 0x0000     │  Reset Handler  │
├────────────┼─────────────────┤
│ 0x0004     │  IRQ 1 Handler  │
├────────────┼─────────────────┤
│ 0x0008     │  IRQ 2 Handler  │
├────────────┼─────────────────┤
│ ...        │  ...            │
└────────────┴─────────────────┘
```

## 2.10 Modern CPU Features

### 2.10.1 Branch Prediction
```
┌────────────────────────────────────┐
│      Branch Prediction             │
│                                    │
│  IF (condition) THEN               │
│      Path A  ← Predicted           │
│  ELSE                              │
│      Path B                        │
│                                    │
│  Prediction Accuracy: ~95%         │
│                                    │
│  Types:                            │
│  - Static: Always predict taken    │
│  - Dynamic: Use branch history     │
│  - Two-level adaptive              │
└────────────────────────────────────┘

Branch History Table (BHT):
┌──────────┬───────────────┐
│  PC      │  Prediction   │
├──────────┼───────────────┤
│ 0x1000   │ Strongly Taken│
│ 0x1010   │ Weakly Taken  │
│ 0x1020   │ Not Taken     │
└──────────┴───────────────┘

2-bit Saturating Counter:
     ┌─────────────────┐
     │  Strongly Taken │ ←──┐
     └────┬────────────┘    │
   Taken  │  Not Taken      │ Taken
          ▼                 │
     ┌─────────────────┐    │
     │  Weakly Taken   │ ───┘
     └────┬────────────┘
          │  Not Taken
          ▼
     ┌─────────────────┐
     │ Weakly Not Taken│
     └────┬────────────┘
          │  Taken
          ▼
     ┌─────────────────┐
     │Strongly Not Taken
     └─────────────────┘
```

### 2.10.2 Out-of-Order Execution
```
Program Order:        Execution Order:
┌────────┐            ┌────────┐
│ I1     │            │ I1     │
├────────┤            ├────────┤
│ I2     │ (depends   │ I3     │ (independent)
├────────┤  on I1)    ├────────┤
│ I3     │            │ I4     │ (independent)
├────────┤            ├────────┤
│ I4     │            │ I2     │ (now ready)
└────────┘            └────────┘

Components:
- Reservation Stations
- Reorder Buffer (ROB)
- Register Renaming
```

### 2.10.3 Speculative Execution
```
Execute instructions before knowing if needed

┌────────────────────────────────┐
│  Branch Instruction            │
└──────┬────────────────┬────────┘
       │                │
   ┌───▼───┐        ┌───▼───┐
   │Path A │        │Path B │
   │Execute│        │Execute│
   │Speculat.       │Speculat.
   └───────┘        └───────┘
       │                │
       └────────┬───────┘
                ▼
         Commit correct
         Discard wrong
```

---

**Key Takeaways:**
1. CPU consists of ALU, Control Unit, and Registers
2. Instruction cycle: Fetch → Decode → Execute
3. Performance depends on clock speed, CPI, and instruction count
4. Modern CPUs use pipelining, branch prediction, and OOO execution
5. Interrupts allow CPU to respond to external events

**Next:** [Memory Hierarchy](./03-memory-hierarchy.md)


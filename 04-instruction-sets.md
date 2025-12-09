# Chapter 4: Instruction Set Architecture (ISA)

## 4.1 ISA Overview

The Instruction Set Architecture is the interface between hardware and software.

```
┌───────────────────────────────────────────┐
│          Software Stack                   │
│                                           │
│  ┌──────────────────────────────┐         │
│  │      Applications            │         │
│  └───────────┬──────────────────┘         │
│              ▼                            │
│  ┌──────────────────────────────┐         │
│  │   Operating System           │         │
│  └───────────┬──────────────────┘         │
│              ▼                            │
│  ┌──────────────────────────────┐         │
│  │   Compiler/Assembler         │         │
│  └───────────┬──────────────────┘         │
└──────────────┼────────────────────────────┘
               ▼
╔══════════════════════════════════════════╗
║    INSTRUCTION SET ARCHITECTURE (ISA)    ║
║  - Instructions                          ║
║  - Registers                             ║
║  - Addressing Modes                      ║
║  - Data Types                            ║
║  - Memory Model                          ║
╚══════════════┬═══════════════════════════╝
               ▼
┌────────────────────────────────────────────┐
│         Hardware Implementation            │
│  ┌──────────────────────────────┐          │
│  │     Microarchitecture        │          │
│  │  (Pipeline, Cache, etc.)     │          │
│  └──────────────────────────────┘          │
└────────────────────────────────────────────┘

ISA defines:
1. What operations the CPU can perform
2. How to specify operands
3. How instructions are encoded
4. How CPU state is organized
```

## 4.2 ISA Classifications

### 4.2.1 CISC (Complex Instruction Set Computer)

```
Characteristics:
- Many instructions (100s)
- Variable instruction length
- Multiple addressing modes
- Complex instructions
- Fewer lines of assembly code
- More work per instruction

Examples: x86, x86-64 (Intel, AMD), VAX

┌────────────────────────────────────┐
│   CISC Instruction Examples        │
│                                    │
│  MOVS: Move string                 │
│  LOOP: Decrement and branch        │
│  ENTER: Make stack frame           │
│  PUSHA: Push all registers         │
│                                    │
│  Single instruction can:           │
│  - Access memory multiple times    │
│  - Perform complex operations      │
│  - Update multiple registers       │
└────────────────────────────────────┘

Example: x86 instruction lengths
┌────────────────┬─────────┐
│  Instruction   │  Bytes  │
├────────────────┼─────────┤
│  NOP           │    1    │
│  MOV AX, BX    │    2    │
│  MOV EAX, [mem]│   5-6   │
│  Complex inst. │  up to  │
│                │   15    │
└────────────────┴─────────┘

Advantages:
  + Compact code
  + Rich instruction set
  + Good code density
  + Backward compatibility

Disadvantages:
  - Complex hardware
  - Variable decode time
  - Hard to pipeline
  - Power consumption
```

### 4.2.2 RISC (Reduced Instruction Set Computer)

```
Characteristics:
- Few, simple instructions
- Fixed instruction length
- Load/store architecture
- Limited addressing modes
- One operation per instruction
- More assembly instructions
- Designed for pipelining

Examples: ARM, MIPS, RISC-V, PowerPC

┌────────────────────────────────────┐
│   RISC Philosophy                  │
│                                    │
│  1. Simple instructions            │
│  2. Fixed-length (32-bit typical)  │
│  3. Register-oriented              │
│  4. Single cycle (if possible)     │
│  5. Load/Store only for memory     │
│  6. Efficient pipelining           │
└────────────────────────────────────┘

RISC Instruction Format (ARM):
┌──────┬────┬────┬────┬────┬────┐
│Cond  │Op  │Rn  │Rd  │Op2 │Imm │
│(4)   │(4) │(4) │(4) │(12)│... │
└──────┴────┴────┴────┴────┴────┘
All instructions: 32 bits

Example: String copy (CISC vs RISC)

CISC (x86):
  REP MOVSB    ; One instruction!

RISC (ARM):
  loop:
    LDRB r1, [r2], #1   ; Load byte, post-increment
    STRB r1, [r3], #1   ; Store byte, post-increment
    SUBS r4, r4, #1     ; Decrement counter
    BNE loop            ; Branch if not zero

More instructions, but each is simpler and faster

Advantages:
  + Simple, fast instructions
  + Easy to pipeline
  + Lower power
  + Regular instruction format
  + Easier compiler design

Disadvantages:
  - More instructions needed
  - Larger code size
  - More memory bandwidth
```

### 4.2.3 Comparison Table

```
┌──────────────────┬─────────────┬─────────────┐
│   Feature        │    CISC     │    RISC     │
├──────────────────┼─────────────┼─────────────┤
│ Instructions     │   Many      │    Few      │
│ Instruction Size │  Variable   │   Fixed     │
│ Addressing Modes │   Many      │   Few       │
│ Execution Time   │  Variable   │  Uniform    │
│ Code Density     │   High      │   Lower     │
│ Pipeline         │  Complex    │   Simple    │
│ Register Count   │   Fewer     │   More      │
│ Memory Access    │  Any inst.  │ Load/Store  │
│ Decode           │  Complex    │   Simple    │
│ Microcode        │   Common    │   Rare      │
└──────────────────┴─────────────┴─────────────┘

Modern Reality:
- Distinction blurred
- x86 (CISC) uses RISC-like micro-ops internally
- ARM (RISC) has SIMD and complex features
- Hybrid approaches dominate
```

## 4.3 Addressing Modes

Methods to specify operand location.

### 4.3.1 Immediate Addressing
```
Operand is part of the instruction

┌──────────┬──────────┐
│  Opcode  │   Value  │
└──────────┴──────────┘

Example:
  MOV R1, #5      ; R1 = 5
  ADD R2, R2, #1  ; R2 = R2 + 1

┌──────────────────┐
│ Instruction      │
│  ┌────┬────┐    │
│  │ADD │ #5 │    │
│  └────┴──┬─┘    │
│         │       │
│         └──► 5  │ (Immediate value)
└──────────────────┘

Advantages:
  + Fast (no memory access)
  + Simple
Disadvantages:
  - Limited range (instruction size)
  - Can't modify value at runtime
```

### 4.3.2 Register Addressing (Direct)
```
Operand is in a register

Example:
  MOV R1, R2      ; R1 = R2
  ADD R3, R1, R2  ; R3 = R1 + R2

┌──────────┐
│Instruction│
│  ┌─────┐ │
│  │ ADD │ │
│  └──┬──┘ │
│     │    │
│     ▼    │
│  [R1] ──► Value in R1
└──────────┘

Advantages:
  + Very fast (on-chip)
  + No address calculation
Disadvantages:
  - Limited number of registers
```

### 4.3.3 Direct (Absolute) Addressing
```
Address is part of instruction

┌──────────┬──────────┐
│  Opcode  │ Address  │
└──────────┴──────────┘

Example:
  LOAD R1, [1000]  ; R1 = Memory[1000]
  STORE R2, [2000] ; Memory[2000] = R2

┌────────────────────┐
│   Instruction      │
│    ┌────────┐     │
│    │ LOAD   │     │
│    │ [1000] │     │
│    └────┬───┘     │
│         │         │
│         ▼         │
│   Memory[1000] ──► Value
└────────────────────┘

Advantages:
  + Simple
Disadvantages:
  - Limited address range
  - Not relocatable
```

### 4.3.4 Indirect Addressing
```
Address points to another address

Example:
  LOAD R1, (R2)    ; R1 = Memory[Memory[R2]]
  LOAD R1, @1000   ; R1 = Memory[Memory[1000]]

┌──────────────────────────┐
│      Instruction         │
│       ┌──────┐           │
│       │ LOAD │           │
│       │ (R2) │           │
│       └───┬──┘           │
│           │              │
│           ▼              │
│       R2: 1000           │
│           │              │
│           ▼              │
│    Memory[1000]: 5000    │
│           │              │
│           ▼              │
│    Memory[5000] ──► Value│
└──────────────────────────┘

Used for:
  - Pointers
  - Dynamic data structures
  - Function pointers

Advantages:
  + Flexible
  + Supports pointers
Disadvantages:
  - Slow (multiple memory accesses)
```

### 4.3.5 Register Indirect Addressing
```
Register contains address

Example:
  LOAD R1, [R2]    ; R1 = Memory[R2]
  STORE R3, [R4]   ; Memory[R4] = R3

┌─────────────────┐
│   Instruction   │
│    ┌──────┐     │
│    │ LOAD │     │
│    │ [R2] │     │
│    └───┬──┘     │
│        │        │
│        ▼        │
│    R2: 1000     │
│        │        │
│        ▼        │
│ Memory[1000] ──► Value
└─────────────────┘

Common in RISC architectures

Advantages:
  + Fast (register access)
  + Flexible addressing
  + Good for arrays/pointers
```

### 4.3.6 Indexed Addressing
```
Address = Base + Index

Example:
  LOAD R1, [R2 + R3]      ; R1 = Memory[R2 + R3]
  LOAD R1, array[R2]      ; R1 = Memory[array + R2]

┌──────────────────────────┐
│     Instruction          │
│      ┌──────────┐        │
│      │   LOAD   │        │
│      │ [R2 + R3]│        │
│      └──┬───┬───┘        │
│         │   │            │
│    R2:1000  R3:8         │
│         │   │            │
│         └───┼───► 1008   │
│             │            │
│             ▼            │
│      Memory[1008] ──► Value
└──────────────────────────┘

Perfect for arrays:
  array[i] = base_address + i * element_size

Advantages:
  + Excellent for arrays
  + Flexible
Disadvantages:
  - Requires addition
```

### 4.3.7 Base + Displacement Addressing
```
Address = Base Register + Offset

Example:
  LOAD R1, 8[R2]        ; R1 = Memory[R2 + 8]
  LOAD R1, [R2 + #100]  ; R1 = Memory[R2 + 100]

┌────────────────────────┐
│    Instruction         │
│     ┌────────┐         │
│     │  LOAD  │         │
│     │ 8[R2]  │         │
│     └──┬──┬──┘         │
│        │  │            │
│    R2:1000  +8         │
│        │  │            │
│        └──┼──► 1008    │
│           │            │
│           ▼            │
│    Memory[1008] ──► Value
└────────────────────────┘

Used for:
  - Structure/object field access
  - Stack frames (SP + offset)
  - Local variables

Example (C structure):
  struct Point { int x; int y; };
  Point p;
  
  p.x → [R_p + 0]
  p.y → [R_p + 4]

Advantages:
  + Good for structures
  + Efficient stack access
  + Common in modern CPUs
```

### 4.3.8 PC-Relative Addressing
```
Address = PC + Offset

Example:
  BRANCH +100    ; PC = PC + 100
  LOAD R1, [PC+8]; R1 = Memory[PC + 8]

┌────────────────────────┐
│    Instruction         │
│     ┌────────┐         │
│     │ BRANCH │         │
│     │  +100  │         │
│     └──┬──┬──┘         │
│        │  │            │
│   PC:1000  +100        │
│        │  │            │
│        └──┼──► 1100    │
│           │            │
│           ▼            │
│      New PC: 1100      │
└────────────────────────┘

Used for:
  - Branches/jumps
  - Position-independent code
  - Relative data access

Advantages:
  + Position-independent
  + Relocatable code
  + Efficient branching
```

### 4.3.9 Auto-increment/Auto-decrement
```
Register updated automatically

Pre-increment:  [++R]  (Increment, then use)
Post-increment: [R++]  (Use, then increment)
Pre-decrement:  [--R]  (Decrement, then use)
Post-decrement: [R--]  (Use, then decrement)

Example:
  LOAD R1, [R2++]  ; R1 = Memory[R2]; R2 = R2 + 4
  STORE R3, [--R4] ; R4 = R4 - 4; Memory[R4] = R3

Post-increment:
┌─────────────────────────┐
│  Initially: R2 = 1000   │
│                         │
│  LOAD R1, [R2++]        │
│    1. R1 = Mem[1000]    │
│    2. R2 = 1004         │
│                         │
│  Perfect for loops!     │
└─────────────────────────┘

Array traversal:
  loop:
    LOAD R1, [R2++]   ; Load and advance
    ADD R3, R3, R1    ; Sum
    CMP R2, R4        ; Check end
    BNE loop          ; Continue

Advantages:
  + Efficient loops
  + Less instructions
  + Good for stacks/queues

Common in: VAX, PDP-11, ARM
```

### 4.3.10 Summary Table

```
┌──────────────────┬─────────────────┬──────────────┐
│  Addressing Mode │     Example     │  Effective   │
│                  │                 │   Address    │
├──────────────────┼─────────────────┼──────────────┤
│ Immediate        │ ADD #5          │     N/A      │
│ Register         │ ADD R1          │     N/A      │
│ Direct           │ ADD [1000]      │    1000      │
│ Indirect         │ ADD @1000       │  Mem[1000]   │
│ Register Ind.    │ ADD [R1]        │     R1       │
│ Indexed          │ ADD [R1+R2]     │   R1 + R2    │
│ Base+Displ.      │ ADD 8[R1]       │   R1 + 8     │
│ PC-Relative      │ BRA +100        │   PC + 100   │
│ Auto-increment   │ ADD [R1++]      │ R1 (then R1++)│
│ Auto-decrement   │ ADD [--R1]      │ (R1-- then R1)│
└──────────────────┴─────────────────┴──────────────┘
```

## 4.4 Instruction Types

### 4.4.1 Data Movement
```
Transfer data between locations

┌────────────────────────────────────────┐
│  MOV   dest, src   ; dest = src        │
│  LOAD  reg, [mem]  ; reg = memory      │
│  STORE [mem], reg  ; memory = reg      │
│  PUSH  value       ; Stack push        │
│  POP   dest        ; Stack pop         │
│  XCHG  a, b        ; Swap a and b      │
└────────────────────────────────────────┘

Example: Data transfer
┌──────────────────┐
│  Registers       │     ┌──────────┐
│  ┌────┐          │     │  Memory  │
│  │ R1 │◄─────────┼─────┤  [1000]  │ LOAD
│  └────┘          │     └──────────┘
│  ┌────┐          │     ┌──────────┐
│  │ R2 │──────────┼────►│  [2000]  │ STORE
│  └────┘          │     └──────────┘
└──────────────────┘

Stack Operations:
   Before PUSH R1     After PUSH R1
     SP ──►┌────┐       ┌────┐
           │    │       │ R1 │◄── SP
           ├────┤       ├────┤
           │    │       │    │
           └────┘       └────┘
```

### 4.4.2 Arithmetic
```
Mathematical operations

┌────────────────────────────────────────┐
│  ADD   R1, R2, R3  ; R1 = R2 + R3      │
│  SUB   R1, R2, R3  ; R1 = R2 - R3      │
│  MUL   R1, R2, R3  ; R1 = R2 × R3      │
│  DIV   R1, R2, R3  ; R1 = R2 ÷ R3      │
│  INC   R1          ; R1 = R1 + 1       │
│  DEC   R1          ; R1 = R1 - 1       │
│  NEG   R1          ; R1 = -R1          │
│  ABS   R1          ; R1 = |R1|         │
└────────────────────────────────────────┘

Flags affected:
┌────┬──────────────────────────┐
│ Z  │ Zero (result = 0)        │
│ C  │ Carry (unsigned overflow)│
│ N  │ Negative (MSB = 1)       │
│ V  │ Overflow (signed)        │
└────┴──────────────────────────┘

Example: 8-bit addition
  1 0 1 1 0 1 1 0  (182)
+ 0 1 1 0 1 0 0 1  (105)
─────────────────
 10 0 0 1 1 1 1 1  (31, with carry)
 ↑               ↑
 C=1            V=1 (signed overflow)
```

### 4.4.3 Logical
```
Bit manipulation

┌────────────────────────────────────────┐
│  AND  R1, R2, R3  ; R1 = R2 & R3       │
│  OR   R1, R2, R3  ; R1 = R2 | R3       │
│  XOR  R1, R2, R3  ; R1 = R2 ^ R3       │
│  NOT  R1, R2      ; R1 = ~R2           │
│  TEST R1, R2      ; R1 & R2 (flags)    │
└────────────────────────────────────────┘

Example operations:
  AND: Masking
    1 0 1 1 0 1 0 0
  & 0 0 0 0 1 1 1 1  (mask)
  ─────────────────
    0 0 0 0 0 1 0 0  (extract bits)

  OR: Setting bits
    1 0 1 1 0 1 0 0
  | 0 0 0 0 1 0 0 0  (set bit 3)
  ─────────────────
    1 0 1 1 1 1 0 0

  XOR: Toggling bits
    1 0 1 1 0 1 0 0
  ^ 0 0 0 0 1 0 0 0  (toggle bit 3)
  ─────────────────
    1 0 1 1 1 1 0 0

  TEST: Check bits
    AND without storing result
    Only updates flags
```

### 4.4.4 Shift and Rotate
```
Bit shifting operations

┌────────────────────────────────────────┐
│  SHL  R1, n  ; Shift left logical      │
│  SHR  R1, n  ; Shift right logical     │
│  SAR  R1, n  ; Shift right arithmetic  │
│  ROL  R1, n  ; Rotate left             │
│  ROR  R1, n  ; Rotate right            │
│  RCL  R1, n  ; Rotate left thru carry  │
│  RCR  R1, n  ; Rotate right thru carry │
└────────────────────────────────────────┘

Shift Left Logical (SHL):
  Original: 0 1 0 1 1 0 1 0
  SHL 2:    1 0 1 1 0 1 0 0
            ◄─────────── 0 0
  Effect: Multiply by 2^n

Shift Right Logical (SHR):
  Original: 0 1 0 1 1 0 1 0
  SHR 2:    0 0 0 1 0 1 1 0
            0 0 ─────────►
  Effect: Divide by 2^n (unsigned)

Shift Right Arithmetic (SAR):
  Original: 1 0 1 1 0 1 0 0 (negative)
  SAR 2:    1 1 1 0 1 1 0 1
            1 1 ─────────► (sign extend)
  Effect: Divide by 2^n (signed)

Rotate Left (ROL):
  Original: 0 1 0 1 1 0 1 0
  ROL 2:    1 0 1 1 0 1 0 0
            ◄───────────┐ 1
                        └─► MSB to LSB

Rotate Right (ROR):
  Original: 0 1 0 1 1 0 1 0
  ROR 2:    1 0 0 1 0 1 1 0
              ┌───────────►
              └─► LSB to MSB
```

### 4.4.5 Control Flow
```
Branch and jump instructions

┌────────────────────────────────────────┐
│  Unconditional:                        │
│    JMP  addr       ; Jump              │
│    CALL addr       ; Function call     │
│    RET             ; Return            │
│                                        │
│  Conditional:                          │
│    JZ   addr       ; Jump if zero      │
│    JNZ  addr       ; Jump if not zero  │
│    JE   addr       ; Jump if equal     │
│    JNE  addr       ; Jump if not equal │
│    JG   addr       ; Jump if greater   │
│    JL   addr       ; Jump if less      │
│    JGE  addr       ; Jump if >= (signed)│
│    JLE  addr       ; Jump if <= (signed)│
│    JA   addr       ; Jump above (unsigned)│
│    JB   addr       ; Jump below (unsigned)│
└────────────────────────────────────────┘

Control Flow Diagram:
┌──────────────────────────────────┐
│  Instruction 1                   │
│  Instruction 2                   │
│  CMP R1, R2      ; Compare       │
│  JE  equal_label ; Branch if =   │
│                                  │
│  ; not equal path                │
│  Instruction 3                   │
│  JMP done        ; Skip          │
│                                  │
│ equal_label:                     │
│  ; equal path                    │
│  Instruction 4                   │
│                                  │
│ done:                            │
│  Instruction 5                   │
└──────────────────────────────────┘

Function Call Stack:
Before CALL:           After CALL:
┌────────┐            ┌────────┐
│  ...   │            │  ...   │
├────────┤            ├────────┤
│        │◄─ SP       │ Return │◄─ SP
└────────┘            │ Address│
                      └────────┘

After RET:
┌────────┐
│  ...   │
├────────┤
│        │◄─ SP (restored)
└────────┘
PC = Return Address
```

### 4.4.6 Comparison
```
Compare values and set flags

┌────────────────────────────────────────┐
│  CMP  R1, R2  ; Compare (R1 - R2)      │
│  TEST R1, R2  ; Bitwise test (R1 & R2) │
│  TST  R1      ; Test if zero           │
└────────────────────────────────────────┘

CMP operation:
  Performs subtraction but doesn't store result
  Only updates flags

Example:
  CMP R1, R2    ; Compare R1 with R2
  
  If R1 = 5, R2 = 3:
    5 - 3 = 2  (positive, non-zero)
    Flags: Z=0, N=0, C=0, V=0
    
  If R1 = 3, R2 = 5:
    3 - 5 = -2 (negative)
    Flags: Z=0, N=1, C=1, V=0
    
  If R1 = 5, R2 = 5:
    5 - 5 = 0  (zero)
    Flags: Z=1, N=0, C=0, V=0

Usage pattern:
  CMP R1, #10      ; Compare R1 with 10
  BGT greater      ; Branch if R1 > 10
  BEQ equal        ; Branch if R1 = 10
  BLT less         ; Branch if R1 < 10
```

## 4.5 Assembly Language Basics

### 4.5.1 Assembly Structure

```
; Comments start with semicolon
; or // or # depending on assembler

; ========== SECTIONS ==========

.data              ; Data section
  msg:   .ascii "Hello"   ; String
  value: .word 42         ; 32-bit integer
  array: .space 40        ; Reserve 40 bytes
  const: .equ 100         ; Constant

.bss               ; Uninitialized data
  buffer: .space 256

.text              ; Code section
  .global main     ; Export symbol

; ========== LABELS ==========
main:              ; Label (address marker)
  ; instructions here
  
loop_start:        ; Local label
  ; loop body
  branch loop_start

; ========== INSTRUCTIONS ==========
  MOV R0, #5       ; Load immediate
  ADD R1, R2, R3   ; Addition
  B loop_start     ; Branch

; ========== DIRECTIVES ==========
  .align 4         ; Align to 4-byte boundary
  .include "file.s"; Include another file
```

### 4.5.2 Example: ARM Assembly

```
; ARM assembly example: Sum array

.data
  array: .word 1, 2, 3, 4, 5
  size:  .word 5
  sum:   .word 0

.text
  .global _start

_start:
  LDR R0, =array    ; R0 = address of array
  LDR R1, size      ; R1 = size
  MOV R2, #0        ; R2 = sum = 0
  
loop:
  CMP R1, #0        ; Compare size with 0
  BEQ done          ; If size == 0, exit loop
  
  LDR R3, [R0]      ; R3 = *array
  ADD R2, R2, R3    ; sum += R3
  ADD R0, R0, #4    ; array++ (4 bytes per word)
  SUB R1, R1, #1    ; size--
  B loop            ; Repeat
  
done:
  LDR R0, =sum      ; R0 = address of sum
  STR R2, [R0]      ; *sum = R2
  
  ; Exit (system call)
  MOV R7, #1        ; syscall: exit
  SWI 0             ; Software interrupt
```

### 4.5.3 Example: MIPS Assembly

```
# MIPS assembly example: Fibonacci

.data
  n:      .word 10
  result: .word 0

.text
  .globl main

main:
  lw $t0, n        # Load n
  
  # Base cases
  li $v0, 0        # fib(0) = 0
  beq $t0, $zero, done
  
  li $v0, 1        # fib(1) = 1
  li $t1, 1
  beq $t0, $t1, done
  
  # fib(n) = fib(n-1) + fib(n-2)
  li $t1, 0        # fib(n-2)
  li $t2, 1        # fib(n-1)
  li $t3, 2        # counter = 2
  
fib_loop:
  bgt $t3, $t0, fib_done
  
  add $t4, $t1, $t2   # fib(n) = fib(n-2) + fib(n-1)
  move $t1, $t2       # fib(n-2) = fib(n-1)
  move $t2, $t4       # fib(n-1) = fib(n)
  
  addi $t3, $t3, 1    # counter++
  j fib_loop
  
fib_done:
  move $v0, $t2       # result = fib(n)
  
done:
  sw $v0, result      # Store result
  
  # Exit
  li $v0, 10          # syscall: exit
  syscall
```

### 4.5.4 Example: x86 Assembly (Intel Syntax)

```
; x86 assembly example: String length

section .data
  msg db 'Hello, World!', 0  ; Null-terminated string

section .bss
  length resd 1              ; Reserve 1 dword

section .text
  global _start

_start:
  mov esi, msg     ; ESI = pointer to string
  xor ecx, ecx     ; ECX = 0 (counter)
  
strlen_loop:
  mov al, [esi]    ; AL = *ESI (load byte)
  cmp al, 0        ; Compare with null
  je strlen_done   ; If null, done
  
  inc ecx          ; counter++
  inc esi          ; pointer++
  jmp strlen_loop  ; Repeat
  
strlen_done:
  mov [length], ecx ; Store length
  
  ; Exit
  mov eax, 1       ; syscall: exit
  xor ebx, ebx     ; status = 0
  int 0x80         ; System call
```

## 4.6 Instruction Encoding

### 4.6.1 Fixed-Length Encoding (RISC)

```
MIPS R-Type Instruction (32 bits):
┌──────┬────┬────┬────┬─────┬────────┐
│OP(6) │Rs  │Rt  │Rd  │Shamt│Funct(6)│
│      │(5) │(5) │(5) │ (5) │        │
└──────┴────┴────┴────┴─────┴────────┘
  6      5    5    5     5      6  bits

Example: ADD $t0, $t1, $t2
  OP    = 000000 (R-type)
  Rs    = 01001  ($t1 = register 9)
  Rt    = 01010  ($t2 = register 10)
  Rd    = 01000  ($t0 = register 8)
  Shamt = 00000  (no shift)
  Funct = 100000 (ADD function)

Binary: 000000 01001 01010 01000 00000 100000
Hex:    0x012A4020

MIPS I-Type (Immediate):
┌──────┬────┬────┬────────────────────┐
│OP(6) │Rs  │Rt  │   Immediate(16)    │
│      │(5) │(5) │                    │
└──────┴────┴────┴────────────────────┘

Example: ADDI $t0, $t1, 100
  OP   = 001000 (ADDI)
  Rs   = 01001  ($t1)
  Rt   = 01000  ($t0)
  Imm  = 0000000001100100 (100)

Binary: 001000 01001 01000 0000000001100100
Hex:    0x21280064
```

### 4.6.2 Variable-Length Encoding (CISC)

```
x86 Instruction Format (Variable: 1-15 bytes):

┌────────┬────────┬──────┬──────┬─────┬──────┬──────┐
│ Prefix │ Opcode │ModR/M│ SIB  │Displ│ Imm  │      │
│ 0-4B   │ 1-3B   │ 0-1B │ 0-1B │0-4B │ 0-4B │      │
└────────┴────────┴──────┴──────┴─────┴──────┴──────┘
Optional   Required  Optional fields

Examples:

1. NOP (No operation):
   Opcode: 0x90
   Length: 1 byte

2. MOV EAX, EBX:
   Opcode: 0x89
   ModR/M: 0xD8
   Length: 2 bytes
   
   Binary: 10001001 11011000
           │         ││ └─── BX (source)
           │         │└────── AX (dest)
           │         └─────── Register mode
           └─────────────────── MOV r/m32, r32

3. MOV EAX, [EBX + ECX*4 + 100]:
   Opcode: 0x8B
   ModR/M: 0x84
   SIB:    0x8B
   Disp:   0x64 0x00 0x00 0x00
   Length: 7 bytes

ModR/M byte:
┌───┬───┬────┐
│Mod│Reg│R/M │
│(2)│(3)│(3) │
└───┴───┴────┘

Mod: Addressing mode
  00: [reg]
  01: [reg + disp8]
  10: [reg + disp32]
  11: register direct

SIB (Scale-Index-Base) byte:
┌─────┬─────┬────┐
│Scale│Index│Base│
│ (2) │ (3) │(3) │
└─────┴─────┴────┘

Scale: multiply factor (1, 2, 4, 8)
Index: index register
Base:  base register
```

### 4.6.3 Encoding Comparison

```
┌───────────────┬────────┬───────────┬──────────┐
│  Instruction  │ Arch.  │  Encoding │  Bytes   │
├───────────────┼────────┼───────────┼──────────┤
│ ADD R1, R2    │  MIPS  │ Fixed     │    4     │
│               │  ARM   │ Fixed     │    4     │
│               │  x86   │ Variable  │   2-3    │
├───────────────┼────────┼───────────┼──────────┤
│ NOP           │  MIPS  │ Fixed     │    4     │
│               │  ARM   │ Fixed     │    4     │
│               │  x86   │ Variable  │    1     │
├───────────────┼────────┼───────────┼──────────┤
│ Complex mem   │  MIPS  │ Multiple  │  12-16   │
│ addressing    │  ARM   │ Multiple  │   8-12   │
│               │  x86   │ Single    │   6-7    │
└───────────────┴────────┴───────────┴──────────┘

Trade-offs:

Fixed-Length:
  + Simple decode
  + Easy pipeline
  + Predictable
  - Wastes space (simple instructions)
  - May need multiple instructions

Variable-Length:
  + Compact code
  + Flexible
  + Better code density
  - Complex decode
  - Harder to pipeline
  - Alignment issues
```

---

**Key Takeaways:**
1. ISA defines the programmer-visible interface
2. CISC vs RISC represent different design philosophies
3. Addressing modes provide flexibility in operand specification
4. Instruction types cover data movement, arithmetic, logic, and control
5. Assembly language is human-readable machine code
6. Instruction encoding affects code density and decode complexity

**Next:** [Pipelining](./05-pipelining.md)


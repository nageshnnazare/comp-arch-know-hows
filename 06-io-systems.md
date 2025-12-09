# Chapter 6: I/O Systems

## 6.1 I/O Organization

Input/Output systems connect the CPU to external devices.

```
┌────────────────────────────────────────────┐
│              System Overview               │
│                                            │
│  ┌──────┐      ┌────────┐                  │
│  │ CPU  │◄────►│ Memory │                  │
│  └───┬──┘      └────────┘                  │
│      │                                     │
│      └─────────┬─────────────────┐         │
│                ▼                 ▼         │
│        ┌──────────────┐  ┌──────────────┐  │
│        │I/O Controller│  │I/O Controller│  │
│        └──────┬───────┘  └──────┬───────┘  │
│               │                 │          │
│               ▼                 ▼          │
│         ┌─────────┐       ┌─────────┐      │
│         │ Device  │       │ Device  │      │
│         │(Disk,   │       │(Network,│      │
│         │Keyboard)│       │ Display)│      │
│         └─────────┘       └─────────┘      │
└────────────────────────────────────────────┘

I/O Device Characteristics:
┌──────────────┬──────────┬─────────┬──────────┐
│   Device     │  Speed   │  Unit   │  Usage   │
├──────────────┼──────────┼─────────┼──────────┤
│ Keyboard     │ 10 B/s   │  Byte   │  Input   │
│ Mouse        │ 100 B/s  │  Byte   │  Input   │
│ Display      │ 30 MB/s  │  Pixel  │  Output  │
│ Network (1G) │ 125 MB/s │  Packet │  Both    │
│ SSD          │ 500 MB/s │  Block  │  Both    │
│ HDD          │ 100 MB/s │  Block  │  Both    │
│ USB 3.0      │ 500 MB/s │  Byte   │  Both    │
└──────────────┴──────────┴─────────┴──────────┘

Speed Range: 6 orders of magnitude!
```

## 6.2 I/O Communication Methods

### 6.2.1 Programmed I/O (Polling)

CPU repeatedly checks device status.

```
┌────────────────────────────────────────┐
│  Programmed I/O Flow                   │
│                                        │
│  ┌──────────────────┐                  │
│  │ CPU checks       │                  │
│  │ device status    │                  │
│  └────────┬─────────┘                  │
│           │                            │
│           ▼                            │
│      ┌─────────┐                       │
│      │ Ready?  │                       │
│      └────┬────┘                       │
│      No   │   Yes                      │
│     ┌─────┼─────┐                      │
│     ▼     │     ▼                      │
│  [Wait]   │  [Transfer]                │
│     │     │     │                      │
│     └─────┘     ▼                      │
│              [Done]                    │
└────────────────────────────────────────┘

Example Code:
while (status_register & BUSY_BIT) {
    // Wait (busy loop)
}
// Device ready, transfer data
data_register = data;

Timing Diagram:
CPU:  [Poll][Poll][Poll][Poll][Read][Process]
           ↓    ↓    ↓    ↓
Device:   [Busy................][Ready][Transfer]
          └── Wasted CPU time ──┘

Advantages:
  + Simple hardware
  + Simple software
  + Fast for immediately-ready devices

Disadvantages:
  - Wastes CPU cycles
  - Poor for slow devices
  - Can't do other work while waiting
  - Inefficient

Example: Reading from disk (100,000 cycles)
  CPU @ 3 GHz = 3×10⁹ cycles/sec
  Disk latency = 10 ms = 0.01 sec
  Wasted cycles = 3×10⁹ × 0.01 = 30,000,000 cycles!
```

### 6.2.2 Interrupt-Driven I/O

Device interrupts CPU when ready.

```
┌─────────────────────────────────────────┐
│      Interrupt-Driven I/O               │
│                                         │
│  CPU ──► Initiate I/O ──► Continue work │
│            │                     │      │
│            │                     │      │
│         Device                   │      │
│         working                  │      │
│            │                     │      │
│            ▼                     │      │
│      [Ready] ─────Interrupt─────►│      │
│                                  │      │
│                        Save state│      │
│                        Run ISR   │      │
│                        Restore   │      │
│                        Resume    │      │
└─────────────────────────────────────────┘

Interrupt Service Routine (ISR):
┌──────────────────────────────────┐
│ 1. Save CPU state (registers)    │
│ 2. Identify interrupt source     │
│ 3. Service the device            │
│ 4. Clear interrupt               │
│ 5. Restore CPU state             │
│ 6. Return to interrupted program │
└──────────────────────────────────┘

Timing Diagram:
CPU:  [Initiate][Work][Work][Work][ISR][Work]
           │                       ↑
           ▼                       │
Device:   [Busy............][Ready]
                           Interrupt

Interrupt Flow:
┌────────────────────────────────────┐
│ Normal Execution                   │
│  PC: 1000  [INST]                  │
│  PC: 1004  [INST]                  │
│  PC: 1008  [INST] ← Interrupt!     │
│                                    │
│ ┌────────────────────────┐         │
│ │ 1. Save PC (1008)      │         │
│ │ 2. Save PSW            │         │
│ │ 3. Disable interrupts  │         │
│ │ 4. Load ISR address    │         │
│ └────────────────────────┘         │
│                                    │
│ ISR Execution:                     │
│  PC: 5000  [PUSH regs]             │
│  PC: 5004  [Service device]        │
│  PC: 5008  [Clear interrupt]       │
│  PC: 500C  [POP regs]              │
│  PC: 5010  [IRET]                  │
│                                    │
│ ┌────────────────────────┐         │
│ │ 1. Restore PSW         │         │
│ │ 2. Restore PC (1008)   │         │
│ │ 3. Enable interrupts   │         │
│ └────────────────────────┘         │
│                                    │
│ Resume Execution:                  │
│  PC: 1008  [INST] ← Continue here  │
└────────────────────────────────────┘

Interrupt Vector Table:
┌─────────┬──────────────────────────┐
│ Address │        Handler           │
├─────────┼──────────────────────────┤
│ 0x0000  │  Reset                   │
│ 0x0004  │  Illegal Instruction     │
│ 0x0008  │  Division by Zero        │
│ 0x000C  │  Timer Interrupt         │
│ 0x0010  │  Keyboard Interrupt      │
│ 0x0014  │  Disk Interrupt          │
│ 0x0018  │  Network Interrupt       │
│   ...   │  ...                     │
└─────────┴──────────────────────────┘

Interrupt Priority:
┌──────────────────┬──────────┐
│     Source       │ Priority │
├──────────────────┼──────────┤
│ Power Fail       │ Highest  │
│ Machine Check    │    ↑     │
│ Timer            │    │     │
│ Disk             │    │     │
│ Network          │    │     │
│ Keyboard         │    ↓     │
│ Software         │ Lowest   │
└──────────────────┴──────────┘

Nested Interrupts:
Time ──────────────────────────►
Main: ████████░░░░░░░░████████
           ↑
Low Int:   ░░████░░░░░░
              ↑
High Int:     ░░██░░

High-priority can interrupt low-priority

Advantages:
  + CPU can do other work
  + Efficient for slow devices
  + Good response time

Disadvantages:
  - Interrupt overhead (save/restore)
  - Complex software
  - Context switch cost
  
Overhead:
  Save/restore: 50-100 cycles
  ISR execution: 100-1000 cycles
  Total: ~200-1100 cycles
```

### 6.2.3 Direct Memory Access (DMA)

Device transfers data directly to/from memory without CPU.

```
┌───────────────────────────────────────────┐
│         DMA Architecture                  │
│                                           │
│  ┌──────┐      ┌────────┐                 │
│  │ CPU  │      │ Memory │                 │
│  └───┬──┘      └────┬───┘                 │
│      │              │                     │
│      │    System Bus                      │
│      ├──────────────┼──────────┐          │
│      │              │          │          │
│      ▼              ▼          ▼          │
│  ┌─────────┐   ┌────────┐ ┌────────┐      │
│  │   DMA   │   │  Disk  │ │Network │      │
│  │Controller   │ Ctrl   │ │  Ctrl  │      │
│  └─────┬───┘   └────┬───┘ └────┬───┘      │
│        │            │          │          │
│        └────────────┼──────────┘          │
│                     ▼                     │
│              ┌──────────┐                 │
│              │  Devices │                 │
│              └──────────┘                 │
└───────────────────────────────────────────┘

DMA Operation:

1. Setup Phase:
┌─────────────────────────────────┐
│ CPU configures DMA controller:  │
│  - Source address               │
│  - Destination address          │
│  - Transfer count               │
│  - Direction (read/write)       │
│  - Start transfer               │
└─────────────────────────────────┘

2. Transfer Phase:
┌─────────────────────────────────┐
│ DMA Controller:                 │
│  - Takes control of bus         │
│  - Transfers data               │
│  - Updates addresses/count      │
│  - CPU can continue (mostly)    │
└─────────────────────────────────┘

3. Completion:
┌─────────────────────────────────┐
│ DMA Controller:                 │
│  - Signals interrupt            │
│  - Releases bus                 │
│  - CPU processes completion     │
└─────────────────────────────────┘

DMA Transfer Modes:

1. Burst Mode (Block Transfer):
CPU:    [Setup]░░░░░░░░░░░[ISR][Work]
                ↑          ↑
DMA:            [████████] Done
                Transfer block
   
   - Transfers entire block
   - CPU blocked during transfer
   - Fast, but CPU must wait

2. Cycle Stealing:
CPU:    [Setup][██░██░██░██][Work]
                  ↑
DMA:              [Transfer one word at a time]
   
   - DMA steals bus cycles
   - CPU and DMA interleaved
   - Slower, but CPU can work

3. Transparent Mode:
   - Transfer only when bus idle
   - No CPU slowdown
   - Slowest transfer

DMA Controller Registers:
┌──────────────────┬──────────────────┐
│   Register       │    Purpose       │
├──────────────────┼──────────────────┤
│ Source Address   │ Where to read    │
│ Dest Address     │ Where to write   │
│ Count            │ Bytes to transfer│
│ Control          │ Mode, direction  │
│ Status           │ Done, error      │
└──────────────────┴──────────────────┘

Example: Disk Read (1 MB)

Without DMA:
  CPU reads each word: 1MB / 4 bytes = 256K reads
  Each read: ~100 cycles
  Total: 25.6M cycles wasted!
  
With DMA:
  CPU setup: ~100 cycles
  DMA transfer: CPU free!
  Interrupt: ~100 cycles
  Total CPU: ~200 cycles
  
Speedup: 25.6M / 200 = 128,000×!

Memory Bus Arbitration:
┌──────────────────────────────────┐
│     Who controls bus?            │
│                                  │
│  Priority (highest to lowest):   │
│  1. CPU (default)                │
│  2. DMA (when requested)         │
│  3. Other bus masters            │
│                                  │
│  Arbitration signals:            │
│  - Bus Request (BR)              │
│  - Bus Grant (BG)                │
│  - Bus Busy (BBSY)               │
└──────────────────────────────────┘

Advantages:
  + No CPU involvement in transfer
  + Very efficient for large transfers
  + CPU can do other work
  + Fast

Disadvantages:
  - Extra hardware (DMA controller)
  - Bus contention
  - Cache coherency issues
```

## 6.3 Bus Architecture

### 6.3.1 Bus Types

```
┌────────────────────────────────────────┐
│          Bus Hierarchy                 │
│                                        │
│         ┌──────┐                       │
│         │ CPU  │                       │
│         └───┬──┘                       │
│             │                          │
│         Processor Bus                  │
│        (Fastest, proprietary)          │
│             │                          │
│         ┌───┴──────┐                   │
│         │  Cache/  │                   │
│         │  Bridge  │                   │
│         └───┬──────┘                   │
│             │                          │
│         Memory Bus                     │
│        (High bandwidth)                │
│             │                          │
│         ┌───┴──────┐                   │
│         │ Memory   │                   │
│         │Controller│                   │
│         └───┬──────┘                   │
│             │                          │
│         I/O Bus (PCIe, etc.)           │
│        (Standard, expandable)          │
│             │                          │
│     ┌───────┼───────┬────────┐         │
│     ▼       ▼       ▼        ▼         │
│  [Video] [Disk] [Network] [USB]        │
│                                        │
│         Peripheral Bus                 │
│        (USB, SATA, etc.)               │
│             │                          │
│     ┌───────┼───────┐                  │
│     ▼       ▼       ▼                  │
│  [Mouse][Keyboard][Printer]            │
└────────────────────────────────────────┘

Bus Characteristics:
┌───────────┬──────────┬───────────┬─────────┐
│   Bus     │  Width   │   Speed   │Bandwidth│
├───────────┼──────────┼───────────┼─────────┤
│ Processor │ 64-256   │  GHz      │ 10+ GB/s│
│ Memory    │ 64-128   │  GHz      │  5-25   │
│ PCIe 4.0  │ 1-16     │  16 GT/s  │  2-32   │
│ USB 3.0   │   1      │  5 Gb/s   │  0.5    │
│ SATA III  │   1      │  6 Gb/s   │  0.6    │
└───────────┴──────────┴───────────┴─────────┘
```

### 6.3.2 Bus Signals

```
Bus Signal Types:

┌──────────────────────────────────────┐
│  Address Bus (Unidirectional)        │
│  CPU → Memory/Device                 │
│  ┌────────────────────────────┐      │
│  │ A31 A30 A29 ... A1 A0      │      │
│  └────────────────────────────┘      │
│  Width determines address space      │
│  32-bit → 4 GB addressable           │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│  Data Bus (Bidirectional)            │
│  CPU ↔ Memory/Device                 │
│  ┌────────────────────────────┐      │
│  │ D63 D62 D61 ... D1 D0      │      │
│  └────────────────────────────┘      │
│  Width determines transfer size      │
│  64-bit → 8 bytes per transfer       │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│  Control Bus                         │
│  ┌────────────────────────────┐      │
│  │ - R/W (Read/Write)         │      │
│  │ - MREQ (Memory Request)    │      │
│  │ - IORQ (I/O Request)       │      │
│  │ - IRQ (Interrupt Request)  │      │
│  │ - CLK (Clock)              │      │
│  │ - RESET                    │      │
│  │ - Bus Request/Grant        │      │
│  └────────────────────────────┘      │
└──────────────────────────────────────┘

Bus Transaction (Read):
         ___     ___     ___     ___
CLK   __|   |___|   |___|   |___|   |__
      ____________________________
ADDR  ────<  Valid Address      >─────
      ____________________________
R/W   ────<  HIGH (Read)         >────
                    ____________
MREQ  ─────────────<  ASSERTED  >────
                        __________
DATA  ─────────────────<Valid Data>───

Timing:
  T1: Address valid
  T2: Control signals asserted
  T3: Wait for device
  T4: Data ready
  
Total: 4 clock cycles per transfer
```

### 6.3.3 Bus Arbitration

Multiple devices competing for bus access.

```
Centralized Arbitration:

┌──────────────────────────────────────┐
│       Bus Arbiter (Central)          │
│                                      │
│  Device 1 ──BR1──►┌──────────┐       │
│  Device 2 ──BR2──►│  Arbiter │       │
│  Device 3 ──BR3──►│  (Logic) │       │
│  Device 4 ──BR4──►└─────┬────┘       │
│                          │           │
│  Device 1 ◄──BG1─────────┤           │
│  Device 2 ◄──BG2─────────┤           │
│  Device 3 ◄──BG3─────────┤           │
│  Device 4 ◄──BG4─────────┘           │
│                                      │
│  BR = Bus Request                    │
│  BG = Bus Grant                      │
└──────────────────────────────────────┘

Priority Schemes:

1. Fixed Priority:
   Device 1 > Device 2 > Device 3 > Device 4
   Simple but can starve low-priority devices

2. Round-Robin:
   1 → 2 → 3 → 4 → 1 → ...
   Fair, but doesn't consider urgency

3. Dynamic Priority:
   Priority based on waiting time or importance
   Complex but flexible

Distributed Arbitration (Daisy Chain):

BG ──►[Dev1]──►[Dev2]──►[Dev3]──►[Dev4]
       │        │        │        │
       └────────┴────────┴────────┘
                 │
                BR

Process:
  1. Device asserts BR (Bus Request)
  2. Arbiter asserts BG (Bus Grant)
  3. BG propagates through chain
  4. First device requesting gets grant
  5. Stops propagation
  
Advantage: Simple hardware
Disadvantage: Priority by position

Self-Selection:

Each device has unique ID
Arbitration by comparing IDs on bus

┌──────────────────────────────────┐
│  Device 1 puts ID on bus: 0001   │
│  Device 2 puts ID on bus: 0010   │
│  Device 3 puts ID on bus: 0100   │
│                                  │
│  Highest ID wins                 │
│  (Wired-OR logic)                │
│                                  │
│  Result: 0111                    │
│  Winner: Device with bit 2 set   │
└──────────────────────────────────┘
```

## 6.4 I/O Addressing

### 6.4.1 Memory-Mapped I/O

```
I/O devices mapped into memory address space

┌────────────────────────────────────┐
│    Memory Address Space (4 GB)    │
│                                    │
│  0xFFFFFFFF ┌────────────────┐    │
│             │                │    │
│             │   I/O Devices  │    │
│  0xFF000000 ├────────────────┤    │
│             │                │    │
│             │      ROM       │    │
│  0xF0000000 ├────────────────┤    │
│             │                │    │
│             │                │    │
│             │   Main RAM     │    │
│             │                │    │
│             │                │    │
│  0x00000000 └────────────────┘    │
└────────────────────────────────────┘

Example: Video memory at 0xA0000000
  MOV R1, #0xA0000000   ; Video address
  MOV R2, #0xFF         ; Pixel value
  STR R2, [R1]          ; Write to video

Advantages:
  + Same instructions for I/O and memory
  + No special I/O instructions
  + Easy to map devices
  + Can use all addressing modes

Disadvantages:
  - Uses memory address space
  - Cache coherency issues
  - Harder to distinguish I/O from memory
```

### 6.4.2 Isolated I/O (Port-Mapped)

```
Separate address space for I/O devices

CPU has separate I/O instructions

x86 Example:
  IN  AL, PORT     ; Read from I/O port
  OUT PORT, AL     ; Write to I/O port
  
  IN  AL, 0x60     ; Read keyboard
  OUT 0x20, AL     ; Write to interrupt controller

┌───────────────────┐  ┌──────────────────┐
│ Memory Space      │  │  I/O Space       │
│  4 GB             │  │  64 KB           │
│                   │  │                  │
│ 0xFFFFFFFF        │  │ 0xFFFF           │
│     ┌──────┐      │  │  ┌──────┐        │
│     │      │      │  │  │ Port │        │
│     │ RAM  │      │  │  │ 80   │        │
│     │      │      │  │  ├──────┤        │
│     │      │      │  │  │ Port │        │
│ 0x00000000        │  │  │ 60   │        │
│     └──────┘      │  │  └──────┘        │
│                   │  │ 0x0000           │
└───────────────────┘  └──────────────────┘

Address Decoding:
┌─────────────────────────────────────┐
│  CPU asserts:                       │
│  - Address on address bus           │
│  - IORQ (I/O Request) signal        │
│                                     │
│  Device decodes address:            │
│  IF (IORQ == 1 AND Address == Mine) │
│     Respond to request              │
│  END IF                             │
└─────────────────────────────────────┘

Advantages:
  + Separate address spaces
  + No memory address waste
  + Clear I/O distinction
  + No cache coherency issues

Disadvantages:
  - Need special instructions
  - Limited I/O address space
  - More complex CPU
```

## 6.5 Storage Systems

### 6.5.1 Hard Disk Drive (HDD)

```
Physical Structure:

         Spindle
           │
    ┌──────┼──────┐
    │      │      │  ← Platter
    ├──────┼──────┤
    │      │      │
    ├──────┼──────┤
    │      │      │
    └──────┴──────┘
        ↑
      Read/Write Head
        (on actuator arm)

Top View of Platter:
    ┌────────────────────┐
    │    ╱───────╲       │
    │   ╱         ╲      │
    │  │  ┌─────┐  │     │ ← Track
    │  │  │     │  │     │
    │  │  └─────┘  │     │
    │   ╲         ╱      │
    │    ╲───────╱       │
    └────────────────────┘
         ↑         ↑
       Sector   Cylinder

Characteristics:
  Capacity: 500 GB - 20 TB
  RPM: 5400, 7200, 10000, 15000
  Latency: 5-10 ms
  Transfer: 100-200 MB/s

Access Time Components:

1. Seek Time (move head to track):
   ┌───────────────────────┐
   │ Average: 4-8 ms       │
   │ Max: 10-15 ms         │
   └───────────────────────┘

2. Rotational Latency (wait for sector):
   ┌───────────────────────┐
   │ Average: 1/2 rotation │
   │ 7200 RPM:             │
   │   60s/7200 ÷ 2 = 4ms │
   └───────────────────────┘

3. Transfer Time:
   ┌───────────────────────┐
   │ Bytes / Transfer_Rate │
   │ 4KB / 100MB/s = 40μs  │
   └───────────────────────┘

Total: Seek + Rotational + Transfer
     = 8ms + 4ms + 0.04ms
     ≈ 12 ms per 4KB read

Disk Addressing:

CHS (Cylinder-Head-Sector):
  ┌──────────┬──────┬────────┐
  │ Cylinder │ Head │ Sector │
  └──────────┴──────┴────────┘
  Old, physical addressing

LBA (Logical Block Addressing):
  ┌────────────────────────┐
  │   Block Number (0-N)   │
  └────────────────────────┘
  Modern, linear addressing
  
  LBA 0 = First sector
  LBA 1 = Second sector
  ...

Disk Scheduling Algorithms:

FCFS (First Come First Serve):
  Queue: 98, 183, 37, 122, 14, 124, 65, 67
  Head at: 53
  Path: 53→98→183→37→122→14→124→65→67
  Total: 640 cylinders

SSTF (Shortest Seek Time First):
  Pick closest request
  Path: 53→65→67→37→14→98→122→124→183
  Total: 236 cylinders
  (Better, but can starve)

SCAN (Elevator):
  Go one direction, then reverse
  Path: 53→37→14→0→65→67→98→122→124→183
  Total: 208 cylinders
  (Fair, no starvation)

C-SCAN (Circular SCAN):
  Go one direction, jump back, repeat
  Path: 53→65→67→98→122→124→183→199→0→14→37
  More uniform wait time
```

### 6.5.2 Solid State Drive (SSD)

```
Architecture:

┌──────────────────────────────────────┐
│          SSD Controller              │
│  ┌────────────────────────────────┐  │
│  │   Flash Translation Layer      │  │
│  │   (Wear Leveling, GC, etc.)    │  │
│  └────────────────────────────────┘  │
├──────────────┬───────────────────────┤
│              │                       │
│  ┌───────┬───┴───┬───────┬───────┐   │
│  │Flash  │Flash  │Flash  │Flash  │   │
│  │Chip 0 │Chip 1 │Chip 2 │Chip 3 │   │
│  └───────┴───────┴───────┴───────┘   │
│  ┌───────┬───────┬───────┬───────┐   │
│  │Flash  │Flash  │Flash  │Flash  │   │
│  │Chip 4 │Chip 5 │Chip 6 │Chip 7 │   │
│  └───────┴───────┴───────┴───────┘   │
└──────────────────────────────────────┘

NAND Flash Memory:

Block Structure:
┌─────────────────────┐
│  Block (128-256 KB) │
│  ┌───────────────┐  │
│  │ Page (4-16 KB)│  │ ← Read/Write unit
│  ├───────────────┤  │
│  │ Page          │  │
│  ├───────────────┤  │
│  │    ...        │  │
│  └───────────────┘  │
└─────────────────────┘ ← Erase unit

Operations:
  - Read:  Page level (fast: ~25μs)
  - Write: Page level (slow: ~200μs)
  - Erase: Block level (very slow: ~1.5ms)

Constraints:
  - Must erase before write
  - Limited write cycles (P/E cycles)
    SLC: ~100,000
    MLC: ~10,000
    TLC: ~3,000
    QLC: ~1,000

Characteristics:
  Capacity: 128 GB - 8 TB
  Latency: 0.05-0.15 ms
  Read: 500-3500 MB/s
  Write: 300-3000 MB/s
  Random IOPS: 10K-500K

Advantages over HDD:
  + Much faster access (~100×)
  + No moving parts
  + Shock resistant
  + Lower power
  + Silent

Disadvantages:
  - More expensive per GB
  - Limited write endurance
  - Write amplification
  - Data retention issues (unpowered)

Wear Leveling:
  Distribute writes evenly across blocks
  
  ┌─────────────────────────────┐
  │  Logical    Physical        │
  │  Block  →   Block           │
  │                             │
  │    0    →    100            │
  │    1    →    200            │
  │    2    →     50            │
  │   ...   →    ...            │
  │                             │
  │  Mapping changes to balance │
  │  wear across all blocks     │
  └─────────────────────────────┘

TRIM Command:
  OS tells SSD which blocks are free
  Allows better garbage collection
  Improves performance
```

### 6.5.3 RAID (Redundant Array of Independent Disks)

```
Combine multiple disks for performance/reliability

RAID 0 (Striping):
┌─────┬─────┐
│Disk0│Disk1│
├─────┼─────┤
│ A0  │ A1  │ ← Block A split
│ B0  │ B1  │ ← Block B split
│ C0  │ C1  │
└─────┴─────┘

Capacity: N × disk_size
Performance: N × single_disk
Reliability: Worse (any disk fails = data loss)
Use: Performance, no redundancy

RAID 1 (Mirroring):
┌─────┬─────┐
│Disk0│Disk1│
├─────┼─────┤
│  A  │  A  │ ← Duplicated
│  B  │  B  │
│  C  │  C  │
└─────┴─────┘

Capacity: 1 × disk_size
Performance: 2× read, 1× write
Reliability: Can lose 1 disk
Use: Reliability, simple

RAID 5 (Parity):
┌─────┬─────┬─────┐
│Disk0│Disk1│Disk2│
├─────┼─────┼─────┤
│ A0  │ A1  │ Ap  │ ← A0⊕A1=Ap
│ B0  │ Bp  │ B1  │ ← Parity rotates
│ Cp  │ C0  │ C1  │
└─────┴─────┴─────┘

Capacity: (N-1) × disk_size
Performance: Good read, slower write
Reliability: Can lose 1 disk
Rebuild: XOR to recover

Example Recovery:
  If Disk1 fails:
    A0 ⊕ Ap = A1
    C0 ⊕ C1 = Cp → Derive missing

RAID 6 (Double Parity):
┌─────┬─────┬─────┬─────┐
│Disk0│Disk1│Disk2│Disk3│
├─────┼─────┼─────┼─────┤
│ A0  │ A1  │ Ap  │ Aq  │ ← 2 parity
│ B0  │ Bp  │ Bq  │ B1  │
│ Cp  │ Cq  │ C0  │ C1  │
└─────┴─────┴─────┴─────┘

Capacity: (N-2) × disk_size
Reliability: Can lose 2 disks
Use: High reliability

RAID 10 (1+0):
Mirror first, then stripe

┌───────────┬───────────┐
│  Disk0-1  │  Disk2-3  │
│  (Mirror) │  (Mirror) │
├─────┬─────┼─────┬─────┤
│ A   │ A   │ B   │ B   │
│ C   │ C   │ D   │ D   │
└─────┴─────┴─────┴─────┘
    └─── Striped ───┘

Capacity: N/2 × disk_size
Performance: Best
Reliability: Can lose 1 per mirror pair

Comparison:
┌──────┬────────┬──────┬──────────┬──────┐
│RAID  │Capacity│ Perf │Reliabilty│Cost  │
├──────┼────────┼──────┼──────────┼──────┤
│  0   │  100%  │ Best │  Worst   │ Low  │
│  1   │   50%  │ Good │  Good    │ High │
│  5   │(N-1)/N │ Good │  Good    │ Med  │
│  6   │(N-2)/N │ Fair │  Better  │ Med  │
│ 10   │   50%  │ Best │  Best    │ High │
└──────┴────────┴──────┴──────────┴──────┘
```

---

**Key Takeaways:**
1. I/O systems bridge CPU and external devices
2. Three communication methods: Polling, Interrupts, DMA
3. Interrupts provide efficient asynchronous I/O
4. DMA offloads bulk transfers from CPU
5. Bus architecture hierarchical for performance
6. Storage: HDD (capacity), SSD (speed), RAID (reliability)
7. I/O addressing: Memory-mapped vs. Port-mapped

**Next:** [Advanced Topics](./07-advanced-topics.md) | [Heterogeneous Computing](./08-heterogeneous-computing.md)


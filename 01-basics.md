# Chapter 1: Basics and Fundamentals

## 1.1 Number Systems

### 1.1.1 Decimal System (Base-10)
The decimal system uses 10 digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Position values: ..., 10³, 10², 10¹, 10⁰
- Example: 247₁₀ = (2 × 10²) + (4 × 10¹) + (7 × 10⁰)

### 1.1.2 Binary System (Base-2)
The binary system uses 2 digits: 0, 1
- Position values: ..., 2³, 2², 2¹, 2⁰
- Example: 1011₂ = (1 × 2³) + (0 × 2²) + (1 × 2¹) + (1 × 2⁰) = 11₁₀

```
Binary to Decimal Conversion:
   Position:  7   6   5   4   3   2   1   0
   Power:     2⁷  2⁶  2⁵  2⁴  2³  2²  2¹  2⁰
   Value:    128  64  32  16   8   4   2   1
   
   Example: 10110101₂
            1×128 + 0×64 + 1×32 + 1×16 + 0×8 + 1×4 + 0×2 + 1×1
            = 128 + 32 + 16 + 4 + 1 = 181₁₀
```

### 1.1.3 Hexadecimal System (Base-16)
Hexadecimal uses 16 digits: 0-9, A-F (A=10, B=11, C=12, D=13, E=14, F=15)
- Position values: ..., 16³, 16², 16¹, 16⁰
- Example: 2F₁₆ = (2 × 16¹) + (15 × 16⁰) = 47₁₀

```
Hex-Binary-Decimal Conversion Table:
┌─────┬────────┬─────────┐
│ Hex │ Binary │ Decimal │
├─────┼────────┼─────────┤
│  0  │  0000  │    0    │
│  1  │  0001  │    1    │
│  2  │  0010  │    2    │
│  3  │  0011  │    3    │
│  4  │  0100  │    4    │
│  5  │  0101  │    5    │
│  6  │  0110  │    6    │
│  7  │  0111  │    7    │
│  8  │  1000  │    8    │
│  9  │  1001  │    9    │
│  A  │  1010  │   10    │
│  B  │  1011  │   11    │
│  C  │  1100  │   12    │
│  D  │  1101  │   13    │
│  E  │  1110  │   14    │
│  F  │  1111  │   15    │
└─────┴────────┴─────────┘
```

### 1.1.4 Octal System (Base-8)
Octal uses 8 digits: 0-7
- Position values: ..., 8³, 8², 8¹, 8⁰
- Example: 157₈ = (1 × 8²) + (5 × 8¹) + (7 × 8⁰) = 111₁₀

```
Binary to Octal Conversion (Group by 3 bits):
   Binary: 101 110 111
   Octal:   5   6   7  = 567₈
```

## 1.2 Data Representation

### 1.2.1 Unsigned Integers
- Uses all bits for magnitude
- Range for n bits: 0 to (2ⁿ - 1)
- Example (8-bit): 00000000₂ to 11111111₂ → 0 to 255₁₀

### 1.2.2 Signed Integers

#### Sign-Magnitude Representation
```
┌───┬───────────────────────┐
│ S │    Magnitude (n-1)    │
└───┴───────────────────────┘
MSB = Sign bit (0=positive, 1=negative)

Problems:
- Two representations of zero (+0 and -0)
- Addition/subtraction complex
```

#### One's Complement
```
Positive: Same as unsigned
Negative: Invert all bits (flip 0→1, 1→0)

Example (4-bit):
 +5: 0101
 -5: 1010 (flip all bits)

Problems:
- Two zeros: 0000 (+0) and 1111 (-0)
- Addition needs end-around carry
```

#### Two's Complement (Most Common)
```
Positive: Same as unsigned
Negative: Invert all bits and add 1

Example (8-bit):
 +5:  00000101
 -5:  11111010 + 1 = 11111011

Range for n bits: -2^(n-1) to 2^(n-1) - 1
8-bit: -128 to +127

Advantages:
- Single representation of zero
- Simple addition/subtraction circuits
- MSB indicates sign (0=+, 1=-)

Verification: +5 + (-5) = 0
  00000101
+ 11111011
-----------
 100000000  (carry discarded) = 00000000 ✓
```

### 1.2.3 Floating-Point Representation

#### IEEE 754 Standard

**Single Precision (32-bit):**
```
┌─┬──────────┬───────────────────────┐
│S│ Exponent │      Mantissa         │
│ │  (8)     │        (23)           │
└─┴──────────┴───────────────────────┘
 1    8 bits      23 bits

Value = (-1)^S × 1.M × 2^(E-127)
```

**Double Precision (64-bit):**
```
┌─┬──────────┬─────────────────────────────────────┐
│S│ Exponent │           Mantissa                  │
│ │  (11)    │            (52)                     │
└─┴──────────┴─────────────────────────────────────┘
 1   11 bits         52 bits

Value = (-1)^S × 1.M × 2^(E-1023)
```

**Example: Representing 12.625₁₀**
```
Step 1: Convert to binary
  12.625 = 12 + 0.625
  12 = 1100₂
  0.625 × 2 = 1.25  → 1
  0.25  × 2 = 0.5   → 0
  0.5   × 2 = 1.0   → 1
  = 1100.101₂

Step 2: Normalize (scientific notation)
  1100.101 = 1.100101 × 2³

Step 3: Extract components
  Sign: 0 (positive)
  Exponent: 3 + 127 = 130 = 10000010₂
  Mantissa: 100101 (23 bits: 10010100000000000000000)

Result: 0 10000010 10010100000000000000000
```

**Special Values:**
```
┌──────────────────┬──────────┬──────────┐
│      Value       │ Exponent │ Mantissa │
├──────────────────┼──────────┼──────────┤
│       Zero       │    0     │    0     │
│     Infinity     │   All 1s │    0     │
│       NaN        │   All 1s │  Non-0   │
│ Denormalized     │    0     │  Non-0   │
└──────────────────┴──────────┴──────────┘
```

### 1.2.4 Character Representation

#### ASCII (7-bit, 128 characters)
```
Range: 0-127
Examples:
  'A' = 65  = 0x41 = 01000001₂
  'a' = 97  = 0x61 = 01100001₂
  '0' = 48  = 0x30 = 00110000₂
  ' ' = 32  = 0x20 = 00100000₂
  '\n'= 10  = 0x0A = 00001010₂

┌─────────┬────────────────────────┐
│  Range  │      Characters        │
├─────────┼────────────────────────┤
│  0-31   │  Control characters    │
│ 32-47   │  Special symbols       │
│ 48-57   │  Digits (0-9)          │
│ 65-90   │  Uppercase (A-Z)       │
│ 97-122  │  Lowercase (a-z)       │
└─────────┴────────────────────────┘
```

#### Unicode (UTF-8, UTF-16, UTF-32)
- UTF-8: Variable length (1-4 bytes)
- UTF-16: 2 or 4 bytes
- UTF-32: Fixed 4 bytes
- Covers all world languages and symbols

## 1.3 Boolean Algebra

### 1.3.1 Basic Operations

**AND (·):**
```
  A · B = C
  
  A  B │ A·B
  ─────┼────
  0  0 │  0
  0  1 │  0
  1  0 │  0
  1  1 │  1
```

**OR (+):**
```
  A + B = C
  
  A  B │ A+B
  ─────┼────
  0  0 │  0
  0  1 │  1
  1  0 │  1
  1  1 │  1
```

**NOT (¯ or '):**
```
  Ā = C
  
  A │ Ā
  ──┼──
  0 │ 1
  1 │ 0
```

**NAND:**
```
  A  B │ NAND
  ─────┼─────
  0  0 │  1
  0  1 │  1
  1  0 │  1
  1  1 │  0
```

**NOR:**
```
  A  B │ NOR
  ─────┼────
  0  0 │  1
  0  1 │  0
  1  0 │  0
  1  1 │  0
```

**XOR (⊕):**
```
  A ⊕ B = C
  
  A  B │ A⊕B
  ─────┼────
  0  0 │  0
  0  1 │  1
  1  0 │  1
  1  1 │  0
```

**XNOR:**
```
  A  B │ XNOR
  ─────┼─────
  0  0 │  1
  0  1 │  0
  1  0 │  0
  1  1 │  1
```

### 1.3.2 Boolean Laws

```
Identity Laws:
  A + 0 = A
  A · 1 = A

Null Laws:
  A + 1 = 1
  A · 0 = 0

Idempotent Laws:
  A + A = A
  A · A = A

Inverse Laws:
  A + Ā = 1
  A · Ā = 0

Commutative Laws:
  A + B = B + A
  A · B = B · A

Associative Laws:
  (A + B) + C = A + (B + C)
  (A · B) · C = A · (B · C)

Distributive Laws:
  A · (B + C) = (A · B) + (A · C)
  A + (B · C) = (A + B) · (A + C)

De Morgan's Laws:
  (A + B)' = A' · B'
  (A · B)' = A' + B'

Absorption Laws:
  A + (A · B) = A
  A · (A + B) = A
```

### 1.3.3 Logic Gates (ASCII Diagrams)

```
AND Gate:
    A ────┐
          │>───── A·B
    B ────┘

OR Gate:
    A ────┐
          │≥1──── A+B
    B ────┘

NOT Gate (Inverter):
    A ────>○──── Ā

NAND Gate:
    A ────┐
          │>○─── (A·B)'
    B ────┘

NOR Gate:
    A ────┐
          │≥1○── (A+B)'
    B ────┘

XOR Gate:
    A ────┐
          │=1──── A⊕B
    B ────┘

XNOR Gate:
    A ────┐
          │=1○─── (A⊕B)'
    B ────┘
```

## 1.4 Combinational vs Sequential Logic

### 1.4.1 Combinational Logic
```
Inputs ──→ [Logic Gates] ──→ Outputs
           (No memory)

Characteristics:
- Output depends only on current inputs
- No feedback loops
- No memory elements
- Examples: Adders, Multiplexers, Decoders
```

### 1.4.2 Sequential Logic
```
         ┌─────────────────┐
Inputs ─→│  Logic Gates    │──→ Outputs
         │  + Memory       │
         └────────┬────────┘
                  │
              [Feedback]
                  │
                  ↑

Characteristics:
- Output depends on current inputs AND previous state
- Contains memory elements (flip-flops)
- Can store information
- Examples: Counters, Registers, State Machines
```

## 1.5 Basic Computer Organization

### 1.5.1 Von Neumann Architecture
```
┌────────────────────────────────────────────────────┐
│                    System Bus                      │
│              (Address, Data, Control)              │
└──┬─────────┬──────────────┬──────────────┬────────┘
   │         │              │              │
   ▼         ▼              ▼              ▼
┌─────┐  ┌──────┐      ┌────────┐    ┌─────────┐
│ CPU │  │Memory│      │ Input  │    │ Output  │
│     │  │      │      │Devices │    │ Devices │
├─────┤  └──────┘      └────────┘    └─────────┘
│ ALU │      ↑
│  +  │      │
│  CU │      │
└─────┘──────┘

Key Features:
1. Single memory for instructions and data
2. Sequential instruction execution
3. Stored program concept
4. Bottleneck: CPU-Memory bandwidth (Von Neumann bottleneck)
```

### 1.5.2 Harvard Architecture
```
┌──────────┐              ┌──────────┐
│Instruction              │   Data   │
│  Memory  │              │  Memory  │
└────┬─────┘              └────┬─────┘
     │                         │
     │ Instruction Bus         │ Data Bus
     │                         │
     └───────────┬─────────────┘
                 ▼
             ┌──────┐
             │ CPU  │
             │      │
             │ ALU  │
             │  +   │
             │  CU  │
             └──────┘

Key Features:
1. Separate memory for instructions and data
2. Parallel access to instructions and data
3. No Von Neumann bottleneck
4. Used in embedded systems, DSPs
```

## 1.6 Binary Arithmetic

### 1.6.1 Binary Addition
```
Rules:
  0 + 0 = 0
  0 + 1 = 1
  1 + 0 = 1
  1 + 1 = 10 (0 with carry 1)
  1 + 1 + 1 = 11 (1 with carry 1)

Example:
    Carry:  1 1 1 1
             1 0 1 1  (11)
           + 0 1 1 1  ( 7)
           ─────────
           1 0 0 1 0  (18)
```

### 1.6.2 Binary Subtraction
```
Rules:
  0 - 0 = 0
  1 - 0 = 1
  1 - 1 = 0
  0 - 1 = 1 (with borrow 1)

Example:
    Borrow:  1
             1 0 1 1  (11)
           - 0 1 0 1  ( 5)
           ─────────
             0 1 1 0  ( 6)

Alternative: Use 2's complement addition
  A - B = A + (-B)
```

### 1.6.3 Binary Multiplication
```
Rules:
  0 × 0 = 0
  0 × 1 = 0
  1 × 0 = 0
  1 × 1 = 1

Example:
           1 0 1 1  (11)
         ×   1 0 1  ( 5)
         ─────────
           1 0 1 1  (11 × 1)
         0 0 0 0    (11 × 0, shifted)
       1 0 1 1      (11 × 1, shifted)
       ─────────
       1 1 0 1 1 1  (55)
```

### 1.6.4 Binary Division
```
Similar to decimal long division

Example: 1101 ÷ 11 (13 ÷ 3)
           1 0 0  (Quotient = 4)
        ┌────────
    11 │ 1 1 0 1
        -1 1
         ───
           0 0
           0 0
           ───
             0 1
           - 0 0
           ─────
               1  (Remainder = 1)

Result: 13 ÷ 3 = 4 remainder 1
```

## 1.7 Data Size Units

```
┌──────────────┬────────────┬─────────────┐
│     Unit     │   Symbol   │    Bytes    │
├──────────────┼────────────┼─────────────┤
│     Bit      │     b      │    1/8      │
│    Nibble    │     -      │    1/2      │
│     Byte     │     B      │      1      │
│     Word     │     -      │  2 or 4     │
│   Kilobyte   │     KB     │    1,024    │
│   Megabyte   │     MB     │  1,024²     │
│   Gigabyte   │     GB     │  1,024³     │
│   Terabyte   │     TB     │  1,024⁴     │
│   Petabyte   │     PB     │  1,024⁵     │
└──────────────┴────────────┴─────────────┘

Note: In some contexts (storage), decimal (1000-based) 
units are used: 1 KB = 1000 B
```

## 1.8 Bit Manipulation Operations

### 1.8.1 Setting a Bit
```
To set bit n: number |= (1 << n)

Example: Set bit 2 in 10010₂
  10010  (18)
| 00100  (1 << 2)
-------
  10110  (22)
```

### 1.8.2 Clearing a Bit
```
To clear bit n: number &= ~(1 << n)

Example: Clear bit 3 in 11010₂
  11010  (26)
& 10111  ~(1 << 3)
-------
  10010  (18)
```

### 1.8.3 Toggling a Bit
```
To toggle bit n: number ^= (1 << n)

Example: Toggle bit 1 in 10010₂
  10010  (18)
^ 00010  (1 << 1)
-------
  10000  (16)
```

### 1.8.4 Checking a Bit
```
To check bit n: (number & (1 << n)) != 0

Example: Check bit 4 in 10110₂
  10110  (22)
& 10000  (1 << 4)
-------
  10000  (non-zero, so bit 4 is set)
```

---

**Key Takeaways:**
1. Computer systems work with binary (base-2) representation
2. Different number systems (binary, octal, hex) are used for convenience
3. Data representation includes integers, floating-point, and characters
4. Boolean algebra forms the foundation of digital logic
5. Understanding these basics is essential for computer architecture

**Next:** [CPU Architecture and Organization](./02-cpu-architecture.md)


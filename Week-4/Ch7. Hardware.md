The primary hardware bottlenecks executing DL workloads are:
- main memory bandwidth
- local (SRAM) memory
- power (primarily from data movement).

Computational capacity and memory bandwidth are two distinct concepts in the field of computer architecture and performance:

1. **Computational Capacity**:
    - Refers to the ability of a computer or processor to perform operations or calculations.
    - Measured typically in terms of the number of operations it can perform per second, often quantified in FLOPS (Floating Point Operations Per Second) for floating point calculations or MIPS (Million Instructions Per Second) for general instructions.
    - Depends on factors like the number of cores, clock speed, and the efficiency of the processor architecture.
    - Higher computational capacity means the processor can handle more complex tasks or perform tasks faster.

2. **Memory Bandwidth**:
    - Refers to the rate at which data can be read from or written to the memory by the processor.
    - Measured in units of data per second, such as gigabytes per second (GB/s).
    - Determines how quickly the processor can access data stored in memory, which is crucial for performance, especially in data-intensive tasks.
    - Limited memory bandwidth can create a bottleneck, slowing down the processor even if it has high computational capacity, because the processor needs to wait for data to be transferred to or from the memory.
    - Depends on factors like the type of memory (e.g., DDR4, DDR5), the memory interface width, and the memory clock speed.

In summary, while computational capacity is about how fast a processor can compute or process instructions, memory bandwidth is about how fast the processor can access the data it needs to perform those computations. Both are critical for overall system performance, but they address different aspects of it.
#### Memory wall or Bandwidth wall:
The slow growth in memory bandwidth relative to compute power. It results in compute units being idle.

**Some of the existing techniques to increase performance are :**
- using a memory hierarchy to facilitate data-reuse
- increasing the memory bandwidth
- placing the memory close to the compute units to reduce access time and energy
- applying a single instruction to multiple data
- reducing the numerical representation and compressing the data
- using specialized logic or dedicated accelerators.

**Dennard's scaling**:
Is a scaling law which states roughly that, as transistors get smaller, their power density stays constant, so that the power use stays in proportion with area; both voltage and current scale with length.

Training requires storing and retrieving the activations across all the layers, which typically involves reading and writing several GB of data (the activations) from and to DRAM.

In training CNNs, the size of the activations typically has a more significant impact on the total memory requirements than the size of the model.

Given the amount of data transfer, using a high bandwidth DRAM, such as HBM2E, for training tasks is beneficial.

**Metrics that Hardware designers should consider**:
1. performance per power and per cost
2. minimal transition work from an existing to a new hardware/software system
3. programmability and ease-of-use
4. high utilization (the device does not sit idle frequently).

Recent algorithmic advancements include depthwise separable convolution, dilated convolutions, residual connections, ReLU variants, and GNNs. New models have irregular memory access patterns, more complex control-flow, and dynamic computation graphs that vary with different input data and **cannot be optimized at compile time**. These models can benefit from higher general-purpose compute. Models with predictable access patterns and primarily dense linear algebra benefit from **dedicated matrix multipliers**.

This text outlines the hardware requirements for two different phases of machine learning (ML) operations: training and serving. Let's break down each line for a better understanding.

### Training Hardware Characteristics:

1. **Masses of $fp16 \rightarrow fp32$, $bf16 \rightarrow fp32$, and sufficient $fp32 \rightarrow fp32$ MACs**:
   - This refers to the need for a large number of Multiply-Accumulate (MAC) units capable of handling different data types. 
   - $fp16 \rightarrow fp32$ and $bf16 \rightarrow fp32$ indicate conversions from 16-bit floating point and bfloat16 formats to 32-bit floating point during computations, which is common in deep learning for better precision.
   - Sufficient $fp32 \rightarrow fp32$ MACs suggest the need for direct 32-bit floating point computations.

2. **High in-die interconnect (IDI) bandwidth for multicore GEMMs and broadcast/reduce collectives**:
   - "In-die interconnect" refers to the data pathways within a single chip (die).
   - High bandwidth is necessary for efficient communication among multiple cores, particularly for General Matrix Multiply (GEMM) operations and collective operations like broadcasting and reducing data across cores.

3. **Sufficiently large SRAM for the weights and some activations**:
   - SRAM (Static Random-Access Memory) is needed to store weights (parameters of the neural network) and some activation values (outputs of neurons) for quick access during training.

4. **High DRAM bandwidth to store and read the activations, or alternatively, a much larger SRAM capacity**:
   - DRAM (Dynamic Random-Access Memory) with high bandwidth is required to store and quickly access larger sets of activations.
   - Alternatively, having a larger SRAM could compensate for lower DRAM bandwidth.

5. **High intra-server inter-node bandwidth for:**
   - **Multinode GEMM**: Ensures efficient matrix multiplication across different nodes (computational units) within a server.
   - **Broadcast/reduce collectives in large embedding layers**: Facilitates efficient data sharing and aggregation in operations involving large embedding layers.
   - **Distributed training across nodes**: Important for scenarios where training is distributed over multiple nodes, which could be either different sockets or groups of cores in servers without specialized accelerators.

6. **High inter-server bandwidth for distributed training**:
   - This is crucial for fast data transfer between different servers involved in distributed training, ensuring efficient parallel processing and synchronization across servers.

### Serving Hardware Characteristics:

1. **Plenty of $fp16 \rightarrow fp32$, $bf16 \rightarrow fp32$, int8→int32, $fp8 \rightarrow fp32$ and some $fp32 \rightarrow fp32$ MACs**:
   - Similar to training, a variety of MAC units are needed for different data type conversions and computations, but with an additional focus on lower-precision formats like int8 and fp8, which are commonly used in inference for efficiency.

2. **High IDI bandwidth for multicore communication for GEMMs & broadcast/reduce collectives**:
   - As in training, high bandwidth within the chip is necessary for efficient multicore operations during serving, especially for GEMM and collective operations.

3. **Sufficiently large SRAM for the weights and some activations**:
   - Similar to the training phase, a large SRAM is needed to store weights and some activation values for quick access during inference.

4. **Video decoder for media analytic workloads (inference on video frames)**:
   - A dedicated video decoder is important for efficiently processing video frames, particularly in applications involving media analytics and video-based inference.

Both the training and serving phases require powerful computational resources, but they differ slightly in their specific requirements, reflecting the distinct computational challenges of each phase. Training often involves more intensive calculations and larger data, necessitating greater memory and bandwidth, while serving focuses on efficiency and speed, often with lower-precision computations.

## Moore, Dennard, and Amdahl
### **Gordon Moore**
The number of transistors that fit in the same chip area would double every two years

### **Robert Dennard**
As transistors shrink, their power density stays approximately constant. Dennard's scaling broke down in the mid-2000s due to current leaking. 

The total consumed power (in Watts or Joules per second) is the sum of the dynamic (or switching) power and the static (or leakage) power. Dennard scaling only accounts for the dynamic power, which is defined as follows:

$P_{D} = Q\cdot E\cdot f = \frac{1}{2}\cdot Q\cdot C\cdot V^2\cdot f$

E is the energy required to open or close a gate
Q is the number of active transistors
C is the capacitance(Capability of a material object or device to store electric charge)
V is the voltage
f is the frequency

Today, reducing the voltage in smaller transistors increases current leakage and increases the power density. Instead of having more clock cycles per second (higher frequency), the focus today is on increasing the instructions per cycle (IPC) or operations per cycle; that is, doing more work per cycle.

There is an ideal voltage that minimizes the sum of the static and dynamic power.
![](https://deeplearningsystems.ai/figures/ch07-02.png)

The propagation time $T_{\mathit{prop}}$ of the current through all the logic gates in its path needs to be less than $1$ clock cycle. Increasing the frequency past f-min linearly increases the required voltage, and (not shown) cubically increases the dynamic power.

Power generates heat, and too much heat can damage the circuits. There is a maximum power that a system can operate without damaging the circuitry, and this limits the maximum frequency. Most operate in the 2-3 GHz range.

The primary contributors to the **increased dark silicon** are the exponential growth in transistors per area, current leakage, and power constraints. **Multicore processors** and **specialized computing** are two methods to mitigate dark silicon. These methods have enable the continued growth in computational capacity at the expense of two new challenges: **Amdahl's law and the memory wall.**

### Gene Amdahl:
formalized the speedup when only a fraction of a program is improved, used to determine the limitations of parallel computing. Using N>1 cores for a particular workload results in a maximum speed up of
$\frac{1}{(1-P)+(P/N)}$
where P is the percentage of the workload that is parallelizable.
![](https://deeplearningsystems.ai/figures/ch07-04.png)

## Memory and Bandwidth
The time to read data from memory is often the main hindrance to performance. One way to reduce exposure to the memory bandwidth is to use a cache memory hierarchy that stores frequently or immediately accessed elements closer to the compute element.

Memory can be described by its capacity (bytes) and data transfer rate or bandwidth (bytes per second). The bandwidth (BW) [can be computed](https://en.wikipedia.org/wiki/Memory_bandwidth#Bandwidth_computation_and_nomenclature) as follows:
$\text{BW } = f_{\mathit{mem}} \times \mbox{number of interfaces} \times \mbox{transfers per clock} \times \mbox{mem bus width},$
where $f_{\mathit{mem}}$ is the memory frequency, the interfaces are typically 2 (dual-channel configuration) in modern processors, and the transfers per clock are 2 for memories that transfer on both the rising and falling clock edge (such as DDR) and 1 otherwise.

The memory types used in production in increasing order of accessed time and, equivalently, in increasing order of memory density (bytes per silicon area) and decreasing monetary cost per byte are as follows:

1. processor registers
2. SRAM: scratchpad, cache (typically with multiple levels)
3. DRAM: HBM2/E, GDDR6, DDR4/5, LPDDR4/5.

Types of random-access memory: **dynamic RAM (DRAM)** and **static RAM (SRAM)**. 
- SRAM uses a bistable circuit design that is faster but more expensive and requires four to six transistors per bit.
- DRAM is slower but less expensive and requires only one transistor (and a capacitor) per bit, and hence it has higher memory density.
- DRAM memory must be frequently refreshed to avoid losing information as the charge leaks.
- SRAM does not require frequent reads and writes. 
- Both DRAM and SRAM are volatile memories; that is, they lose the stored bits when the power is off.

There are [two main types of SRAM](https://dl.acm.org/doi/10.1145/1273440.1250707) configurations: caches and scratchpads. 
- Caches are common in CPUs and GPUs to support general-purpose workloads. 
- Scratchpads are common in embedded and dedicated hardware, such as ASICs and DSPs, for static graph-based workloads to reduce power consumption.

#### Levels of cache memory in modern computing:

1. Modern CPUs have three levels of cache: L1 (Level 1), L2 (Level 2), and L3 (Level 3).
2. L1 cache is the smallest, closest to the CPU, and offers the fastest access time.
3. CPU processors have separate L1 caches for data (L1d) and instructions (L1i).
4. L2 and L3 caches are shared for data and instructions.
5. Modern GPUs have two levels of cache.
6. The memory block loaded from the main memory to cache is called a cache line.
7. Loading a full cache line can waste bandwidth and storage if the memory accesses are sparsely strided.

#### Cache replacement policy algorithms:

Variants of the Least Recently Used (LRU) eviction policy are common
- LRU means the cache tracks and evicts the least recently accessed page when adding a new page.
- Example Adaptive Replacement Cache ([ARC](https://en.wikipedia.org/wiki/Adaptive_replacement_cache))
- ARC tracks frequently used, recently used, and recently evicted pages.

### Memory Structures in Computing

#### Scratchpad Memory
- **Efficiency and Software Complexity**: Scratchpads are efficient with 1 - 2 clock cycles per memory access, but require sophisticated software.
- **Functionality**: Manages memory accesses and replacement policy; involves explicit software-controlled DMA transfers.
- **Performance Sensitivity**: Mismatch in memory accesses can cause significant performance degradation.
- **Usage**: Best suited for DL workloads with static graphs where data accesses are predictable at compile-time.
- **Cost-Benefit Analysis**: Power and execution time savings can be substantial over the model's lifetime, potentially outweighing software complexity.

#### Hybrid Memory Systems
- **Cache and Scratchpad Combination**: Nvidia architectures (except Pascal) use both configurations for application-specific optimizations.
- **Terminology**: Nvidia refers to scratchpad and cache as shared and automatic memory, respectively.
- **Research**: Efforts are ongoing for a unified configuration (e.g., Stash and Buffets).

#### Cache Types and Tradeoffs
1. **Fully Associative Cache**
   - **Access Time**: Slowest, minimizes conflicts.
2. **Direct Mapped Cache**
   - **Access Time**: Fastest, maximizes conflicts.
3. **N-way Set-Associative Cache**
   - **Balance**: Compromise between access time and conflicts.

- **Common Usage**: Most CPU caches in production are N-way set-associative.
- **Design Implications**: Cache associativity is important for DL topology design; optimal dimensions can reduce cache conflicts.
### DRAM Types and Characteristics

#### Overview of Synchronous DRAM (SDRAM)
- **Cost and Efficiency**: SDRAM is cheaper and requires less silicon area than SRAM but consumes more energy and has longer access times.
- **Types**: Includes DDR, HBM, GDDR, and LPDDR, with various generations within each type.

#### DDR (Double Data Rate) Memory
1. **DDR4**
   - **Usage**: Common in servers, workstations, laptops, and some inference accelerators.
   - **Improvement through Channels**: More memory channels improve bandwidth but are limited by package design.
2. **DDR5**
   - **Advantages**: Offers higher bandwidth and density.
   - **Support**: Expected in Intel Sapphire Rapids and AMD Genoa processors.

#### High-Bandwidth Memory (HBM)
- **Usage**: Dominant in GPUs and accelerators for training, HPC, and cryptomining (e.g., Nvidia {P, V, A}100, Habana Gaudi, Google TPU).
- **Specification**: HBM2 features a 1024-bit wide interface with 307 GB/s bandwidth per stack.
- **Design**: Utilizes an interposer for connection, leading to higher bandwidth and lower power but at a higher manufacturing cost.
- **Market Position**: Considered as 2.5D memory technology with potential cost reduction upon wider adoption.

#### Graphics DDR (GDDR)
- **GDDR6**: Employed in latest gaming graphics cards and data center inference GPUs (e.g., Nvidia T4).
- **Comparison with HBM**: Cheaper and lower latency than HBM but with lower bandwidth and density.

#### Low-power DDR (LPDDR)
- **LP-DDR4 and LP-DDR4X**: Common in low power devices like mobile phones, characterized by low latency.
- **LP-DDR5**: Found in latest mobile phones, expanding to tablets, ultra-thin notebooks, automotive, and possibly DL inference processors.
### Set-Associative Cache

#### Matrix Storage
- Stored in **column-major order**.
- Consecutive column values in memory.
- Number of rows is the **leading dimension**.

#### GEMM Data Reuse
- Accessing data in blocks to fit in the cache.
- Reuse values to avoid cache misses.

#### Cache Mapping and Leading Dimensions
- Blocks mapping to the same cache set can lead to evictions.
- **Optimal leading dimensions**: Multiples of 16, but not of 256.
- Affects DL model design (e.g., RNN layers with 1008 vs. 1024 units).

#### Set-associative Caches
- `N-way` set-associative caches divide the cache into sets.
- Memory addresses mapped based on index bits.
- Accessing a full set evicts one entry.

#### Cache Sets and Address Mapping
- Number of sets: `SN = sizeof(cache) / (N x sizeof(cache line))`.
- Addresses modulo `WN` map to the same set.
- `WN = sizeof(cache) / N`.

#### Example Configuration
- 8-way set-associative `L1` cache.
- 64 sets, 64-byte cache lines, 32KiB total size.
- Address interval for same set mapping: 4096 bytes.

## Roofline Modeling 20 mins

Roofline [modeling](https://dl.acm.org/doi/10.1145/1498765.1498785) estimates the maximum performance that a computational kernel or set of kernels can attain on a particular hardware.
It has 3 components:
1. processor peak performance in operations (ops) per second (ops/s or OPS)
2. memory bandwidth in bytes per second (B/s)
3. kernel arithmetic intensity (ops/B).

A processor's peak performance depends on the frequency, number of cores, number of ops per core per cycle, and hardware's efficiency. This efficiency can be estimated by using [CS Roofline Toolkit](https://bitbucket.org/berkeleylab/cs-roofline-toolkit/). Running a suite of micro-kernels or an appropriate stream benchmark provides a more accurate observable bandwidth. 
#### Operational intensity (OI)
The ratio of the number of operations required to compute a kernel divided by the bytes read from DRAM memory. 
- The number of operations depends on the kernel and is typically independent of the processor.
- The number of bytes depends on both the kernel and the local SRAM memory size; a large SRAM facilitates data reuse.
- A system with no SRAM is assumed to illustrate the worse case OI.
- If every operand is read from DRAM and every result is written to DRAM, then each operation (two reads and one write) would have an arithmetic intensity of $1/(3\times \mathit{sizeof} (\mbox{datatype}))$
	- In the given scenario, for each operation, there are two read operations (to fetch the operands) and one write operation (to store the result).
- In the ideal case, the operands and result fit in SRAM and the OI is:
$
\mathit{OI}_{\mathit{kernel}} = \frac{\mathit{ops}}{\mathit{sizeof} (\mbox{input activations}) + \mathit{sizeof} (\mbox{weights})+ \mathit{sizeof}(\mbox{output activations})}.
$
#### Roofline model

![](https://deeplearningsystems.ai/figures/ch07-07.png)

The graph, often referred to as a "roofline model," is a visual representation used to understand the performance of a computational kernel on a particular hardware architecture. It relates operational intensity and performance, showing the maximum attainable operations per second (ops/s) for a given kernel. Here's a breakdown of its components:

1. **Operational Intensity (OI)**: This is on the x-axis and is measured in operations per byte. It represents how many operations are performed for each byte of memory traffic. Higher operational intensity means more computation with less data movement.

2. **Attainable Operations per Second**: The y-axis shows the performance in terms of operations per second that can be achieved.

3. **Peak ops/s**: This is the maximum number of operations per second the system can theoretically perform. It's represented by the horizontal red line. No kernel can exceed this performance due to the limitations of the processor's computational capabilities.

4. **Bandwidth-Bound**: The sloped line represents the performance limit imposed by the system's memory bandwidth. If a kernel's performance falls on this line, it means the kernel is "bandwidth-bound," which means its performance is limited by the memory system's ability to supply data.

5. **Compute-Bound**: The vertical dashed line marks the transition point where kernels become "compute-bound." If a kernel is on the right side of this line, its performance is limited by the peak computational capabilities of the system, not by memory bandwidth.

6. **Ridge Point**: This is the intersection of the bandwidth-bound line and the peak performance line. It indicates the operational intensity above which the kernels become compute-bound.

7. **Kernels 1, 2, and 3**: These points represent different computational kernels plotted based on their operational intensity and attainable ops/s. For example:
   - Kernel 1: It is bandwidth-bound because it lies on the sloped line, indicating its performance is limited by memory bandwidth.
   - Kernel 2: It is also bandwidth-bound but has a higher operational intensity than Kernel 1.
   - Kernel 3: It is compute-bound, as it lies to the right of the ridge point, indicating its performance is limited by the processor's compute capabilities, not by memory bandwidth.


The relation between roofline and computation time is as follows: the time $T$ it takes to execute a kernel, assuming perfect overlap of communication and computation, is:
$
T= \max \left( \frac{\text{number of ops to compute kernel}} {\text{peak processor OPS}}, \frac{\text{bytes to read from memory}} {\text{peak memory bandwidth}} \right).
$
**Data reuse is key to achieving high OI.** 
Data reuse means reusing the operands or the result for multiple cycles. The OI for a kernel function can vary considerably depending on how much data is reused.

- **Operational Intensity (OI):** The text begins by explaining the concept of "Operational Intensity" (OI), which is a measure of the number of operations performed per byte of memory traffic. **High OI is desirable as it means the computation is doing more work relative to the amount of data it moves, which is efficient use of the memory bandwidth.**

- **Data Reuse:** The document states that achieving high OI often involves reusing data, so that the same data is used for multiple operations before being discarded. This reduces the need to fetch new data from memory, which is a slow operation compared to the speed of processing.

- **CNN vs. GEMM Kernel:** It contrasts the OI of a typical CNN kernel (high OI) with that of a GEMM kernel used in MLPs (Multi-Layer Perceptrons), RNNs (Recurrent Neural Networks), or other fully-connected layers (low OI).

- **OI Calculation for GEMM:** It then provides a formula for calculating the OI of a GEMM operation. The GEMM operation is essentially matrix multiplication, where matrix A (with dimensions MxK) is multiplied with matrix B (with dimensions KxN) to produce matrix C (with dimensions MxN). The formula for OI in this context is:
$
\mathit{OI} = \frac{2\mathit{MKN}}{\mathit{sizeof}(\mbox{datatype}) \times (2\mathit{MN} + \mathit{MK} + \mathit{KN})},
$
  The numerator accounts for the number of operations (2 for each multiply and add), and the denominator accounts for the amount of data movement (considering the sizes of matrices A, B, and C, as well as the size of the data type in bytes).

- **OI Calculation for Convolution:** The document also gives a formula for the OI of a convolution operation in a neural network. A convolution involves sliding a smaller matrix (kernel) across a larger matrix (input feature map) and performing element-wise multiplications and additions. The formula provided is:
$
\mathit{OI} = \frac{2NKCRS \tilde{H}\tilde{W}} {\mathit{sizeof}(\mbox{datatype}) \times (2N\tilde{H} \tilde{W}K + KCRS + NHWC)}.
$
  In this formula, N  is the batch size,  K is the number of output feature maps, C is the number of input feature maps, $R \times S$  is the size of the kernel, and $H' \times W'$ are the dimensions of the output feature map. The formula accounts for the operations involved in the convolution and the data movement between the memory and the compute units.

These formulas are used to measure and optimize the efficiency of neural network operations, particularly in hardware that is designed for high-performance computing, such as GPUs or specialized AI accelerators. The aim is to maximize the computational work done per unit of data transferred to and from memory, as memory access is often the bottleneck in such computations.

1. **GEMM Data Reuse**: 
   - When matrices do not fit in SRAM, GEMM can still be efficient by reusing data.
   - In the matrix multiplication C = A x B, each element of B is used M times (once for each element of A in the same row).
   - Each element of C is the result of K multiplications (for K elements in a row of A and a column of B).
   - Data reuse in GEMM is linked to the batch size N; with N=1, there's no advantage of batching in terms of weight reuse.

2. **Convolution Data Reuse**: 
   - More data reuse is possible in convolution operations than in GEMM.
   - Weights of a convolutional filter can be applied across the entire input tensor's N dimension, enhancing reuse.
   - Activation maps can be reused across all the weights of the filter for different parts of the input.

In short, both GEMM and convolution operations in deep learning algorithms are designed to maximize the reuse of data, which is crucial for computational efficiency, especially when dealing with limitations in memory capacity like SRAM.

## Processor Designs

In this section, we review instruction sets, and the type of processors used in DL, specifically CPUs, GPUs, FPGAs, CGRAs, DSPs, and ASICs, used separately or as components of a heterogeneous design.
![](https://deeplearningsystems.ai/figures/ch07-08.png)

#### Instruction set architecture (ISA):
Defines the operators, data types, and memory management for an abstract computer architecture.
Different processors with different frequencies and memory sizes can implement the same ISA and execute the same binary. The specific implementation is called a _microarchitecture_.

Two general types of instruction sets are the **complex instruction set computer (CISC)** and the **reduced instruction set computer (RISC)**.
- The CISC ISA aims to execute multiple low-level operations per instruction.
- The RISC ISA is smaller and simpler than the CISC ISA and aims to provide higher IPC rates.

The most common instruction sets and the typical devices that use them are as follows:

- CISC x86 ISA in computer processors from laptops to supercomputers
- RISC Arm ISA in smartphones with some adoption in laptops (such as the [Apple M1](https://en.wikipedia.org/wiki/Apple_M1)) and single-board computers and starting to enter the server market by [Ampere](https://amperecomputing.com/), [AWS-Graviton](https://aws.amazon.com/ec2/graviton/), [Marvell](https://www.marvell.com/server-processors/thunderx-arm-processors/thunderx-cp/), and [Huawei](https://www.forbes.com/sites/huawei/2019/01/22/huawei-unveils-industrys-highest-performance-arm-based-cpu/)
- RISC open-sourced RISC-V ISA in academia with some small traction in production at [Alibaba](https://www.t-head.cn/product/c910) and [SiFive](https://www.sifive.com/cores/u74-mc)
- RISC Power ISA in IBM POWER microprocessors and some supercomputers.
#### SIMD and SIMT

- **SIMD (Single Instruction, Multiple Data)**
  - Utilized by CPU vector processors.
  - Executes a single instruction across multiple data points within a core's execution unit.
  - Example: AVX-512 in CPUs processes 16 floating-point 32-bit (fp32) values simultaneously.

- **SIMT (Single Instruction, Multiple Threads)**
  - Coined by Nvidia and used by GPU vector processors.
  - Extends SIMD by applying one instruction to multiple threads.
  - Nvidia GPUs: Operate on a warp (32 threads).
  - AMD GPUs: Operate on a wavefront (64 threads).
  - GPUs feature coalesced loads for efficient data access across threads.
#### Instruction Sets and Extensions

- **Existing SIMD Extensions**
  - SSE, MMX, AVX, AVX-2, and AVX-512 (also referred to as AVX-3).
  
- **Arm Extensions**
  - NEON and Scalable Vector Extensions (SVE) for the Arm Instruction Set Architecture (ISA).
#### Differences in ISA Extensions

- **Key Distinctions**
  - Variation in the count of supported instructions.
  - Differences in data size each instruction can manage.
  
- **Data Size Handling**
  - AVX-512: Can handle 512-bit data sizes.
  - AVX-2: Limited to 256-bit data sizes.
#### Simultaneous Multithreading (SMT) and Hyper-Threading

- **SMT/Hyper-Threading Overview**
  - Allows a single CPU core to run multiple threads simultaneously.
  - Intel brands this technology as Hyper-Threading.
  - Aims to improve utilization of execution units (EUs) that might be idle.

- **Performance Considerations**
  - For optimized kernels, EUs may not be idle, thus SMT might not boost performance significantly.
  - With high operational intensity (OI) kernels, SMT could hinder performance due to the overhead of thread switching.
  - Performance impact of SMT varies and requires testing on specific workloads.
#### Very Long Instruction Word (VLIW)

- **VLIW Characteristics**
  - Executes multiple instructions in parallel within a single long instruction word.
  - Optimized for regular, predictable code to allow compilers to maximize parallelism.

- **VLIW Implementations**
  - Historical: Itanium processors (now retired).
  - Current: Habana AI processors, Google's TPU v2, and possibly TPU v3 and v4.
#### Dataflow Parallelism

- **Systolic Architectures**
  - Comprises multiple simple processing engines (PEs) arranged in a mesh pattern.
  - PEs perform simple computations (e.g., multiply-accumulate operations) and pass results to neighboring PEs.

- **Performance and Efficiency**
  - Results in high throughput due to collective processing.
  - Power-efficient due to simple circuitry and short interconnects between PEs.
  
- **Hardware Integration**
  - Found in specialized hardware, including domain-specific additions to CPUs and GPUs (e.g., Intel's AMX, Nvidia's tensor cores).

- **Compiler and Programming Requirements**
  - Achieving near-peak performance requires advanced compilers or programming that accounts for memory hierarchy.
  - Inefficiencies in memory access patterns can severely degrade performance.

### Central Processing Unit (CPU)

- **CPU Components**
  - Includes RAM, registers, and execution units.

- **Server CPU Characteristics**
  - Generally has faster but fewer cores compared to GPUs or dedicated deep learning (DL) accelerators.
  - Offers a balance for complex workloads, benefiting both parallel and serial processing.
  - Single-core performance is enhanced by high-frequency capabilities.

- **Amdahl's Law on Performance**
  - Execution time does not decrease linearly with increased core count.
  
- **Programming and Flexibility**
  - CPUs offer maximum flexibility and are typically easier to program.
  - Built-in logic for control flow like branch prediction.
  
- **Power Consumption**
  - Higher power needed to decode and execute instructions per core.
  
- **Usage for Parallel Workloads**
  - For static graph workloads, dedicated processors may be more efficient.

### Graphical Processing Unit (GPU)

- **GPU Components**
  - Comprises RAM, registers, and compute units.

- **Design for Parallel Tasks**
  - Initially created for image processing tasks.
  - Now also targets deep learning tasks like matrix multiplications.

- **CPU vs. GPU Core Functionality**
  - CPU cores can operate independently, while GPU cores cannot.
  - GPU cores execute the same instruction in sync with their group (warp or wavefront).

- **Flexibility and Efficiency**
  - CPU cores are more flexible; GPU cores are more energy-efficient.

### GPU Core Limitations and SMT

- **Independence and Shared Resources**
  - GPU cores do not have independent operation like CPU cores.
  - Shared SRAM among GPU cores in a warp or wavefront.
  
- **Register Files and Latency**
  - GPUs have large register files to support many threads, resulting in higher throughput but increased latency.

### Memory Bandwidth and Design

- **Bottlenecks**
  - Limited memory bandwidth is a common bottleneck.

- **Mitigation Strategies**
  - Increase the SRAM for each compute unit.

- **Design Variations**
  - Nvidia's V100 with large High Bandwidth Memory (HBM2) and small local SRAM.
  - Graphcore's Colossus with no DRAM and large SRAM units.

- **Batch Size and Efficiency**
  - More local SRAM enables higher efficiency with smaller batch sizes.

### Field-Programmable Gate Array (FPGA)

- **FPGA Characteristics**
  - Contains small, reconfigurable compute elements.
  - Beneficial for adapting to new workloads.

- **Usage and Challenges**
  - Used for simulating ASICs and processor designs.
  - Long reprogramming times and limited deep learning software tools.

### Coarse-Grained Reconfigurable Array (CGRA)

- **CGRA Overview**
  - Considered as an FPGA with coarser reconfigurability.
  - Easier programmability but less flexibility than FPGA.

- **Adoption Limitations**
  - Limited software tools affecting widespread adoption.

### Digital Signal Processor (DSP)

- **DSP Specialization**
  - Optimized for signal processing tasks with specialized ISA.
  
- **Modularity**
  - Has a consistent base ISA and specialized extension ISAs.

- **Reconfigurability and Programmability**
  - Not reconfigurable but programmable with a competent compiler.

- **Utilization**
  - Often integrated in heterogeneous designs alongside other hardware.

### Application-Specific Integrated Circuit (ASIC)

- **ASIC Optimization**
  - Offers the best performance for specific applications with limited flexibility.

- **Control Logic and Programming**
  - Limited control logic; relies on programmer/compiler for data management.
  
- **Compiler Maturity**
  - Requires advanced programming or a mature deep learning compiler.

- **Design Recommendations**
  - Pack many transistors for MACs within power and size constraints.
  - Allocate silicon for SRAM and I/O operations.

### Dataflow Parallelism in MACs

- **MAC Implementation**
  - Multiple architectures reviewed by Chen et al. and Sze et al.
  
- **PE Array Configuration**
  - Consists of a PE array with ALU or FPU for MAC operations, local control, and possibly local SRAM.

- **Network-on-Chip (NoC)**
  - PEs connected through a NoC with global SRAM for data management.

### Dataflow Architectures

- **Types of Dataflow Architectures**
  - No Local Reuse
  - Weight-Stationary
  - Output-Stationary
  - Row-Stationary

### No Local Reuse Architecture

- **Design**
  - Eliminates local PE memory to maximize global SRAM size.
  
- **Data Handling**
  - Weights and activations pass directly from global SRAM to PEs.
  - Accumulated sums are passed between neighboring PEs in a row.

### Weight-Stationary Architecture

- **Weight Handling**
  - Stores weights in PE's local memory to maximize weight reuse.

- **Data Flow**
  - Activations are broadcast to relevant PEs.
  - Accumulated sums flow between neighboring PEs in a row.

- **Efficiency**
  - Efficient for traditional convolutional layers with high weight reuse.
  - Less efficient for fully-connected or depthwise separable convolutional layers.

### Output-Stationary Architecture

- **Accumulated Sum Reuse**
  - Keeps accumulated sums in PE's local memory to maximize their reuse.

- **Broadcast and Flow**
  - Weights are broadcast to all relevant PEs.
  - Activations flow between neighboring PEs in a row.

### Row-Stationary Architecture

- **Data Reuse Optimization**
  - Maximizes reuse across both weights and activations.

- **Flow of Data**
  - Accumulated sums flow vertically from bottom to top columns.

- **Performance per Watt**
  - Proposed by Chen et al. to provide the best efficiency for convolutions and fully-connected layers.

### Computational Kernel Distribution

- **dMazeRunner**
  - Tool for exploring efficient splitting of computational kernels in dataflow accelerators.

### Customization for Sparse Matrix Multiplications

- **ExTensor Accelerator**
  - Developed by Nvidia researchers.
  - Optimized for sparse multiplications by avoiding unnecessary operations.

### Compute-in-Memory Processors

- **Analog Computations**
  - Uses analog components for computation.

- **Components**
  - Tunable resistors for weights.
  - Voltage for activations.
  - Measured current for accumulated sums.

- **Challenges**
  - Requires precise tuning.
  - DAC and ADC components can limit power efficiency gains.

### Neuromorphic Processors

- **Brain-Inspired Design**
  - Aims to reduce power consumption for deep learning hardware.
  
- **Neural Network Type**
  - Utilizes spiking neural networks (SNNs).

- **Power Efficiency**
  - Operates at very low power but has limitations in complex domains.

- **Differentiation Limitation**
  - The input-output function is nondifferentiable, limiting the use of backpropagation.
## High-Performance Interconnects

The types of interconnects discussed in this section are host-to-device, such as PCIe and CXL and device-to-device and host-to-host, such as InfiniBand, OmniPath, and Ethernet/IP. 

Host-to-host and device-to-device interactions focus on supporting parallel computation involving multiple instances of the host or the device, such as for distributed training.
#### Serializer/Deserializer (SerDes)
Is used for data transfers, converts parallel data to serial data to be transmitted over a much higher speed connection and vice versa. Proprietary SerDes are used in Nvidia's [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/), Nvidia Mellanox [InfiniBand](https://www.mellanox.com/products/interconnect/infiniband-overview), and AMD's [Infinity Fabric](https://en.wikichip.org/wiki/amd/infinity_fabric). SerDes standards include the Peripheral Component Interconnect Express (PCIe) bus and Ethernet.

The theoretical peak bandwidths per direction using 16 lanes (written as x16) are doubling in almost every generation as follows:
- PCIe 3.0 x 16: 16 GB/s (most common)
- PCIe 4.0 x 16: 31.5 GB/s (recently available)
- PCIe 5.0 x 16: 63 GB/s (future)

#### Compute Express Link ([CXL](https://www.computeexpresslink.org/about-cxl)):
is a new high-speed CPU-to-device interconnect that can maintain memory coherency between the CPU and device memory. CXL and PCIe are electrically identical. However, CXL and PCIe have different and incompatible protocols.'

### High-Performance Network Protocols

#### Remote Direct Memory Access (RDMA)
- **Protocols Supporting RDMA**: 
  - RDMA over Converged Ethernet (RoCE)
  - InfiniBand
  - iWARP
  - Omni-Path
- **Functionality**: These protocols facilitate efficient scaling by allowing the network interface controller (NIC) direct access to memory.

#### Utilization of Memory Subsystem
- **Direct NIC Access**: RDMA enables NICs to access memory regions remotely, bypassing CPU bottlenecks.
- **Technological Support**: Various technologies are employed to map memory for RDMA, ensuring efficient data destination targeting.

#### Device-to-Device Memory Access
- **CPU Memory Mapping**: Involves the CPU's memory controller for RDMA to CPU memory.
- **PCIe Peer-to-Peer Transactions**: Enables direct device communication for supported devices.
- **NVLink and Infinity Fabric**: 
  - NVLink is used for Nvidia GPU intercommunication.
  - AMD GPUs utilize Infinity Fabric.

#### Proprietary Interfaces for Accelerators
- **Device-Specific Interfaces**: Accelerators may use customized interfaces for optimized device-to-device memory sharing.

### Protocol vs. Physical Interconnect

#### Chip-to-Chip Interconnect Protocols
- **PCIe**: The PHY Interface for PCIe (PIPE) is utilized for connections like PCIe, CXL, and USB 3.0 SuperSpeed.

#### Standards and Definitions
- **Organizations**: 
  - PCI-SIG oversees PCIe standards.
  - IEEE 802.3 is responsible for Ethernet standards.
  - The Optical Internetworking Forum (OIF) supports the development of interoperable SerDes devices.
- **Interoperability**: Emphasizes the development of devices capable of supporting multiple protocol types.
### Physical Network Topologies

In the early days of high-performance computing (HPC) interconnects, low-radix topologies were the norm. Higher radix network topologies are becoming more common as the pin bandwidth increases and can be efficiently partitioned across multiple ports

![](https://deeplearningsystems.ai/figures/ch07-15.png)


## Processors in Production:

1. **General Overview**:
   - The landscape of processors for DL is expanding, with numerous processors in production, development, and research.
   - A consolidation is expected, similar to the trend observed with DL frameworks.

2. **CPUs in Deep Learning**:
   - CPUs, especially server CPUs, are primarily used for inference and training smaller models or models requiring large memory.
   - Notable CPU advancements include Intel's 2nd-generation Xeon Scalable processors (up to 56 cores) and AMD's 2nd-gen EPYC (up to 64 cores).
   - CPUs are evolving to include specialized circuits and instructions for DL computations, such as Apple's A13 CPU and Arm's Armv8-A architectures.
   - Intel introduced AVX512 ISA extensions and Advanced Matrix Extensions (Intel AMX) for acceleration in DL.

3. **GPUs in Deep Learning**:
   - GPUs are favored for parallel tasks like batch GEMMs and convolutions, thanks to thousands of threads.
   - Nvidia added tensor cores in its Volta, Turing, and Ampere microarchitectures for enhanced DL computations.
   - Nvidia is developing the RC-18 architecture for low power inference and possibly integrating it into the Hopper microarchitecture.

4. **AMD's Role in GPUs**:
   - AMD's success in GPUs is less than Nvidia's, mainly due to a limited DL software ecosystem.
   - AMD transitioned from GCN to RDNA and CDNA microarchitectures, adding bfloat16 support in their ROCm MIOpen DL software library.

5. **Intel's GPU Initiatives**:
   - Intel plans to release a discrete GPU based on the Xe architecture, optimized for HPC modeling and DL training.

6. **FPGAs in Machine Learning**:
   - FPGAs are being adopted for ML workloads, with Microsoft providing ML services on FPGAs.
   - Intel and Xilinx, the primary FPGA makers, offer high-level software libraries for DL primitives.

7. **Specialized Deep Learning Processors**:
   - Established companies and startups are developing specialized DL processors, often combining ASICs and DSPs.
   - Google's TPU, AWS Inferentia, Alibaba's Hanguang 800 NPU, and Baidu's Kunlun are examples of such processors.

8. **Other Specialized Processors**:
   - Intel's Habana processors, Qualcomm's Cloud AI 100, and Huawei's Ascend processors are notable in this category.

9. **Innovations by Startups**:
   - Cerebras Systems, Graphcore, SambaNova, and Groq are prominent startups with unique processor designs.
   - These companies focus on processors with large SRAM for efficient training and inference.

10. **Challenges and Potential Solutions**:
   - Large SRAM in processors like CS-1, Graphcore's IPU, and Groq's TSP reduces data fetching time but faces challenges in storing activations.
   - Pipeline parallelism and recent algorithmic improvements may help realize the full potential of these chips.

The text indicates an evolving and competitive landscape in the development of processors for deep learning, with advancements in CPU and GPU technologies and the rise of specialized DL processors and startups.
## Platforms Strengths and Challenges:

1. **General Overview of DL Platforms**:
   - DL platforms need to balance memory, bandwidth, general-purpose compute, and dedicated compute.
   - They must be programmable and flexible for different workloads.
   - Focus is on training platforms, including Nvidia DGX, Google TPU POD, Habana HLS-1, Graphcore M2000, Cerebras CS-1, AMD CDNA-based, Hyve Solutions Catalina, and Facebook Zion platform.

2. **Nvidia DGX and POD**:
   - Utilizes GPU-accelerated servers and racks.
   - Features NVLink and NVSwitch for intra-server connectivity, and InfiniBand for inter-server connectivity.
   - Known for mature software, robust ecosystem, and economies of scale.
   - Introduces multi-instance GPU (MIG) in Ampere microarchitecture for efficient small-batch inference.

3. **Google TPU POD**:
   - Connects TPUs with custom interconnects for single-workload operations.
   - Offers over 100 PFLOPS on TPUv3 POD, with a flat 2D torus network topology.
   - First-to-market data center DL accelerator, offering cost advantages.
   - Challenges include low utilization for small-batch processing and software complexity.

4. **Habana HLS-1**:
   - Developed by Habana, an Intel company.
   - Features Gaudi cards with RDMA engines, using Ethernet for communication.
   - No in-built CPU; relies on user-defined CPU to Gaudi ratio.
   - Decouples management and scale-out traffic to mitigate bottlenecks.

5. **Graphcore M2000**:
   - Contains 4 MK2 IPU devices, with a 2D torus physical topology.
   - Offers large IPU SRAM capacity for efficient, small-batch processing.
   - Lacks fp32 accumulation in MK2 IPU fp16 MACs, potentially affecting training accuracy.

6. **Cerebras CS-1**:
   - Consolidates multiple DL chips into a large processor for distributed training.
   - Offers high bandwidth and large SRAM capacity.
   - Challenges include rigidity in compute, bandwidth, and CPU ratios and underutilization of compute resources.

7. **AMD CDNA-based GPU Servers**:
   - Utilizes AMD Infinity Fabric Link for GPU connectivity.
   - Focuses on the high-bandwidth, low latency Infinity Architecture.
   - Growing ecosystem with hyperscalers, offering potential ease in DL market entry.

8. **Hyve Solutions Catalina**:
   - Uses 3rd-generation Intel Xeon Scalable processors.
   - Versatile for DL training, inference, ML, HPC, and general workloads.
   - Faces lower performance compared to dedicated DL training platforms.

9. **Facebook Zion Platform**:
   - A unified training platform with disaggregated memory, compute, and network components.
   - Features vendor-agnostic OCP accelerator modules (OAMs).
   - Strength lies in its capacity to support diverse DL workloads.
   - Challenges include software complexity and efficient utilization of compute resources.
## Evaluating Devices and Platforms:

1. **Total Cost of Hardware Operation**:
   - Includes the cost of hardware, maintenance over its lifetime, and software engineering for programming.
   - Ease of programming for various topologies is crucial for high performance.

2. **Use Cases of Different Products**:
   - Training hardware: Optimized for throughput.
   - Inference hardware: Optimized for latency.
   - Edge hardware: Optimized for power and size.
   - Performance can vary based on the processor and topology.

3. **Evaluating Topology in Hardware Context**:
   - Metrics: Statistical performance, computational performance, power consumption.
   - Importance of evaluating topology within the specific hardware architecture.

4. **Platforms for Evaluating DL Hardware**:
   - FireSim: FPGA-based hardware simulator on AWS FPGA instances.
   - Nvidia DL Accelerator (NVDLA): Integrated on FireSim.
   - SMAUG and Eyexam packages: Model performance of topology on an accelerator design.
   - ParaDnn tool: Benchmarks DL platforms against TPUs, GPUs, and CPUs.
   - Studies by Wang et al. and Dai et al. provide performance comparisons on various hardware.

5. **Development of DL Benchmarks**:
   - DeepBench: Evaluates primitives.
   - DAWNBench: Assesses performance and cost on public cloud services.
   - MLPerf: Popular benchmark backed by a consortium, evaluates performance across established models.
   - Caution against overfitting hardware and software designs to benchmarks.

6. **Importance of Programmability and Flexibility**:
   - Need for benchmarks that measure a platform’s ability to support diverse workloads.
   - Emphasis on programmability and flexibility in benchmarks.

7. **DL Hardware Components and Design Space**:
   - Discussion on performance vs. ease-of-programmability trade-offs across various hardware designs.
   - Recommendation for high-level ASIC design for optimizing OPS per watt.
   - Challenges with accessing DRAM memory and pipeline parallelism in large SRAM training processors.

8. **Flexibility in Platform Design**:
   - Need for a flexible platform to adapt to unforeseen algorithmic and model innovations.
   - Importance of a disaggregated CPU-to-accelerator ratio, standard form factor module, and standard interconnect for scalability.
   - Flexibility enables evaluation and adoption of heterogeneous processors, avoiding vendor lock-in.

9. **Software Management and Compiler Challenges**:
   - Challenges in software-managed memory and extracting high performance.
   - Need for compilers to efficiently map programs to the target hardware.

10. **Future Outlook**:
   - Upcoming chapter to review compiler basics and describe standard compiler optimization passes for DL workloads.



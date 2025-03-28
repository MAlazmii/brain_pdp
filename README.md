# Parallel Brain Simulation (MPI-based)

This project implements a parallel simulation of a biologically-inspired brain model using **MPI (Message Passing Interface)**. The goal is to simulate neuron and nerve signal propagation across a distributed system using an **event-based coordination** pattern.

The original serial code was transformed into a scalable parallel version with proper MPI communication, node partitioning, and signal handling across ranks. The simulation uses structured input files to describe neurons, nerves, and their connections (edges), and produces summary output reports with signal statistics for analysis.

---

## Compilation

To compile the MPI-based simulation:

```bash
make
```

This will produce an executable called `brain_mpi`.

You can also manually compile with:

```bash
mpicc -O2 -Wall -o brain_mpi main.c neuron.c input_loader.c event_handler.c signal.c
```

Additionally, ensure you load the necessary modules:

```bash
module load openmpi/4.1.6
module load gcc/10.2.0
```

## Running the Simulation

Execute using `mpirun` with the desired number of MPI processes:

```bash
mpirun -np <num_processes> ./brain_serial <input_file> <simulation_duration_ns>
```

Example:

```bash
mpirun -np 4 ./brain_serial ../small 5
```

- `<input_file>`: Path to the brain graph input file.
- `<simulation_duration_ns>`: Number of nanoseconds to simulate.

---
## Input Format

The input graph file consists of:

- Neuron/Nerve definitions with:
  - `<neuron>` or `<nerve>`, followed by `<id>`, `<x>`, `<y>`, `<z>`, `<type>`
- Edge definitions with:
  - `<edge>`, `<from>`, `<to>`, `<direction>`, `<max_value>`, `<weighting_*>`

Example node block:
```xml
<neuron>
  <id>1</id>
  <x>0.5</x>
  <y>0.1</y>
  <z>0.8</z>
  <type>sensory</type>
</neuron>
```

---

## Output

After simulation, a file named `summary_report` is generated. It contains:

- Total number of neurons, nerves, and edges
- Input/output signal breakdown per nerve
- Total signal count received by each neuron

---


## Example Run (on Cirrus or local MPI system)

```bash
mpirun -np 8 ./brain_mpi ../large 5
```



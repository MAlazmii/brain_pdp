# Molecular Dynamics Simulation – README

## Project Overview

This project implements a gravitational N-body simulation with forces including central attraction, pairwise interactions, viscosity, and wind. The code has been optimized for performance using vectorization, inlining, and interprocedural optimizations. The simulation tracks the motion and collision behavior of particles over a series of timesteps.

---

## Compilation & Build Instructions

### Required Modules on Cirrus
Before compiling, load the required Intel compiler modules:

```bash
module load oneapi
module load compiler
```

### Compilation

To compile the project:

```bash
make clean
make
```

### Compilation Flags Used

```bash
-O3 -ipo -qopt-report=max -qopt-report-phase=vec,loop,ipo
```

Final optimization step used:

```bash
-Ofast -ipo -xHost
```

These flags enable aggressive optimizations, including interprocedural optimizations (`-ipo`) and architecture-specific tuning (`-xHost`).

---

## Running the Simulation

### Default Execution

```bash
./MD
```

This will run the simulation using the settings specified in `input.dat`.

### Input File

- `input.dat`: Contains initial particle positions, velocities, and simulation parameters.

### Output Files

- `output.dat100`, `output.dat200`, ..., `output.dat500`: Simulation output files after different numbers of timesteps.
- `output_reference.dat`: The original unoptimized baseline reference output.
- `gprof_report.txt`: If profiling is enabled with `-pg`.

---

## Correctness Validation

To validate correctness:

```bash
./diff-output output_reference.dat output.dat100
```

You should see:

```bash
max=0.000000
```

Also check for invalid values:

```bash
grep NaN output.dat100
```

Check difference lines:

```bash
./diff-output output_reference.dat output.dat100 | wc -l
```

Expected result for a perfect match is `1` (just the header line).

For later steps:

```bash
for step in 100 200 300 400 500; do
  echo -n "output.dat$step: "
  ./diff-output output_reference.dat output.dat$step | grep max=
done
```

---

## Performance Notes

- `-O0`: Baseline execution time ~152.78 seconds (100 timesteps)
- `-O3 -ipo`: Optimized time ~16.99 seconds
- `-Ofast -ipo -xHost`: Final optimized time ~**3.60 seconds**

Collision count shows stable physical behavior:

```
timestep 100
collisions 68404
100 timesteps took 3.603098 seconds
500 timesteps took 18.308584 seconds
```

---

## File Descriptions

| File                     | Description                                      |
|--------------------------|--------------------------------------------------|
| `MD.c`                  | Main simulation code                             |
| `control.c` / `control.o`| I/O and control logic                            |
| `util.c` / `util.o`     | Utility functions                                |
| `coord.h`               | Global declarations                              |
| `Makefile`              | Compilation instructions                         |
| `input.dat`             | Simulation input                                 |
| `output.dat*`           | Simulation outputs for different steps           |
| `output_reference.dat`  | Reference output for correctness validation      |
| `diff-output`           | Tool for comparing output files                  |
| `gprof_report.txt`      | Performance report (if profiling enabled)        |

---

## Validation Summary

| Timestep | max= Difference | Diff Lines | Interpretation                          |
|----------|------------------|------------|------------------------------------------|
| 100      | 0.000000         | 1          | Perfect match                          |
| 200-500  | Increasing       | ~16,400    | Floating-point drift due to chaos      |

---

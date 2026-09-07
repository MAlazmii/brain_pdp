# Parallel Brain Signal Simulation

**A C exploration of signal propagation through a graph of neurons and nerves, with serial and MPI implementations.**

The simulation reads a brain graph, generates signals at nerve nodes, applies neuron and edge weights, and records signal activity. The parallel implementation distributes neuron processing across MPI ranks and exchanges signals between them.

## Explore the implementation

| File | Purpose |
| --- | --- |
| [`code.c`](code.c) | Standalone serial implementation, including graph parsing and reporting |
| [`main.c`](main.c) | MPI initialization, node distribution, simulation loop, and gathering counts |
| [`brain.h`](brain.h) | Node, edge, signal, and event types; simulation constants |
| [`input_loader.c`](input_loader.c) | Graph parsing and node-to-edge linking |
| [`neuron.c`](neuron.c) | Signal generation, propagation, weighting, and report output |
| [`event_handler.c`](event_handler.c) | Local event dispatch and MPI signal transport |
| `small`, `medium`, `large`, `massive` | Included graph inputs |
| [`summary_report`](summary_report) | Included text report from a previous run |

There are six neuron categories and ten signal types. Signals are processed through per-node inboxes; the code tracks nerve input/output counts and total signals received by neurons.

## Serial starting point

The serial source uses a C compiler and POSIX timing facilities. From a local clone:

```sh
cc -O2 -Wall code.c -o brain_serial
./brain_serial small 1
```

The arguments are the graph filename and the number of simulated nanoseconds. `MIN_LENGTH_NS` sets a wall-clock interval used to advance simulation time; simulated nanoseconds are not real-time execution durations. A run writes `summary_report` in the working directory, replacing any existing file with that name. Preserve the included report before running.

These commands follow the source interface; they have not been validated in a fresh runtime environment.

## MPI implementation status

The parallel source requires an MPI development environment, but the committed build is incomplete:

- The `Makefile` targets `brain_serial`, uses `gcc`, and references `signal.c`, which is absent.
- MPI datatype initialization and several utility functions are declared or used without definitions in the modular source files.
- The committed `brain_mpi` binary does not establish that the current source can be rebuilt.

Treat the MPI files as an implementation to inspect and complete before running distributed experiments. No reproducible speedup benchmark or serial/parallel equivalence claim is made here.

## Scope

This is an abstract graph simulation for studying computation and communication. It is not a validated biological model. Randomness is seeded from wall-clock time, and the serial and MPI versions should not be assumed to produce identical results.

See [`LICENSE`](LICENSE) for the repository's existing license terms.

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
| [`simulation_utils.c`](simulation_utils.c) | MPI datatype and shared numerical/time helpers |
| `small`, `medium`, `large`, `massive` | Included graph inputs |
| [`summary_report`](summary_report) | Included text report from a previous run |

There are six neuron categories and ten signal types. Signals are processed through per-node inboxes; the code tracks nerve input/output counts and total signals received by neurons.

## Build and run

Use a C compiler and POSIX system. MPI additionally requires an MPI development environment providing `mpicc` and `mpiexec`.

```sh
make serial
./brain_serial tests/tiny.graph 1

make mpi
mpiexec -n 2 ./brain_mpi tests/tiny.graph 1
```

Arguments are a graph filename and a nonnegative number of simulated nanoseconds. `MIN_LENGTH_NS` converts a wall-clock interval to simulation steps; simulated nanoseconds are not elapsed physical execution time. New runs write `summary_report.generated` in the working directory. The included historical `summary_report` is preserved. Run separate experiments in different working directories to keep their generated reports.

The default `make` target builds the standalone serial program. Compiled executables are ignored and should be built locally.

## MPI maintenance and validation

The modular build supplies the missing datatype/helpers. Each rank loads graph connectivity and processes only its assigned nodes. Remote messages use retained nonblocking send buffers; iteration synchronization checks that all messages have been received before advancing. Rank zero controls the common stopping time, and report counts are gathered across ranks. Directed edges are attached only to their sending nodes. Invalid node IDs, non-finite capacities/weights, and capacities too small to reduce a signal fail explicitly.

```sh
python -m pip install pytest
python -m pytest --rootdir=. tests
```

Regression checks build both programs, verify a serial terminal neuron can receive signals without crashing, and exercise a tiny MPI graph with one and two ranks, signal delivery, malformed graph rejection, invalid duration/IDs/endpoints/capacity, and nerve counters owned by a non-root rank. They also load all four bundled graphs with a zero duration. MPI tests require local process communication. These are functional checks; they do not establish performance scaling, large-graph simulation correctness, or serial/parallel equivalence.

## Scope and reproducibility

This remains an abstract graph simulation, not a validated biological model. The MPI generator uses a fixed rank-dependent seed, but the number and ordering of operations remain wall-clock dependent, so output counts are not deterministic benchmarks. The serial implementation retains its original timing and randomness policy, including direct indexing that requires node IDs to match their zero-based input positions. Both implementations reject malformed records, invalid endpoints, non-finite numeric fields, and nonpositive edge capacities before simulation. Large inputs remain subject to the existing `MAX_NODE_ID` and inbox limits; inbox overflow can drop signals. Further scientific or performance claims require a separately designed validation experiment.

See [`LICENSE`](LICENSE) for the repository's existing license terms.

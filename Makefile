CC ?= cc
MPICC ?= mpicc
CFLAGS ?= -O2 -Wall -Wextra
MPI_SRC = main.c input_loader.c neuron.c event_handler.c simulation_utils.c

.PHONY: all serial mpi clean test
all: serial
serial: brain_serial
mpi: brain_mpi

brain_serial: code.c
	$(CC) $(CFLAGS) -o $@ $<

brain_mpi: $(MPI_SRC) brain.h
	$(MPICC) $(CFLAGS) -o $@ $(MPI_SRC)

test:
	python -m pytest --rootdir=. tests

clean:
	rm -f *.o brain_serial brain_mpi

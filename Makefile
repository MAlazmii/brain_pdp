# Makefile for Brain Simulation
CC = gcc
CFLAGS = -O2 -Wall
LDFLAGS =

SRC = main.c input_loader.c neuron.c signal.c
OBJ = $(SRC:.c=.o)
EXE = brain_serial

all: $(EXE)

$(EXE): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c brain.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o $(EXE)


#include "brain.h"
#include <stddef.h>

MPI_Datatype MPI_PackedSignal;

void create_mpi_signal_type(void) {
    int lengths[] = {1, 1, 1};
    MPI_Aint offsets[] = {offsetof(PackedSignal, type),
                          offsetof(PackedSignal, target),
                          offsetof(PackedSignal, value)};
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_FLOAT};
    MPI_Type_create_struct(3, lengths, offsets, types, &MPI_PackedSignal);
    MPI_Type_commit(&MPI_PackedSignal);
}

/* Same half-open interval as the original serial implementation. */
int getRandomInteger(int from, int to) {
    if (to <= from) return from;
    return rand() % (to - from) + from;
}

float generateDecimalRandomNumber(int to) {
    return ((float)rand() / RAND_MAX) * to;
}

time_t getCurrentSeconds(void) {
    return time(NULL);
}

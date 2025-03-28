// -------------------------------
// event_handler.c
// -------------------------------

#include "brain.h"
#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define TAG_SIGNAL 100

// -------------------------------
// External Variables
// -------------------------------
extern int rank, size;
extern MPI_Datatype MPI_PackedSignal;
extern int *id_to_index_map;
extern int *id_to_index;
extern int num_brain_nodes;

// -------------------------------
// Struct for MPI Communication
// -------------------------------
typedef struct {
    int type;
    int target;
    float value;
} PackedSignal;

// -------------------------------
// Get the owning MPI rank of a neuron by its ID
// -------------------------------
int getOwnerRankById(int id) {
    if (!id_to_index || id < 0 || id >= MAX_NODE_ID)
        return -1;

    int global_idx = id_to_index[id];
    if (global_idx == -1)
        return -1;

    int base = num_brain_nodes / size;
    int extra = num_brain_nodes % size;

    for (int r = 0; r < size; r++) {
        int start = r * base + (r < extra ? r : extra);
        int count = base + (r < extra ? 1 : 0);
        if (global_idx >= start && global_idx < start + count)
            return r;
    }

    return -1;
}

// -------------------------------
// Send a signal to a local or remote neuron
// -------------------------------
void sendSignalToRank(int tgt_id, struct SignalStruct signal, int sender_rank, int world_size) {
    int owner = getOwnerRankById(tgt_id);
    if (owner == -1) {
        fprintf(stderr, "[Rank %d] Could not determine owner rank for ID %d\n", sender_rank, tgt_id);
        return;
    }

    if (owner == sender_rank) {
        // --- Local delivery ---
        if (!id_to_index_map || tgt_id < 0 || tgt_id >= MAX_NODE_ID || id_to_index_map[tgt_id] == -1) {
            fprintf(stderr, "[Rank %d]️ Invalid ID in local sendSignalToRank: %d\n", sender_rank, tgt_id);
            return;
        }

        int tgt_idx = id_to_index_map[tgt_id];
        if (tgt_idx < 0 || tgt_idx >= num_brain_nodes) {
            fprintf(stderr, "[Rank %d]️ Invalid local neuron index: ID %d → index %d\n", sender_rank, tgt_id, tgt_idx);
            return;
        }

        Event ev = { .type = EVENT_TYPE_SIGNAL, .target = tgt_idx, .signal = signal };
        handle_event(&ev);

    } else {
        // --- Remote delivery ---
        PackedSignal packed = { .type = signal.type, .target = tgt_id, .value = signal.value };
        MPI_Send(&packed, 1, MPI_PackedSignal, owner, TAG_SIGNAL, MPI_COMM_WORLD);
    }
}

// -------------------------------
// Receive and dispatch incoming MPI signals
// -------------------------------
void receiveIncomingSignals(int current_rank) {
    MPI_Status status;
    int flag;

    while (1) {
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_SIGNAL, MPI_COMM_WORLD, &flag, &status);
        if (!flag)
            break;

        PackedSignal recv_signal;
        MPI_Recv(&recv_signal, 1, MPI_PackedSignal, status.MPI_SOURCE, TAG_SIGNAL, MPI_COMM_WORLD, &status);

        int tgt_id = recv_signal.target;

        // Validate incoming signal target
        if (!id_to_index_map || tgt_id < 0 || tgt_id >= MAX_NODE_ID || id_to_index_map[tgt_id] == -1) {
            fprintf(stderr, "[Rank %d]️ Invalid received ID in receiveIncomingSignals: %d\n", current_rank, tgt_id);
            continue;
        }

        int tgt_idx = id_to_index_map[tgt_id];
        if (tgt_idx < 0 || tgt_idx >= num_brain_nodes) {
            fprintf(stderr, "[Rank %d]️ Mapped invalid target index in receiveIncomingSignals: ID %d → index %d\n", current_rank, tgt_id, tgt_idx);
            continue;
        }

        struct SignalStruct signal = { .type = recv_signal.type, .value = recv_signal.value };
        Event ev = { .type = EVENT_TYPE_SIGNAL, .target = tgt_idx, .signal = signal };
        handle_event(&ev);
    }
}

// -------------------------------
// Central event dispatcher
// -------------------------------
void handle_event(Event *event) {
    if (!event)
        return;

    switch (event->type) {
        case EVENT_TYPE_SIGNAL:
            // Queue signal in inbox if there’s space
            if (event->target < 0 || event->target >= num_brain_nodes)
                return;
            if (brain_nodes[event->target].num_outstanding_signals < SIGNAL_INBOX_SIZE) {
                brain_nodes[event->target].signalInbox[brain_nodes[event->target].num_outstanding_signals++] = event->signal;
            } else {
                printf("[Rank %d] Signal dropped (inbox full): node %d\n", rank, brain_nodes[event->target].id);
            }
            break;

        case EVENT_TYPE_REPORT:
            generateReport(event->report_filename);
            break;

        case EVENT_TYPE_TERMINATE:
            freeMemory();
            MPI_Finalize();
            exit(0);
            break;

        default:
            fprintf(stderr, "[Rank %d] Unknown event type: %d\n", rank, event->type);
            break;
    }
}


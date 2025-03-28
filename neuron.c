// -------------------------------
// neuron.c
// -------------------------------

#include <stdio.h>
#include <stdlib.h>
#include "brain.h"
#include <mpi.h>

// -------------------------------
// Constants and Globals
// -------------------------------
#define SIGNAL_THRESHOLD 0.001

static const float NEURON_TYPE_SIGNAL_WEIGHTS[6] = {
    0.8, 1.2, 1.1, 2.6, 0.3, 1.8
};

extern int rank, size;
extern int *id_to_index_map;

// -------------------------------
// Update a neuron or nerve node
// -------------------------------
void updateNodes(int node_idx) {
    // --- Random firing for nerves ---
    if (brain_nodes[node_idx].num_edges > 0 &&
        brain_nodes[node_idx].node_type == NERVE) {

        int num_signals_to_fire = getRandomInteger(0, MAX_RANDOM_NERVE_SIGNALS_TO_FIRE);

        for (int i = 0; i < num_signals_to_fire; i++) {
            float signalValue = generateDecimalRandomNumber(MAX_SIGNAL_VALUE);
            int signalType = getRandomInteger(0, NUM_SIGNAL_TYPES);

            if (brain_nodes[node_idx].num_nerve_inputs)
                brain_nodes[node_idx].num_nerve_inputs[signalType]++;

            fireSignal(node_idx, signalValue, signalType);
        }
    }

    // --- Process all inboxed signals ---
    for (int i = 0; i < brain_nodes[node_idx].num_outstanding_signals; i++) {
        struct SignalStruct *sig = &brain_nodes[node_idx].signalInbox[i];

        if (brain_nodes[node_idx].node_type == NERVE &&
            brain_nodes[node_idx].num_nerve_inputs) {
            brain_nodes[node_idx].num_nerve_inputs[sig->type]++;
        }

        handleSignal(node_idx, sig->value, sig->type);
        brain_nodes[node_idx].signals_this_ns++;
    }

    brain_nodes[node_idx].total_signals_recieved += brain_nodes[node_idx].num_outstanding_signals;
    brain_nodes[node_idx].num_outstanding_signals = 0;
}

// -------------------------------
// Handle an individual signal
// -------------------------------
void handleSignal(int node_idx, float signal, int signal_type) {
    if (brain_nodes[node_idx].node_type == NERVE) {
        if (brain_nodes[node_idx].num_nerve_outputs)
            brain_nodes[node_idx].num_nerve_outputs[signal_type]++;

        fireSignal(node_idx, signal, signal_type);
    } else {
        // --- Apply neuron-type weight ---
        float weight = NEURON_TYPE_SIGNAL_WEIGHTS[
            neuronTypeToIndex(brain_nodes[node_idx].neuron_type)
        ];
        signal *= weight;

        // --- Overload logic ---
        int recent = brain_nodes[node_idx].signals_last_ns + brain_nodes[node_idx].signals_this_ns;
        if (recent > 500) {
            if (getRandomInteger(0, 2) == 1) signal /= 2.0;
            if (getRandomInteger(0, 3) == 1) return;
        }

        fireSignal(node_idx, signal, signal_type);
    }
}

// -------------------------------
// Fire a signal through edges
// -------------------------------
void fireSignal(int node_idx, float signal, int signal_type) {
    if (!brain_nodes[node_idx].edges || brain_nodes[node_idx].num_edges <= 0) return;
    if (signal_type < 0 || signal_type >= NUM_SIGNAL_TYPES) return;

    while (signal >= SIGNAL_THRESHOLD) {
        int edge_idx = brain_nodes[node_idx].edges[
            getRandomInteger(0, brain_nodes[node_idx].num_edges)
        ];

        if (edge_idx < 0 || edge_idx >= num_edges) return;

        // --- Determine target ID ---
        int tgt_id = (edges[edge_idx].from == brain_nodes[node_idx].id)
                     ? edges[edge_idx].to
                     : edges[edge_idx].from;

        // --- Limit signal chunk ---
        float chunk = signal;
        if (chunk > edges[edge_idx].max_value)
            chunk = edges[edge_idx].max_value;
        signal -= chunk;

        // --- Apply edge weight ---
        float type_weight = edges[edge_idx].messageTypeWeightings[signal_type];
        chunk *= type_weight;

        if (brain_nodes[node_idx].node_type == NERVE &&
            brain_nodes[node_idx].num_nerve_outputs) {
            brain_nodes[node_idx].num_nerve_outputs[signal_type]++;
        }

        struct SignalStruct s = { .type = signal_type, .value = chunk };
        sendSignalToRank(tgt_id, s, rank, size);
    }
}

// -------------------------------
// Generate simulation report
// -------------------------------
void generateReport(const char *filename) {
    FILE *out = fopen(filename, "w");
    if (!out) {
        fprintf(stderr, "[Rank %d] Failed to open report file: %s\n", rank, filename);
        return;
    }

    fprintf(out, "Simulation ran with %d neurons, %d nerves and %d total edges until %d ns\n\n",
            num_neurons, num_nerves, num_edges, elapsed_ns);

    // --- Report for nerves ---
    int nerve_count = 0;
    for (int i = 0; i < num_brain_nodes; i++) {
        if (brain_nodes[i].node_type == NERVE) {
            fprintf(out, "Nerve %d (ID: %d)\n", nerve_count++, brain_nodes[i].id);
            for (int j = 0; j < NUM_SIGNAL_TYPES; j++) {
                fprintf(out, "----> Type %d: %d inputs, %d outputs\n",
                        j, brain_nodes[i].num_nerve_inputs[j], brain_nodes[i].num_nerve_outputs[j]);
            }
        }
    }

    // --- Report for neurons ---
    fprintf(out, "\n");
    int neuron_count = 0;
    for (int i = 0; i < num_brain_nodes; i++) {
        if (brain_nodes[i].node_type == NEURON) {
            fprintf(out, "Neuron %d (ID: %d), total signals received: %d\n",
                    neuron_count++, brain_nodes[i].id, brain_nodes[i].total_signals_recieved);
        }
    }

    fclose(out);
}

// -------------------------------
// Clean up and free memory
// -------------------------------
void freeMemory() {
    for (int i = 0; i < num_edges; i++) {
        free(edges[i].messageTypeWeightings);
    }
    free(edges);

    for (int i = 0; i < num_brain_nodes; i++) {
        free(brain_nodes[i].signalInbox);
        free(brain_nodes[i].edges);
        free(brain_nodes[i].num_nerve_inputs);
        free(brain_nodes[i].num_nerve_outputs);
    }

    free(brain_nodes);

    // Cleanup shared global map
    if (id_to_index_map != NULL) {
        free(id_to_index_map);
        id_to_index_map = NULL;
    }
}


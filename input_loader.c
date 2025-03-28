// -------------------------------
// input_loader.c
// -------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "brain.h"

// -------------------------------
// External Globals
// -------------------------------
extern int rank;
extern int *id_to_index_map;

// -------------------------------
// Load brain graph from file
// -------------------------------
void loadBrainGraph(char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "[Rank %d] Could not open file %s\n", rank, filename);
        exit(EXIT_FAILURE);
    }

    const char whitespace[] = " \f\n\r\t\v";
    enum ReadMode currentMode = NONE;
    char buffer[MAX_LINE_LEN];
    int node_capacity = 128;
    int edge_capacity = 256;

    brain_nodes = calloc(node_capacity, sizeof(struct NeuronNerveStruct));
    edges = calloc(edge_capacity, sizeof(struct EdgeStruct));
    if (!brain_nodes || !edges) {
        fprintf(stderr, "[Rank %d]Memory allocation failed\n", rank);
        exit(EXIT_FAILURE);
    }

    // Initialize ID-to-index map
    for (int i = 0; i < MAX_NODE_ID; i++) {
        id_to_index_map[i] = -1;
    }

    int currentNeuronIdx = 0;
    int currentEdgeIdx = 0;

    // -------------------------------
    // File Parsing
    // -------------------------------
    while (fgets(buffer, MAX_LINE_LEN, file)) {
        if (buffer[0] == '%') continue;
        char *line_contents = buffer + strspn(buffer, whitespace);

        // --- Begin neuron or nerve node ---
        if (strncmp("<neuron>", line_contents, 8) == 0 || strncmp("<nerve>", line_contents, 7) == 0) {
            currentMode = NEURON_NERVE;

            if (currentNeuronIdx >= node_capacity) {
                node_capacity *= 2;
                brain_nodes = realloc(brain_nodes, node_capacity * sizeof(struct NeuronNerveStruct));
                if (!brain_nodes) {
                    fprintf(stderr, "[Rank %d] realloc failed for brain_nodes\n", rank);
                    exit(EXIT_FAILURE);
                }
            }

            struct NeuronNerveStruct *node = &brain_nodes[currentNeuronIdx];
            memset(node, 0, sizeof(struct NeuronNerveStruct));

            node->signalInbox = calloc(SIGNAL_INBOX_SIZE, sizeof(struct SignalStruct));
            node->num_nerve_inputs = calloc(NUM_SIGNAL_TYPES, sizeof(int));
            node->num_nerve_outputs = calloc(NUM_SIGNAL_TYPES, sizeof(int));
            if (!node->signalInbox || !node->num_nerve_inputs || !node->num_nerve_outputs) {
                fprintf(stderr, "[Rank %d] Allocation failed for node[%d]\n", rank, currentNeuronIdx);
                exit(EXIT_FAILURE);
            }

            node->node_type = (strncmp("<neuron>", line_contents, 8) == 0) ? NEURON : NERVE;

        // --- End neuron or nerve node ---
        } else if (strncmp("</neuron>", line_contents, 9) == 0 || strncmp("</nerve>", line_contents, 8) == 0) {
            currentMode = NONE;
            currentNeuronIdx++;

        // --- Begin edge definition ---
        } else if (strncmp("<edge>", line_contents, 6) == 0) {
            currentMode = EDGE;

            if (currentEdgeIdx >= edge_capacity) {
                edge_capacity *= 2;
                edges = realloc(edges, edge_capacity * sizeof(struct EdgeStruct));
                if (!edges) {
                    fprintf(stderr, "[Rank %d] realloc failed for edges\n", rank);
                    exit(EXIT_FAILURE);
                }
            }

            edges[currentEdgeIdx].messageTypeWeightings = calloc(NUM_SIGNAL_TYPES, sizeof(float));
            if (!edges[currentEdgeIdx].messageTypeWeightings) {
                fprintf(stderr, "[Rank %d] Failed to allocate weightings for edge %d\n", rank, currentEdgeIdx);
                exit(EXIT_FAILURE);
            }

        // --- End edge definition ---
        } else if (strncmp("</edge>", line_contents, 7) == 0 && currentMode == EDGE) {
            currentMode = NONE;
            currentEdgeIdx++;

        // --- Node properties ---
        } else if (strncmp("<id>", line_contents, 4) == 0 && currentMode == NEURON_NERVE) {
            int id = atoi(strstr(line_contents, ">") + 1);
            brain_nodes[currentNeuronIdx].id = id;

            if (id >= MAX_NODE_ID) {
                fprintf(stderr, "[Rank %d] Node ID %d exceeds max supported (%d)\n", rank, id, MAX_NODE_ID);
                exit(EXIT_FAILURE);
            }

            id_to_index_map[id] = currentNeuronIdx;

        } else if (strncmp("<x>", line_contents, 3) == 0) {
            brain_nodes[currentNeuronIdx].x = atof(strstr(line_contents, ">") + 1);

        } else if (strncmp("<y>", line_contents, 3) == 0) {
            brain_nodes[currentNeuronIdx].y = atof(strstr(line_contents, ">") + 1);

        } else if (strncmp("<z>", line_contents, 3) == 0) {
            brain_nodes[currentNeuronIdx].z = atof(strstr(line_contents, ">") + 1);

        } else if (strncmp("<type>", line_contents, 6) == 0) {
            if (brain_nodes[currentNeuronIdx].node_type == NEURON) {
                char *type = strstr(line_contents, ">") + 1;
                if (strncmp(type, "sensory", 7) == 0) brain_nodes[currentNeuronIdx].neuron_type = SENSORY;
                else if (strncmp(type, "motor", 5) == 0) brain_nodes[currentNeuronIdx].neuron_type = MOTOR;
                else if (strncmp(type, "unipolar", 8) == 0) brain_nodes[currentNeuronIdx].neuron_type = UNIPOLAR;
                else if (strncmp(type, "pseudounipolar", 14) == 0) brain_nodes[currentNeuronIdx].neuron_type = PSEUDOUNIPOLAR;
                else if (strncmp(type, "bipolar", 7) == 0) brain_nodes[currentNeuronIdx].neuron_type = BIPOLAR;
                else if (strncmp(type, "multipolar", 10) == 0) brain_nodes[currentNeuronIdx].neuron_type = MULTIPOLAR;
                else {
                    fprintf(stderr, "[Rank %d] Unknown neuron type: %s\n", rank, type);
                    exit(EXIT_FAILURE);
                }
            }

        // --- Edge properties ---
        } else if (strncmp("<from>", line_contents, 6) == 0 && currentMode == EDGE) {
            edges[currentEdgeIdx].from = atoi(strstr(line_contents, ">") + 1);

        } else if (strncmp("<to>", line_contents, 4) == 0 && currentMode == EDGE) {
            edges[currentEdgeIdx].to = atoi(strstr(line_contents, ">") + 1);

        } else if (strncmp("<direction>", line_contents, 11) == 0 && currentMode == EDGE) {
            char *dir = strstr(line_contents, ">") + 1;
            if (strstr(dir, "bidirectional")) edges[currentEdgeIdx].direction = BIDIRECTIONAL;
            else edges[currentEdgeIdx].direction = UNIDIRECTIONAL;

        } else if (strncmp("<max_value>", line_contents, 11) == 0 && currentMode == EDGE) {
            edges[currentEdgeIdx].max_value = atof(strstr(line_contents, ">") + 1);

        } else if (strncmp("<weighting_", line_contents, 11) == 0 && currentMode == EDGE) {
            int idx = atoi(&line_contents[11]);
            if (idx >= 0 && idx < NUM_SIGNAL_TYPES) {
                edges[currentEdgeIdx].messageTypeWeightings[idx] = atof(strstr(line_contents, ">") + 1);
            } else {
                fprintf(stderr, "[Rank %d] Invalid signal weighting index: %d\n", rank, idx);
            }
        }
    }

    fclose(file);

    // -------------------------------
    // Post-processing and Summary
    // -------------------------------
    num_brain_nodes = currentNeuronIdx;
    num_edges = currentEdgeIdx;
    num_neurons = 0;
    num_nerves = 0;

    for (int i = 0; i < num_brain_nodes; i++) {
        if (brain_nodes[i].node_type == NEURON) num_neurons++;
        else if (brain_nodes[i].node_type == NERVE) num_nerves++;
    }

    if (rank == 0) {
        printf("[Rank 0] Loaded %d neurons, %d nerves, %d total nodes, %d edges\n",
               num_neurons, num_nerves, num_brain_nodes, num_edges);
    }
}

// -------------------------------
// Link nodes to their edges
// -------------------------------
void linkNodesToEdges() {
    for (int i = 0; i < num_brain_nodes; i++) {
        int id = brain_nodes[i].id;
        int edge_count = 0;

        for (int j = 0; j < num_edges; j++) {
            if (edges[j].from == id || edges[j].to == id) {
                edge_count++;
            }
        }

        brain_nodes[i].num_edges = edge_count;
        brain_nodes[i].edges = malloc(edge_count * sizeof(int));
        if (!brain_nodes[i].edges) {
            fprintf(stderr, "[Rank %d] Failed to allocate edges for node ID %d\n", rank, id);
            exit(EXIT_FAILURE);
        }

        int count = 0;
        for (int j = 0; j < num_edges; j++) {
            if (edges[j].from == id || edges[j].to == id) {
                brain_nodes[i].edges[count++] = j;
            }
        }

        if (count != edge_count) {
            fprintf(stderr, "[Rank %d]️ Edge mismatch for node ID %d: expected %d, got %d\n", rank, id, edge_count, count);
        }
    }
}

// -------------------------------
// Map neuron type enum to index
// -------------------------------
int neuronTypeToIndex(enum NeuronType type) {
    switch (type) {
        case SENSORY: return 0;
        case MOTOR: return 1;
        case UNIPOLAR: return 2;
        case PSEUDOUNIPOLAR: return 3;
        case BIPOLAR: return 4;
        case MULTIPOLAR: return 5;
        default:
            fprintf(stderr, "[Rank %d] Invalid NeuronType enum value: %d\n", rank, type);
            return -1;
    }
}


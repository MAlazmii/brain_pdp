// -------------------------------
// input_loader.c
// -------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include "brain.h"

// -------------------------------
// External Globals
// -------------------------------
extern int rank;
extern int *id_to_index_map;

static void inputError(const char *message) {
    fprintf(stderr, "[Rank %d] Invalid graph: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

static char *tagValue(char *line) {
    char *start = strchr(line, '>');
    char *end = start ? strchr(start + 1, '<') : NULL;
    if (!start || !end) inputError("malformed field");
    *end = '\0';
    return start + 1;
}

static int strictInt(char *line) {
    char *value = tagValue(line), *end;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || end == value || *end || parsed < INT_MIN || parsed > INT_MAX)
        inputError("integer field is not an integer");
    return (int)parsed;
}

static float strictFloat(char *line) {
    char *value = tagValue(line), *end;
    errno = 0;
    float parsed = strtof(value, &end);
    if (errno || end == value || *end || !isfinite(parsed))
        inputError("numeric field must be finite");
    return parsed;
}

// -------------------------------
// Load brain graph from file
// -------------------------------
void loadBrainGraph(char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "[Rank %d] Could not open file %s\n", rank, filename);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Initialize ID-to-index map
    for (int i = 0; i < MAX_NODE_ID; i++) {
        id_to_index_map[i] = -1;
    }

    int currentNeuronIdx = 0;
    int currentEdgeIdx = 0;
    unsigned node_fields = 0;
    unsigned edge_fields = 0;
    enum NodeType open_node_type = NEURON;
    const unsigned required_node_fields = 1u | 2u | 4u | 8u;
    const unsigned required_edge_fields = (1u << (4 + NUM_SIGNAL_TYPES)) - 1u;

    // -------------------------------
    // File Parsing
    // -------------------------------
    while (fgets(buffer, MAX_LINE_LEN, file)) {
        if (buffer[0] == '%') continue;
        char *line_contents = buffer + strspn(buffer, whitespace);

        // --- Begin neuron or nerve node ---
        if (strncmp("<neuron>", line_contents, 8) == 0 || strncmp("<nerve>", line_contents, 7) == 0) {
            if (currentMode != NONE) inputError("nested node or edge");
            currentMode = NEURON_NERVE;
            node_fields = 0;

            if (currentNeuronIdx >= node_capacity) {
                node_capacity *= 2;
                brain_nodes = realloc(brain_nodes, node_capacity * sizeof(struct NeuronNerveStruct));
                if (!brain_nodes) {
                    fprintf(stderr, "[Rank %d] realloc failed for brain_nodes\n", rank);
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
            }

            struct NeuronNerveStruct *node = &brain_nodes[currentNeuronIdx];
            memset(node, 0, sizeof(struct NeuronNerveStruct));

            node->signalInbox = calloc(SIGNAL_INBOX_SIZE, sizeof(struct SignalStruct));
            node->num_nerve_inputs = calloc(NUM_SIGNAL_TYPES, sizeof(int));
            node->num_nerve_outputs = calloc(NUM_SIGNAL_TYPES, sizeof(int));
            if (!node->signalInbox || !node->num_nerve_inputs || !node->num_nerve_outputs) {
                fprintf(stderr, "[Rank %d] Allocation failed for node[%d]\n", rank, currentNeuronIdx);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            node->node_type = (strncmp("<neuron>", line_contents, 8) == 0) ? NEURON : NERVE;
            open_node_type = node->node_type;

        // --- End neuron or nerve node ---
        } else if (strncmp("</neuron>", line_contents, 9) == 0 || strncmp("</nerve>", line_contents, 8) == 0) {
            enum NodeType closing_type = strncmp("</neuron>", line_contents, 9) == 0 ? NEURON : NERVE;
            if (currentMode != NEURON_NERVE || closing_type != open_node_type)
                inputError("mismatched node terminator");
            if (!(node_fields & 1u) ||
                (open_node_type == NEURON &&
                 ((node_fields & required_node_fields) != required_node_fields || !(node_fields & 16u))))
                inputError("node is missing a required field");
            currentMode = NONE;
            currentNeuronIdx++;

        // --- Begin edge definition ---
        } else if (strncmp("<edge>", line_contents, 6) == 0) {
            if (currentMode != NONE) inputError("nested node or edge");
            currentMode = EDGE;
            edge_fields = 0;

            if (currentEdgeIdx >= edge_capacity) {
                edge_capacity *= 2;
                edges = realloc(edges, edge_capacity * sizeof(struct EdgeStruct));
                if (!edges) {
                    fprintf(stderr, "[Rank %d] realloc failed for edges\n", rank);
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
            }

            memset(&edges[currentEdgeIdx], 0, sizeof(struct EdgeStruct));
            edges[currentEdgeIdx].messageTypeWeightings = calloc(NUM_SIGNAL_TYPES, sizeof(float));
            if (!edges[currentEdgeIdx].messageTypeWeightings) {
                fprintf(stderr, "[Rank %d] Failed to allocate weightings for edge %d\n", rank, currentEdgeIdx);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

        // --- End edge definition ---
        } else if (strncmp("</edge>", line_contents, 7) == 0) {
            if (currentMode != EDGE) inputError("edge terminator outside edge");
            if (edge_fields != required_edge_fields)
                inputError("edge is missing a required field");
            if (!isfinite(edges[currentEdgeIdx].max_value) || !(edges[currentEdgeIdx].max_value > 0)) {
                fprintf(stderr, "Edge capacity must be positive\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            currentMode = NONE;
            for (int type = 0; type < NUM_SIGNAL_TYPES; type++) {
                if (!isfinite(edges[currentEdgeIdx].messageTypeWeightings[type])) {
                    fprintf(stderr, "Edge weights must be finite\n");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
            }
            currentEdgeIdx++;

        // --- Node properties ---
        } else if (strncmp("<id>", line_contents, 4) == 0 && currentMode == NEURON_NERVE) {
            if (node_fields & 1u) inputError("duplicate node ID");
            int id = strictInt(line_contents);
            brain_nodes[currentNeuronIdx].id = id;

            if (id < 0 || id >= MAX_NODE_ID || id_to_index_map[id] != -1) {
                fprintf(stderr, "[Rank %d] Node ID %d exceeds max supported (%d)\n", rank, id, MAX_NODE_ID);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            id_to_index_map[id] = currentNeuronIdx;
            node_fields |= 1u;

        } else if (strncmp("<x>", line_contents, 3) == 0) {
            if (currentMode != NEURON_NERVE) inputError("node field outside node");
            if (node_fields & 2u) inputError("duplicate node field");
            brain_nodes[currentNeuronIdx].x = strictFloat(line_contents);
            node_fields |= 2u;

        } else if (strncmp("<y>", line_contents, 3) == 0) {
            if (currentMode != NEURON_NERVE) inputError("node field outside node");
            if (node_fields & 4u) inputError("duplicate node field");
            brain_nodes[currentNeuronIdx].y = strictFloat(line_contents);
            node_fields |= 4u;

        } else if (strncmp("<z>", line_contents, 3) == 0) {
            if (currentMode != NEURON_NERVE) inputError("node field outside node");
            if (node_fields & 8u) inputError("duplicate node field");
            brain_nodes[currentNeuronIdx].z = strictFloat(line_contents);
            node_fields |= 8u;

        } else if (strncmp("<type>", line_contents, 6) == 0) {
            if (currentMode != NEURON_NERVE || open_node_type != NEURON) inputError("type outside neuron");
            if (node_fields & 16u) inputError("duplicate node field");
            char *type = tagValue(line_contents);
            if (strcmp(type, "sensory") == 0) brain_nodes[currentNeuronIdx].neuron_type = SENSORY;
            else if (strcmp(type, "motor") == 0) brain_nodes[currentNeuronIdx].neuron_type = MOTOR;
            else if (strcmp(type, "unipolar") == 0) brain_nodes[currentNeuronIdx].neuron_type = UNIPOLAR;
            else if (strcmp(type, "pseudounipolar") == 0) brain_nodes[currentNeuronIdx].neuron_type = PSEUDOUNIPOLAR;
            else if (strcmp(type, "bipolar") == 0) brain_nodes[currentNeuronIdx].neuron_type = BIPOLAR;
            else if (strcmp(type, "multipolar") == 0) brain_nodes[currentNeuronIdx].neuron_type = MULTIPOLAR;
            else inputError("unknown neuron type");
            node_fields |= 16u;

        // --- Edge properties ---
        } else if (strncmp("<from>", line_contents, 6) == 0 && currentMode == EDGE) {
            if (edge_fields & 1u) inputError("duplicate edge field");
            edges[currentEdgeIdx].from = strictInt(line_contents);
            edge_fields |= 1u;

        } else if (strncmp("<to>", line_contents, 4) == 0 && currentMode == EDGE) {
            if (edge_fields & 2u) inputError("duplicate edge field");
            edges[currentEdgeIdx].to = strictInt(line_contents);
            edge_fields |= 2u;

        } else if (strncmp("<direction>", line_contents, 11) == 0 && currentMode == EDGE) {
            if (edge_fields & 4u) inputError("duplicate edge field");
            char *dir = tagValue(line_contents);
            if (strcmp(dir, "bidirectional") == 0) edges[currentEdgeIdx].direction = BIDIRECTIONAL;
            else if (strcmp(dir, "unidirectional") == 0) edges[currentEdgeIdx].direction = UNIDIRECTIONAL;
            else inputError("unknown edge direction");
            edge_fields |= 4u;

        } else if (strncmp("<max_value>", line_contents, 11) == 0 && currentMode == EDGE) {
            if (edge_fields & 8u) inputError("duplicate edge field");
            edges[currentEdgeIdx].max_value = strictFloat(line_contents);
            edge_fields |= 8u;

        } else if (strncmp("<weighting_", line_contents, 11) == 0 && currentMode == EDGE) {
            char *index_end;
            errno = 0;
            long idx_value = strtol(&line_contents[11], &index_end, 10);
            int idx = (int)idx_value;
            if (errno || index_end == &line_contents[11] || *index_end != '>' || idx_value < 0 || idx_value >= NUM_SIGNAL_TYPES)
                inputError("invalid signal weighting index");
            if (idx >= 0 && idx < NUM_SIGNAL_TYPES) {
                if (edge_fields & (1u << (4 + idx))) inputError("duplicate edge field");
                edges[currentEdgeIdx].messageTypeWeightings[idx] = strictFloat(line_contents);
                edge_fields |= 1u << (4 + idx);
            } else {
                inputError("invalid signal weighting index");
            }
        }
    }

    fclose(file);

    if (currentMode != NONE) inputError("unterminated node or edge");
    for (int i = 0; i < currentEdgeIdx; i++) {
        if (edges[i].from < 0 || edges[i].from >= MAX_NODE_ID || id_to_index_map[edges[i].from] < 0 ||
            edges[i].to < 0 || edges[i].to >= MAX_NODE_ID || id_to_index_map[edges[i].to] < 0)
            inputError("edge endpoint does not name a node");
    }

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
            if (edges[j].from == id || (edges[j].direction == BIDIRECTIONAL && edges[j].to == id)) {
                edge_count++;
            }
        }

        brain_nodes[i].num_edges = edge_count;
        brain_nodes[i].edges = malloc(edge_count * sizeof(int));
        if (edge_count && !brain_nodes[i].edges) {
            fprintf(stderr, "[Rank %d] Failed to allocate edges for node ID %d\n", rank, id);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        int count = 0;
        for (int j = 0; j < num_edges; j++) {
            if (edges[j].from == id || (edges[j].direction == BIDIRECTIONAL && edges[j].to == id)) {
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

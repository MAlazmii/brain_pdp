// ----------------------------------------
// Includes and Global Declarations
// ----------------------------------------
#include "brain.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Global brain data
struct NeuronNerveStruct *brain_nodes = NULL;
struct EdgeStruct *edges = NULL;
int num_neurons = 0, num_nerves = 0, num_edges = 0, num_brain_nodes = 0, elapsed_ns = 0;

// Global ID mapping pointers
int *id_to_index_map = NULL;
int *id_to_index = NULL;

// MPI globals
int rank, size;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    create_mpi_signal_type();

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <brain_graph_file> <num_nanoseconds>\n", argv[0]);
        MPI_Type_free(&MPI_PackedSignal);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    srand((unsigned)(time(NULL) + rank));

    id_to_index_map = malloc(sizeof(int) * MAX_NODE_ID);
    if (!id_to_index_map) {
        fprintf(stderr, "[Rank %d] Failed to allocate id_to_index_map\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        loadBrainGraph(argv[1]);
        linkNodesToEdges();
    }

    MPI_Bcast(id_to_index_map, MAX_NODE_ID, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_brain_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_neurons, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_nerves, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    id_to_index = malloc(sizeof(int) * MAX_NODE_ID);
    if (!id_to_index) {
        fprintf(stderr, "[Rank %d] Failed to allocate id_to_index\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memcpy(id_to_index, id_to_index_map, sizeof(int) * MAX_NODE_ID);
    for (int i = 0; i < MAX_NODE_ID; i++)
        id_to_index_map[i] = -1;

    if (rank != 0) {
        brain_nodes = calloc(num_brain_nodes, sizeof(struct NeuronNerveStruct));
        edges = calloc(1, sizeof(struct EdgeStruct)); // Dummy
        if (!brain_nodes || !edges) {
            fprintf(stderr, "[Rank %d] Allocation failure for brain_nodes or edges\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    for (int i = 0; i < num_brain_nodes; i++) {
        MPI_Bcast(&brain_nodes[i].id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&brain_nodes[i].node_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&brain_nodes[i].neuron_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&brain_nodes[i].x, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&brain_nodes[i].y, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&brain_nodes[i].z, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&brain_nodes[i].signals_last_ns, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&brain_nodes[i].signals_this_ns, 1, MPI_INT, 0, MPI_COMM_WORLD);

        brain_nodes[i].signalInbox = calloc(SIGNAL_INBOX_SIZE, sizeof(struct SignalStruct));
        brain_nodes[i].num_nerve_inputs = calloc(NUM_SIGNAL_TYPES, sizeof(int));
        brain_nodes[i].num_nerve_outputs = calloc(NUM_SIGNAL_TYPES, sizeof(int));
        brain_nodes[i].num_outstanding_signals = 0;

        if (!brain_nodes[i].signalInbox || !brain_nodes[i].num_nerve_inputs || !brain_nodes[i].num_nerve_outputs) {
            fprintf(stderr, "[Rank %d] Allocation failure in brain_nodes[%d]\n", rank, i);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Bcast(brain_nodes[i].num_nerve_inputs, NUM_SIGNAL_TYPES, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(brain_nodes[i].num_nerve_outputs, NUM_SIGNAL_TYPES, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0 && (!brain_nodes || !edges || num_brain_nodes == 0 || num_edges == 0)) {
        fprintf(stderr, "[Rank 0] Invalid brain graph structure\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("[Rank %d] Loaded: neurons=%d nerves=%d nodes=%d edges=%d\n",
           rank, num_neurons, num_nerves, num_brain_nodes, num_edges);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    int neurons_per_rank = num_brain_nodes / size;
    int remainder = num_brain_nodes % size;
    int start_idx = rank * neurons_per_rank + (rank < remainder ? rank : remainder);
    int local_count = neurons_per_rank + (rank < remainder ? 1 : 0);
    int end_idx = start_idx + local_count;

    printf("[Rank %d] Handling brain nodes from %d to %d (count = %d)\n",
           rank, start_idx, end_idx - 1, local_count);
    fflush(stdout);

    if (rank == 0) {
        printf("\n--- Parallel Brain Simulation ---\n");
        printf("MPI Ranks: %d | Brain Nodes: %d | Simulating %s ns\n", size, num_brain_nodes, argv[2]);
    }

    for (int i = start_idx; i < end_idx; i++)
        id_to_index_map[brain_nodes[i].id] = i;

    int num_ns_to_simulate = atoi(argv[2]);
    int total_iterations = 0, current_ns_iterations = 0;
    int max_iteration_per_ns = -1, min_iteration_per_ns = -1;
    time_t seconds = 0, start_seconds = getCurrentSeconds();

    // ✅ START timing here
    double start_time = MPI_Wtime();

    while (elapsed_ns < num_ns_to_simulate) {
        time_t current_seconds = getCurrentSeconds();
        if (current_seconds != seconds) {
            seconds = current_seconds;

            if ((seconds - start_seconds) % MIN_LENGTH_NS == 0) {
                if (elapsed_ns == 0) {
                    max_iteration_per_ns = min_iteration_per_ns = current_ns_iterations;
                } else {
                    if (current_ns_iterations > max_iteration_per_ns)
                        max_iteration_per_ns = current_ns_iterations;
                    if (current_ns_iterations < min_iteration_per_ns)
                        min_iteration_per_ns = current_ns_iterations;
                }

                elapsed_ns++;
                current_ns_iterations = 0;

                for (int i = start_idx; i < end_idx; i++) {
                    brain_nodes[i].signals_last_ns = brain_nodes[i].signals_this_ns;
                    brain_nodes[i].signals_this_ns = 0;
                }
            }
        }

        receiveIncomingSignals(rank);

        for (int i = 0; i < num_brain_nodes; i++) {
            if (brain_nodes[i].node_type == NERVE)
                updateNodes(i);
        }

        for (int i = start_idx; i < end_idx; i++) {
            if (brain_nodes[i].node_type == NEURON)
                updateNodes(i);
        }

        receiveIncomingSignals(rank);
        MPI_Barrier(MPI_COMM_WORLD);
        current_ns_iterations++;
        total_iterations++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    receiveIncomingSignals(rank);
    usleep(50000);
    MPI_Barrier(MPI_COMM_WORLD);

    int *local_counts = malloc(local_count * sizeof(int));
    for (int i = 0; i < local_count; i++)
        local_counts[i] = brain_nodes[start_idx + i].total_signals_recieved;

    int *global_counts = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        global_counts = malloc(num_brain_nodes * sizeof(int));
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int offset = 0;
        for (int r = 0; r < size; r++) {
            int count = num_brain_nodes / size + (r < remainder ? 1 : 0);
            recvcounts[r] = count;
            displs[r] = offset;
            offset += count;
        }
    }

    MPI_Gatherv(local_counts, local_count, MPI_INT,
                global_counts, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < num_brain_nodes; i++) {
            brain_nodes[i].total_signals_recieved = global_counts[i];
        }

        generateReport(OUTPUT_REPORT_FILENAME);
        printf("\n Simulation complete.\n");
        printf(" Report saved to: %s\n", OUTPUT_REPORT_FILENAME);
        printf(" Iterations: %d (max %d/ns, min %d/ns)\n",
               total_iterations, max_iteration_per_ns, min_iteration_per_ns);

        // ✅ Print total simulation time
        double end_time = MPI_Wtime();
        printf(" Total simulation time: %.6f seconds\n", end_time - start_time);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        freeMemory();
    } else {
        for (int i = 0; i < num_brain_nodes; i++) {
            free(brain_nodes[i].signalInbox);
            free(brain_nodes[i].num_nerve_inputs);
            free(brain_nodes[i].num_nerve_outputs);
        }
        free(brain_nodes);
        free(edges);
    }

    free(id_to_index_map);
    free(id_to_index);
    free(local_counts);
    if (rank == 0) {
        free(global_counts);
        free(recvcounts);
        free(displs);
    }

    MPI_Type_free(&MPI_PackedSignal);
    MPI_Finalize();
    return EXIT_SUCCESS;
}


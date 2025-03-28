#ifndef BRAIN_H
#define BRAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

// -------------------------------
// Simulation Constants
// -------------------------------
#define MAX_LINE_LEN 100
#define NUM_SIGNAL_TYPES 10
#define MIN_LENGTH_NS 2
#define SIGNAL_INBOX_SIZE 16384
#define MAX_NODE_ID 2048
#define MAX_RANDOM_NERVE_SIGNALS_TO_FIRE 20
#define MAX_SIGNAL_VALUE 1000
#define OUTPUT_REPORT_FILENAME "summary_report"

// -------------------------------
// Enumerations
// -------------------------------
enum ReadMode       { NONE, NEURON_NERVE, EDGE };
enum NeuronType     { SENSORY, MOTOR, UNIPOLAR, PSEUDOUNIPOLAR, BIPOLAR, MULTIPOLAR };
enum NodeType       { NEURON, NERVE };
enum EdgeDirection  { BIDIRECTIONAL, UNIDIRECTIONAL };

// -------------------------------
// Signal Structure
// -------------------------------
struct SignalStruct {
    int type;
    float value;
};

// -------------------------------
// Edge (Connection)
// -------------------------------
struct EdgeStruct {
    int from, to;
    enum EdgeDirection direction;
    float *messageTypeWeightings;
    float max_value;
};

// -------------------------------
// Node (Neuron/Nerve)
// -------------------------------
struct NeuronNerveStruct {
    int id;
    int num_edges;
    int num_outstanding_signals;
    int total_signals_recieved;
    int *num_nerve_outputs;
    int *num_nerve_inputs;
    int signals_this_ns;
    int signals_last_ns;
    float x, y, z;
    enum NodeType node_type;
    enum NeuronType neuron_type;
    int *edges;
    struct SignalStruct *signalInbox;
    int is_active;
};

// -------------------------------
// Event Types and Structure
// -------------------------------
typedef enum {
    EVENT_TYPE_SIGNAL = 0,
    EVENT_TYPE_REPORT = 1,
    EVENT_TYPE_TERMINATE = 2
} EventType;

typedef struct {
    EventType type;
    int target;
    struct SignalStruct signal;
    char report_filename[256];
} Event;

// -------------------------------
// Global Brain Data (extern)
// -------------------------------
extern struct NeuronNerveStruct *brain_nodes;
extern struct EdgeStruct *edges;
extern int num_neurons, num_nerves, num_edges, num_brain_nodes, elapsed_ns;

// -------------------------------
// Global ID-to-Index Maps (extern)
// -------------------------------
extern int *id_to_index_map;
extern int *id_to_index;

// -------------------------------
// MPI Signal Type
// -------------------------------
extern MPI_Datatype MPI_PackedSignal;
void create_mpi_signal_type();

// -------------------------------
// Loader Functions
// -------------------------------
void loadBrainGraph(char *filename);
void linkNodesToEdges();
int getNumberOfEdgesForNode(int node_id);
int getNodeIndexById(int id);
int neuronTypeToIndex(enum NeuronType type);

// -------------------------------
// Simulation Logic
// -------------------------------
void updateNodes(int node_idx);
void handleSignal(int node_idx, float signal, int signal_type);
void fireSignal(int node_idx, float signal, int signal_type);
void generateReport(const char *filename);
void freeMemory();

// -------------------------------
// Utility Functions
// -------------------------------
int getRandomInteger(int min, int max);
float generateDecimalRandomNumber(int max_val);
time_t getCurrentSeconds();
void initialize_random();

// -------------------------------
// Event Handling
// -------------------------------
void sendSignalToRank(int tgt_idx, struct SignalStruct signal, int rank, int size);
void receiveIncomingSignals(int rank);
void handle_event(Event *event);

// -------------------------------
// Rank Helpers
// -------------------------------
int getOwnerRank(int node_idx, int total_nodes, int world_size);
int getOwnerRankById(int id);

#endif // BRAIN_H


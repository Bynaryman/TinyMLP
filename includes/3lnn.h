/**
 * @file 3lnn.h
 * @brief Neural network functionality for a 3-layer (INPUT, HIDDEN, OUTPUT) feed-forward, back-prop NN
 * @author Matt Lind
 * @date August 2015
 */

#include "ieee754.hpp"
#include <memory>

// <Mantissa, exp>
typedef IEEE754<2, 1 > float4;
typedef IEEE754<3, 4 > float8;
typedef IEEE754<10, 5 > float16;
typedef IEEE754<23, 8 > float32;
// typedef IEEE754<52, 11 > float64;

typedef float  test_type;

typedef struct Network Network;
typedef struct Layer Layer;
typedef struct Node Node;
typedef struct Vector Vector;

typedef enum LayerType {INPUT, HIDDEN, OUTPUT} LayerType;
typedef enum ActFctType {SIGMOID, TANH, RELU} ActFctType;




/**
 * @brief Dynamic data structure containing defined number of values
 */

struct Vector{
    int size;
    double vals[1000];
    test_type vals_diff_arith[1000];
};




/**
 * @brief Dynamic data structure modeling a neuron with a variable number of connections/weights
 */

struct Node{
    double bias;
    double output;
    test_type bias_diff_arith;
    test_type output_diff_arith;
    int wcount;
    double weights[1000];
    test_type weights_diff_arith[1000];
};




/**
 * @brief Dynamic data structure holding a definable number of nodes that form a layer
 */

struct Layer{
    int ncount;
    Node nodes[];
};


/**
 * @brief Dynamic data structure holding the whole network
 */

struct Network{
    int inpNodeSize;
    int inpLayerSize;
    int hidNodeSize;
    int hidLayerSize;
    int outNodeSize;
    int outLayerSize;
    double learningRate;         ///< Factor by which connection weight changes are applied
    ActFctType hidLayerActType;
    ActFctType outLayerActType;
    Layer layers[];
};




/**
 * @brief Creates a dynamically-sized, 3-layer (INTPUT, HIDDEN, OUTPUT) neural network
 * @param inpCount Number of nodes in the INPUT layer
 * @param hidCount Number of nodes in the HIDDEN layer
 * @param outCount Number of nodes in the OUTPUT layer
 */

Network *createNetwork(int inpCount, int hidCount, int outCount);




/**
 * @brief Feeds some Vector data into the INPUT layer of the NN
 * @param nn A pointer to the NN
 * @param v A pointer to a vector
 */

void feedInput(Network *nn, Vector *v);




/**
 * @brief Feeds input layer values forward to hidden to output layer (calculation and activation fct)
 * @param nn A pointer to the NN
 * @param bool train true if training phase 
 */

void feedForwardNetwork(Network *nn, bool train);




/**
 * @brief Back propagates network error from output layer to hidden layer
 * @param nn A pointer to the NN
 * @param targetClassification Correct classification (=label) of the input stream
 */

void backPropagateNetwork(Network *nn, int targetClassification);




/**
 * @brief Returns the network's classification using the ID of teh node with the hightest output
 * @param nn A pointer to the NN
 */

int getNetworkClassification(Network *nn, int * classifciation_diff_arith);


/**
 * @brief write to a file the weights of each node of the layer
 * @param nn A pointer to the NN
 */
void writeWeightsToFile(Network *nn, char path[]);

/**
 * @brief write to a file the weights of each node of the layer
 * @param nn A pointer to the NN
 */
void writeWeightsToFileForJulia(Network *nn, char path[]);


void displayNetworkWeightsForDebugging(Network *nn);

/**
 * @brief once the neural network has been trained in fp64
 * there is the prossibility to promote every weights to a software emulated ieee-like type
 */
void promoteWeights(Network *nn);

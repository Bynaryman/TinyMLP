#ifndef __HEADER_TEMPLATE_3LNN__
#define __HEADER_TEMPLATE_3LNN__ 

#include <stdint.h>
#include <iostream>
#include <vector>

typedef enum LayerType {INPUT, HIDDEN, OUTPUT} LayerType;
typedef enum ActFctType {SIGMOID, TANH, RELU} ActFctType;

/**
 * @brief Dynamic Template data structure modeling a neuron with a variable number of connections/weights
 */
template <typename T>
struct Neuron {
    T bias;
    T output;
    std::vector<T> weights;
};

/**
 * @brief Dynamic Template data structure holding a definable number of neurons to form a layer 
 */
// template <typename T>
// struct Layer {
//     std::vector<Neuron<T>> neurons;
// };

template <typename T>
using Layer = std::vector<Neuron<T>>;

/**
 * @brief Dynamic Template data structure holding the whole network
 */
// template <typename T>
// struct Network{
//     uint32_t inpNodeSize;
//     uint32_t inpLayerSize;
//     uint32_t hidNodeSize;
//     uint32_t hidLayerSize;
//     uint32_t outNodeSize;
//     uint32_t outLayerSize;
//     double learningRate;         ///< Factor by which connection weight changes are applied
//     ActFctType hidLayerActType;
//     ActFctType outLayerActType;
//     Layer<T> layers[];
// };

template <class T=double>
class Network {
    
private:
    
    // size of layers
    uint32_t m_in_count;   
    uint32_t m_hidden_count;   
    uint32_t m_out_count;
    
    // layers
    Layer<T> m_hidden_layer;
    Layer<T> m_output_layer;
    ActFctType m_hidden_layeract_fct;

public:
    Network(uint32_t in_count, uint32_t hidden_count, uint32_t out_count);
};

template <class T >
Network<T>::Network(uint32_t in_count, uint32_t hidden_count, uint32_t out_count):
    m_in_count(in_count),
    m_hidden_count(hidden_count),
    m_out_count(out_count) 
{
    m_hidden_layer = Layer<T>();
    for (unsigned int i = 0 ; i < in_count ; i++){
    }
}
#endif

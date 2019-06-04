#ifndef __HEADER_TEMPLATE_LAYER__
#define __HEADER_TEMPLATE_LAYER__ 

#include <stdint.h>
#include <vector>
#include "template_neuron.hpp"

typedef enum LayerType {INPUT, HIDDEN, OUTPUT} LayerType;
typedef enum ActFctType {SIGMOID, TANH, RELU} ActFctType;

template <class T=double>
class Layer {

private:

    // size of layer
    uint32_t m_nb_neurons;

    // array of neurons
    std::vector<Neuron<T>> neurons;

    // layer attributes
    ActFctType m_act_fct_type;
    LayerType m_layer_type;

public:
   Layer(uint32_t nb_neurons, ActFctType act_fct_type, LayerType layer_type);
};

template <class T >
Layer<T>::Layer(uint32_t nb_neurons, ActFctType act_fct_type, LayerType layer_type):
    m_nb_neurons(nb_neurons),
    m_act_fct_type(act_fct_type),
    m_layer_type(layer_type) 
{
    for (unsigned int i = 0 ; i < nb_neurons ; i++) {
        n = Neuron<T>()
        neurons.push(n);
    }
}
#endif

#include "template_3lnn.hpp"
#include "ieee754.hpp"
#include <iostream>

typedef IEEE754<3, 4 > float8;
typedef IEEE754<10, 5 > float16;
typedef IEEE754<23, 8 > float32;
typedef IEEE754<52, 11 > float64;

int main (int argc, char** argv) {

    // Create a Network MLP of 784 in, 20 hidden, 10 out
    Network<float8> n(784, 20, 10);
    std::cout << "" << std::endl;
    return 0;
}

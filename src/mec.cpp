#include <torch/extension.h>
#include <stdlib.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>
#include <random>

// WORKS AND IS STABLE (STRICTLY BETTER THAN Python Numpy VERSION)
namespace py = pybind11;
using namespace std;

// Helper functions - note that long double type creates issues for some numpy functions!

long double entropy(py::array_t<long double> xx, unsigned int base){
    auto x = xx.unchecked<1>();
    long double ent = 0.0;
    for(unsigned int i=0;i<x.shape(0);i++){
        if (x(i) > 0.0) ent -= x(i) * logl(x(i)) / logl((long double) base);
    }
    return ent;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("entropy", &entropy, "Entropy (long double safe)");
}
#ifndef BLACKBODY_HPP_INCLUDED
#define BLACKBODY_HPP_INCLUDED

#include <vec/tensor.hpp>

tensor<float, 3> blackbody_temperature_to_linear_rgb(float wavelength);
tensor<float, 3> blackbody_temperature_to_approximate_linear_rgb(double wavelength);
std::array<tensor<float, 3>, 100000> blackbody_table();

#endif // BLACKBODY_HPP_INCLUDED

#ifndef BLACKBODY_HPP_INCLUDED
#define BLACKBODY_HPP_INCLUDED

#include <vec/tensor.hpp>

tensor<float, 3> blackbody_temperature_to_linear_rgb(float wavelength);

#endif // BLACKBODY_HPP_INCLUDED

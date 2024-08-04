#ifndef INIT_BLACK_HOLE_HPP_INCLUDED
#define INIT_BLACK_HOLE_HPP_INCLUDED

#include <toolkit/opencl.hpp>
#include <vec/tensor.hpp>

struct black_hole_data
{
    cl::buffer aIJ;
};

struct black_hole_params
{
    float bare_mass = 0;
    tensor<float, 3> position = {0,0,0};
    tensor<float, 3> linear_momentum = {0,0,0};
    tensor<float, 3> angular_momentum = {0,0,0};
};

black_hole_data init_black_hole(const black_hole_params& params);

#endif // INIT_BLACK_HOLE_HPP_INCLUDED

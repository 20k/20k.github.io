#ifndef INIT_BLACK_HOLE_HPP_INCLUDED
#define INIT_BLACK_HOLE_HPP_INCLUDED

#include <toolkit/opencl.hpp>
#include <vec/tensor.hpp>
#include "../common/single_source.hpp"

struct black_hole_data
{
    std::array<cl::buffer, 6> aij;

    black_hole_data(cl::context& ctx) : aij{ctx, ctx, ctx, ctx, ctx, ctx}{}
};

struct black_hole_params
{
    float bare_mass = 0;
    tensor<float, 3> position = {0,0,0};
    tensor<float, 3> linear_momentum = {0,0,0};
    tensor<float, 3> angular_momentum = {0,0,0};
};

black_hole_data init_black_hole(cl::context& ctx, cl::command_queue& cqueue, const black_hole_params& params, tensor<int, 3> dim, float scale);

#endif // INIT_BLACK_HOLE_HPP_INCLUDED

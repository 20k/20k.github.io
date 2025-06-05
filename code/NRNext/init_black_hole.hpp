#ifndef INIT_BLACK_HOLE_HPP_INCLUDED
#define INIT_BLACK_HOLE_HPP_INCLUDED

#include <toolkit/opencl.hpp>
#include <vec/tensor.hpp>
#include "../common/single_source.hpp"
#include "value_alias.hpp"

struct black_hole_data
{
    std::array<cl::buffer, 6> aij;
    cl::buffer conformal_guess;

    black_hole_data(cl::context& ctx) : aij{ctx, ctx, ctx, ctx, ctx, ctx}, conformal_guess{ctx}{}
};

struct black_hole_params
{
    float bare_mass = 0;
    tensor<float, 3> position = {0,0,0};
    tensor<float, 3> linear_momentum = {0,0,0};
    tensor<float, 3> angular_momentum = {0,0,0};
};

tensor<valuef, 3, 3> get_pointlike_aIJ(v3f world_pos, v3f pos, v3f angular_momentum, v3f momentum);
black_hole_data init_black_hole(cl::context& ctx, cl::command_queue& cqueue, black_hole_params params, tensor<int, 3> dim, float scale);

#endif // INIT_BLACK_HOLE_HPP_INCLUDED

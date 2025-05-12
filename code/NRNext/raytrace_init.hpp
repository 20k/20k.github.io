#ifndef RAYTRACE_INIT_HPP_INCLUDED
#define RAYTRACE_INIT_HPP_INCLUDED

#include <toolkit/opencl.hpp>
#include "value_alias.hpp"

struct inverse_tetrad
{
    std::array<v4f, 4> v_lo;

    v4f into_frame_of_reference(v4f in)
    {
        v4f out;

        for(int i=0; i < 4; i++)
            out[i] = dot(in, v_lo[i]);

        return out;
    }
};

struct tetrad
{
    std::array<v4f, 4> v;

    v4f into_coordinate_space(v4f in)
    {
        return v[0] * in.x() + v[1] * in.y() + v[2] * in.z() + v[3] * in.w();
    }

    inverse_tetrad invert()
    {
        inverse_tetrad invert;

        tensor<valuef, 4, 4> as_matrix;

        for(int i=0; i < 4; i++)
        {
            for(int j=0; j < 4; j++)
            {
                as_matrix[i, j] = v[i][j];
            }
        }

        tensor<valuef, 4, 4> inv = as_matrix.asymmetric_invert();

        invert.v_lo[0] = {inv[0, 0], inv[0, 1], inv[0, 2], inv[0, 3]};
        invert.v_lo[1] = {inv[1, 0], inv[1, 1], inv[1, 2], inv[1, 3]};
        invert.v_lo[2] = {inv[2, 0], inv[2, 1], inv[2, 2], inv[2, 3]};
        invert.v_lo[3] = {inv[3, 0], inv[3, 1], inv[3, 2], inv[3, 3]};

        return invert;
    }
};

tetrad calculate_tetrad(m44f m, v3f local_velocity, bool should_orient);

void build_raytrace_init_kernels(cl::context ctx);

#endif // RAYTRACE_INIT_HPP_INCLUDED

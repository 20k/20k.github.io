#ifndef RAYTRACE_HPP_INCLUDED
#define RAYTRACE_HPP_INCLUDED

#include "../common/vec/tensor.hpp"
#include "../common/vec/dual.hpp"
#include "single_source.hpp"
#include "bssn.hpp"
#include "init.hpp"

using valuef = value<float>;
using valuei = value<int>;

using v2f = tensor<valuef, 2>;
using v3f = tensor<valuef, 3>;
using v4f = tensor<valuef, 4>;
using v2i = tensor<valuei, 2>;
using v3i = tensor<valuei, 3>;
using m44f = metric<valuef, 4, 4>;

using mut_v4f = tensor<mut<valuef>, 4>;
using mut_v3f = tensor<mut<valuef>, 3>;

template<typename T>
using dual = dual_types::dual_v<T>;

#define UNIVERSE_SIZE 100

template<typename T>
inline
auto function_trilinear(T&& func, v3f pos)
{
    v3f floored = floor(pos);
    v3f frac = pos - floored;

    v3i ipos = (v3i)floored;

    auto c000 = func(ipos + (v3i){0,0,0});
    auto c100 = func(ipos + (v3i){1,0,0});

    auto c010 = func(ipos + (v3i){0,1,0});
    auto c110 = func(ipos + (v3i){1,1,0});

    auto c001 = func(ipos + (v3i){0,0,1});
    auto c101 = func(ipos + (v3i){1,0,1});

    auto c011 = func(ipos + (v3i){0,1,1});
    auto c111 = func(ipos + (v3i){1,1,1});

    ///numerically symmetric across the centre of dim
    auto c00 = c000 - frac.x() * (c000 - c100);
    auto c01 = c001 - frac.x() * (c001 - c101);

    auto c10 = c010 - frac.x() * (c010 - c110);
    auto c11 = c011 - frac.x() * (c011 - c111);

    auto c0 = c00 - frac.y() * (c00 - c10);
    auto c1 = c01 - frac.y() * (c01 - c11);

    return c0 - frac.z() * (c0 - c1);
}

template<typename T>
inline
auto function_quadlinear(T&& func, v4f pos)
{
    v4f floored = floor(pos);
    v4i ipos = (v4i)floored;

    v4f frac = pos - floored;

    auto a000 = func(ipos + (v4i){0,0,0,0});
    auto a100 = func(ipos + (v4i){1,0,0,0});

    auto a010 = func(ipos + (v4i){0,1,0,0});
    auto a110 = func(ipos + (v4i){1,1,0,0});

    auto a001 = func(ipos + (v4i){0,0,1,0});
    auto a101 = func(ipos + (v4i){1,0,1,0});

    auto a011 = func(ipos + (v4i){0,1,1,0});
    auto a111 = func(ipos + (v4i){1,1,1,0});

    auto a00 = a000 - frac.x() * (a000 - a100);
    auto a01 = a001 - frac.x() * (a001 - a101);

    auto a10 = a010 - frac.x() * (a010 - a110);
    auto a11 = a011 - frac.x() * (a011 - a111);

    auto a0 = a00 - frac.y() * (a00 - a10);
    auto a1 = a01 - frac.y() * (a01 - a11);

    auto linear_1 = a0 - frac.z() * (a0 - a1);

    auto c000 = func(ipos + (v4i){0,0,0,1});
    auto c100 = func(ipos + (v4i){1,0,0,1});

    auto c010 = func(ipos + (v4i){0,1,0,1});
    auto c110 = func(ipos + (v4i){1,1,0,1});

    auto c001 = func(ipos + (v4i){0,0,1,1});
    auto c101 = func(ipos + (v4i){1,0,1,1});

    auto c011 = func(ipos + (v4i){0,1,1,1});
    auto c111 = func(ipos + (v4i){1,1,1,1});

    auto c00 = c000 - frac.x() * (c000 - c100);
    auto c01 = c001 - frac.x() * (c001 - c101);

    auto c10 = c010 - frac.x() * (c010 - c110);
    auto c11 = c011 - frac.x() * (c011 - c111);

    auto c0 = c00 - frac.y() * (c00 - c10);
    auto c1 = c01 - frac.y() * (c01 - c11);

    auto linear_2 = c0 - frac.z() * (c0 - c1);

    return linear_1 - frac.w() * (linear_1 - linear_2);
}

inline
adm_variables adm_at(v3i pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    bssn_args args(pos, dim, in);

    return bssn_to_adm(args);
}

///takes GRID coordinates
inline
adm_variables admf_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    auto Yij_at = [&](v3i pos)
    {
        return adm_at(pos, dim, in).Yij;
    };

    auto Kij_at = [&](v3i pos)
    {
        return adm_at(pos, dim, in).Kij;
    };

    auto gA_at = [&](v3i pos)
    {
        return adm_at(pos, dim, in).gA;
    };

    auto gB_at = [&](v3i pos)
    {
        return adm_at(pos, dim, in).gB;
    };

    adm_variables out;
    out.Yij = function_trilinear(Yij_at, pos);
    out.Kij = function_trilinear(Kij_at, pos);
    out.gA = function_trilinear(gA_at, pos);
    out.gB = function_trilinear(gB_at, pos);

    pin(out.Yij);
    pin(out.Kij);
    pin(out.gA);
    pin(out.gB);

    return out;
}

using block_precision_t = valuef;

void init_rays3(execution_context& ectx, literal<valuei> screen_width, literal<valuei> screen_height,
                buffer_mut<v4f> positions_out,
                buffer_mut<v4f> velocities_out,
                buffer<v4f> e0, buffer<v4f> e1, buffer<v4f> e2, buffer<v4f> e3,
                buffer<v4f> position, literal<v4f> camera_quat,
                literal<v3i> dim, literal<valuef> scale, literal<valuei> is_adm,
                bssn_args_mem<buffer<valuef>> in);

void trace3(execution_context& ectx, literal<valuei> screen_width, literal<valuei> screen_height,
            read_only_image<2> background, write_only_image<2> screen,
            literal<valuei> background_width, literal<valuei> background_height,
            literal<v4f> camera_quat,
            buffer<v4f> positions, buffer<v4f> velocities,
            literal<v3i> dim,
            literal<valuef> scale,
            bssn_args_mem<buffer<valuef>> in,
            bssn_derivatives_mem<buffer<derivative_t>> derivatives);

#if 0
void trace4(execution_context& ectx, literal<valuei> screen_width, literal<valuei> screen_height,
            read_only_image<2> background, write_only_image<2> screen,
            literal<valuei> background_width, literal<valuei> background_height,
            literal<v4f> camera_quat,
            buffer_mut<v4f> positions, buffer_mut<v3f> velocities,
            literal<v3i> dim,
            literal<valuef> scale,
            literal<valuef> time_lower,
            literal<valuef> time_upper,
            bssn_args_mem<buffer<valuef>> lower,
            bssn_args_mem<buffer<valuef>> upper,
            bssn_derivatives_mem<buffer<derivative_t>> lower_derivatives,
            bssn_derivatives_mem<buffer<derivative_t>> upper_derivatives,
            buffer_mut<valuei> full_result);
#endif

inline
valuef dot(v4f u, v4f v, m44f m) {
    v4f lowered = m.lower(u);

    return dot(lowered, v);
}

inline
v4f gram_project(v4f u, v4f v, m44f m) {
    valuef top = dot_metric(u, v, m);
    valuef bottom = dot_metric(u, u, m);

    return (top / bottom) * u;
}

inline
v4f normalise(v4f in, m44f m)
{
    valuef d = dot_metric(in, in, m);

    return in / sqrt(fabs(d));
}

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

inline
tetrad gram_schmidt(v4f v0, v4f v1, v4f v2, v4f v3, m44f m)
{
    using namespace single_source;

    v4f u0 = v0;

    v4f u1 = v1;
    u1 = u1 - gram_project(u0, u1, m);

    pin(u1);

    v4f u2 = v2;
    u2 = u2 - gram_project(u0, u2, m);
    u2 = u2 - gram_project(u1, u2, m);

    pin(u2);

    v4f u3 = v3;
    u3 = u3 - gram_project(u0, u3, m);
    u3 = u3 - gram_project(u1, u3, m);
    u3 = u3 - gram_project(u2, u3, m);

    pin(u3);

    u0 = normalise(u0, m);
    u1 = normalise(u1, m);
    u2 = normalise(u2, m);
    u3 = normalise(u3, m);

    pin(u0);
    pin(u1);
    pin(u2);
    pin(u3);

    return {u0, u1, u2, u3};
}

template<typename T>
inline
void swap(const T& v1, const T& v2)
{
    auto intermediate = single_source::declare_e(v1);
    as_ref(v1) = v2;
    as_ref(v2) = intermediate;
}

///specifically: cartesian minkowski
inline
m44f get_local_minkowski(const tetrad& tetrads, const m44f& met)
{
    m44f minkowski;

    tensor<valuef, 4, 4> m;

    for(int i=0; i < 4; i++)
    {
        m[0, i] = tetrads.v[0][i];
        m[1, i] = tetrads.v[1][i];
        m[2, i] = tetrads.v[2][i];
        m[3, i] = tetrads.v[3][i];
    }

    for(int a=0; a < 4; a++)
    {
        for(int b=0; b < 4; b++)
        {
            valuef sum = 0;

            for(int mu=0; mu < 4; mu++)
            {
                for(int v=0; v < 4; v++)
                {
                    sum += met[mu, v] * m[a, mu] * m[b, v];
                }
            }

            minkowski[a, b] = sum;
        }
    }

    return minkowski;
}

inline
valuei calculate_which_coordinate_is_timelike(const tetrad& tetrads, const m44f& met)
{
    m44f minkowski = get_local_minkowski(tetrads, met);

    using namespace single_source;

    mut<valuei> lowest_index = declare_mut_e(valuei(0));
    mut<valuef> lowest_value = declare_mut_e(valuef(0));

    for(int i=0; i < 4; i++)
    {
        if_e(minkowski[i, i] < lowest_value, [&] {
            as_ref(lowest_index) = valuei(i);
            as_ref(lowest_value) = minkowski[i, i];
        });
    }

    return lowest_index;
}

inline
v4f get_timelike_vector(v3f velocity, const tetrad& tetrads)
{
    v4f coordinate_time = {1, velocity.x(), velocity.y(), velocity.z()};

    valuef lorentz_factor = 1/sqrt(1 - (velocity.x() * velocity.x() + velocity.y() * velocity.y() + velocity.z() * velocity.z()));

    v4f proper_time = lorentz_factor * coordinate_time;

    ///put into curved spacetime
    return proper_time.x() * tetrads.v[0] + proper_time.y() * tetrads.v[1] + proper_time.z() * tetrads.v[2] + proper_time.w() * tetrads.v[3];
}

inline
tetrad boost_tetrad(v3f velocity, const tetrad& tetrads, const metric<valuef, 4, 4>& m)
{
    using namespace single_source;

    v4f u = tetrads.v[0];
    v4f v = get_timelike_vector(velocity, tetrads);

    v4f u_l = m.lower(u);
    v4f v_l = m.lower(v);

    valuef Y = -dot(v_l, u);

    ///https://arxiv.org/pdf/2404.05744 18
    tensor<valuef, 4, 4> B;

    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            valuef kronecker = (i == j) ? 1 : 0;

            B[i, j] = kronecker + ((v[i] + u[i]) * (v_l[j] + u_l[j]) / (1 + Y)) - 2 * v[i] * u_l[j];
        }
    }

    tetrad next;

    for(int a=0; a < 4; a++)
    {
        for(int i=0; i < 4; i++)
        {
            valuef sum = 0;

            for(int j=0; j < 4; j++)
            {
                sum += B[i, j] * tetrads.v[a][j];
            }

            next.v[a][i] = sum;
        }
    }

    return next;
}

inline
v3f project(v3f u, v3f v)
{
    return (dot(u, v) / dot(u, u)) * u;
}

inline
std::array<v3f, 3> orthonormalise(v3f i1, v3f i2, v3f i3)
{
    v3f u1 = i1;
    v3f u2 = i2;
    v3f u3 = i3;

    u2 = u2 - project(u1, u2);

    u3 = u3 - project(u1, u3);
    u3 = u3 - project(u2, u3);

    u1 = u1.norm();
    u2 = u2.norm();
    u3 = u3.norm();

    return {u1, u2, u3};
};

void build_raytrace_kernels(cl::context ctx);

#endif // RAYTRACE_HPP_INCLUDED

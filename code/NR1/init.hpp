#ifndef INIT_HPP_INCLUDED
#define INIT_HPP_INCLUDED

#include <string>
#include "bssn.hpp"
#include "bssn.hpp"
#include "../common/vec/dual.hpp"
#include "derivatives.hpp"
#include "tensor_algebra.hpp"

template<typename T>
using dual = dual_types::dual_v<T>;


inline
auto diff_analytic(auto&& func, const v4f& position, int direction) {
    auto pinned = position;
    single_source::pin(pinned);

    m44f metric = func(pinned);

    tensor<valuef, 4, 4> differentiated;

    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            dual<value_base> as_dual = replay_value_base<dual<value_base>>(metric[i, j], [&](const value_base& in)
            {
                if(equivalent(in, pinned[direction]))
                    return dual<value_base>(in, in.make_constant_of_type(1.f));
                else
                    return dual<value_base>(in, in.make_constant_of_type(0.f));
            });

            differentiated[i, j].set_from_base(as_dual.dual);
        }
    }

    return differentiated;
}

template<typename T>
inline
bssn_args metric_to_bssn(T&& func, v4f pos, v3i dim, valuef scale)
{
    using namespace single_source;

    bssn_args args;

    v3i centre = (dim - (v3i){1,1,1}) / 2;

    v3f fpos = (v3f)pos.yzw();
    v3f fcentre = (v3f)centre;

    v3f wpos = (fpos - fcentre) * scale;

    metric<valuef, 4, 4> Guv = func((v4f){pos.x() * scale, wpos.x(), wpos.y(), wpos.z()});

    tensor<valuef, 4, 4, 4> dGuv;

    for(int k=0; k < 4; k++)
    {
        auto ldguv = diff_analytic(func, (v4f){pos.x() * scale, wpos.x(), wpos.y(), wpos.z()}, k);

        for(int i=0; i < 4; i++)
        {
            for(int j=0; j < 4; j++)
            {
                dGuv[k, i, j] = ldguv[i, j];
            }
        }
    }

    metric<valuef, 3, 3> Yij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Yij[i, j] = Guv[i+1, j+1];
        }
    }

    tensor<valuef, 3, 3, 3> Yij_derivatives;

    for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                Yij_derivatives[k, i, j] = dGuv[k+1, i+1, j+1];
            }
        }
    }

    tensor<valuef, 3, 3, 3> Yij_christoffel = christoffel_symbols_2(Yij.invert(), Yij_derivatives);

    pin(Yij_christoffel);

    auto covariant_derivative_low_vec_e = [&](const tensor<valuef, 3>& lo, const tensor<valuef, 3, 3>& dlo)
    {
        ///DcXa
        tensor<valuef, 3, 3> ret;

        for(int a=0; a < 3; a++)
        {
            for(int c=0; c < 3; c++)
            {
                valuef sum = 0;

                for(int b=0; b < 3; b++)
                {
                    sum += Yij_christoffel[b, c, a] * lo[b];
                }

                ret[c, a] = dlo[c, a] - sum;
            }
        }

        return ret;
    };

    tensor<valuef, 3> gB_lower;
    tensor<valuef, 3, 3> dgB_lower;

    for(int i=0; i < 3; i++)
    {
        gB_lower[i] = Guv[0, i+1];

        for(int k=0; k < 3; k++)
        {
            dgB_lower[k, i] = dGuv[k+1, 0, i+1];
        }
    }

    tensor<valuef, 3> gB = raise_index(gB_lower, Yij.invert(), 0);

    pin(gB);

    valuef gB_sum = sum_multiply(gB, gB_lower);

    ///g00 = nini - n^2
    ///g00 - nini = -n^2
    ///-g00 + nini = n^2
    ///n = sqrt(-g00 + nini)
    valuef gA = sqrt(-Guv[0, 0] + gB_sum);

    ///https://clas.ucdenver.edu/math-clinic/sites/default/files/attached-files/master_project_mach_.pdf 4-19a
    tensor<valuef, 3, 3> DigBj = covariant_derivative_low_vec_e(gB_lower, dgB_lower);

    pin(DigBj);

    tensor<valuef, 3, 3> Kij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Kij[i, j] = (1/(2 * gA)) * (DigBj[i, j] + DigBj[j, i] - dGuv[0, i+1, j+1]);
        }
    }

    valuef W = pow(Yij.det(), -1/6.f);
    metric<valuef, 3, 3> cY = W*W * Yij;
    valuef K = trace(Kij, Yij.invert());

    tensor<valuef, 3, 3> cA = W*W * (Kij - (1.f/3.f) * Yij.to_tensor() * K);

    tensor<int, 2> index_table[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    /*for(int i=0; i < 6; i++)
    {
        tensor<int, 2> idx = index_table[i];

        as_ref(to_fill.cY[i][lid]) = cY[idx.x(), idx.y()];
        as_ref(to_fill.cA[i][lid]) = cA[idx.x(), idx.y()];
    }

    for(int i=0; i < 3; i++)
    {
        as_ref(to_fill.cG[i][lid]) = valuef(0);
        as_ref(to_fill.gB[i][lid]) = gB[i];
    }

    as_ref(to_fill.gA[lid]) = gA;
    as_ref(to_fill.W[lid]) = W;
    as_ref(to_fill.K[lid]) = K;*/

    args.cY = cY;
    args.cA = cA;
    args.cG = (v3f){0,0,0};
    args.gB = gB;
    args.gA = gA;
    args.W = W;
    args.K = K;
    return args;
}

std::string make_initial_conditions();
std::string init_christoffel();

#endif // INIT_HPP_INCLUDED

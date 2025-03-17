#include "bssn.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include "tensor_algebra.hpp"
#include "derivatives.hpp"
#include <iostream>
#include "../common/vec/dual.hpp"
#include "plugin.hpp"
#include "init_general.hpp"

template<typename T>
using dual = dual_types::dual_v<T>;

///https://arxiv.org/pdf/2405.06035 (92)
template<typename T>
tensor<T, 3> calculate_cG(const inverse_metric<T, 3, 3>& icY, const tensor<T, 3, 3, 3>& dcY)
{
    tensor<T, 3> ret;

    for(int i=0; i < 3; i++)
    {
        T sum = 0;

        for(int j=0; j < 3; j++)
        {
            T lsum = 0;

            for(int k=0; k < 3; k++)
            {
                for(int l=0; l < 3; l++)
                {
                    lsum += icY[k, l] * dcY[k, l, j];
                }
            }

            sum += icY[i, j] * lsum;
        }

        ret[i] = sum;
    }

    return ret;
}

tensor<valuef, 3> bssn_args::cG_undiff(const bssn_derivatives& derivs)
{
    //#define SUBSTITUTE_CG
    #ifdef SUBSTITUTE_CG
    auto icY = cY.invert();

    single_source::pin(icY);

    return calculate_cG(icY, derivs.dcY);;
    #else
    return cG;
    #endif
}

v3i get_coordinate(valuei id, v3i dim)
{
    using namespace single_source;

    valuei x = id % dim.x();
    valuei y = (id / dim.x()) % dim.y();
    valuei z = id / (dim.x() * dim.y());

    pin(x);
    pin(y);
    pin(z);

    return {x, y, z};
}

v3i get_coordinate_including_boundary(valuei id, v3i dim)
{
    return get_coordinate_including_boundary(id, dim, 2);
}

v3i get_coordinate_including_boundary(valuei id, v3i dim, int boundary_size)
{
    return get_coordinate(id, dim - (v3i){boundary_size*2,boundary_size*2,boundary_size*2}) + (v3i){boundary_size,boundary_size,boundary_size};
}

///so: a much better variant of this would be to only calculate aij_raised's derivative and store s1
///because that's the only component we actually *need* to calculate the momentum constraint in the evolution kernel
tensor<valuef, 3> calculate_momentum_constraint(bssn_args& args, const derivative_data& d, v3f Si_lower)
{
    tensor<valuef, 3> dW;

    for(int i=0; i < 3; i++)
        dW[i] = diff1(args.W, i, d);

    ///https://arxiv.org/pdf/1205.5111v1.pdf (54)
    tensor<valuef, 3, 3> aij_raised = raise_index(args.cA, args.cY.invert(), 1);

    tensor<valuef, 3> dPhi = -dW / (2 * max(args.W, valuef(0.0001f)));

    tensor<valuef, 3> Mi;

    for(int i=0; i < 3; i++)
    {
        valuef s1 = 0;

        for(int j=0; j < 3; j++)
        {
            s1 += diff1(aij_raised[i, j], j, d);
        }

        valuef s2 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                s2 += -0.5f * args.cY.invert()[j, k] * diff1(args.cA[j, k], i, d);
            }
        }

        valuef s3 = 0;

        for(int j=0; j < 3; j++)
        {
            s3 += 6 * dPhi[j] * aij_raised[i, j];
        }

        valuef s4 = -(2.f/3.f) * diff1(args.K, i, d);

        Mi[i] = s1 + s2 + s3 + s4;
    }

    Mi += -8 * M_PI * Si_lower;

    return Mi;
}

valuef calculate_momentum_constraint_summed(bssn_args& args, const derivative_data& d, v3f Si_lower)
{
    v3f cst = calculate_momentum_constraint(args, d, Si_lower);

    return sqrt(cst.x() * cst.x() + cst.y() * cst.y() + cst.z() * cst.z());
}


void make_derivatives(cl::context ctx)
{
    auto differentiate = [](execution_context&, buffer<valuef> in, std::array<buffer_mut<derivative_t>, 3> out, literal<v3i> ldim, literal<valuef> scale, literal<valuei> work_size)
    {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= work_size.get(), [&] {
            return_e();
        });

        v3i pos = get_coordinate_including_boundary(lid, dim, 1);
        pin(pos);

        valuef v1 = in[pos, dim];

        for(int i=0; i < 3; i++)
            as_ref(out[i][pos, dim]) = (derivative_t)diff1_boundary(in, i, scale.get(), pos, dim);
    };

    cl::async_build_and_cache(ctx, [=]
    {
        return value_impl::make_function(differentiate, "differentiate");;
    }, {"differentiate"});
}

tensor<valuef, 3, 3> calculate_cRij(bssn_args& args, bssn_derivatives& derivs, const derivative_data& d)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    pin(icY);

    auto christoff1 = christoffel_symbols_1(derivs.dcY);
    auto christoff2 = christoffel_symbols_2(icY, derivs.dcY);

    pin(christoff1);
    pin(christoff2);

    tensor<valuef, 3, 3> cRij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            valuef s1 = 0;

            for(int l=0; l < 3; l++)
            {
                for(int m=0; m < 3; m++)
                {
                    s1 += -0.5f * icY[l, m] * diff2(args.cY[i, j], m, l, derivs.dcY[m, i, j], derivs.dcY[l, i, j], d);
                }
            }

            valuef s2 = 0;

            for(int k=0; k < 3; k++)
            {
                s2 += 0.5f * (args.cY[k, i] * diff1(args.cG[k], j, d) + args.cY[k, j] * diff1(args.cG[k], i, d));
            }

            valuef s3 = 0;

            for(int k=0; k < 3; k++)
            {
                s3 += 0.5f * args.cG_undiff(derivs)[k] * (christoff1[i, j, k] + christoff1[j, i, k]);
            }

            valuef s4 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int l=0; l < 3; l++)
                {
                    valuef inner1 = 0;
                    valuef inner2 = 0;

                    for(int k=0; k < 3; k++)
                    {
                        inner1 += 0.5f * (2 * christoff2[k, l, i] * christoff1[j, k, m] + 2 * christoff2[k, l, j] * christoff1[i, k, m]);
                    }

                    for(int k=0; k < 3; k++)
                    {
                        inner2 += christoff2[k, i, m] * christoff1[k, l, j];
                    }

                    s4 += icY[l, m] * (inner1 + inner2);
                }
            }

            cRij[i, j] = s1 + s2 + s3 + s4;
        }
    }

    pin(cRij);

    return cRij;
}

///https://arxiv.org/pdf/1307.7391 (9)
///https://iopscience.iop.org/article/10.1088/1361-6382/ac7e16/pdf 2.6
///this calculates the quantity W^2 * Rij
tensor<valuef, 3, 3> calculate_W2Rij(bssn_args& args, bssn_derivatives& derivs, const derivative_data& d)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    pin(icY);

    auto christoff2 = christoffel_symbols_2(icY, derivs.dcY);

    pin(christoff2);

    tensor<valuef, 3, 3> didjW;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            didjW[i, j] = double_covariant_derivative(args.W, derivs.dW, christoff2, d)[j, i];
        }
    }

    tensor<valuef, 3, 3> w2Rphiij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            valuef v1 = didjW[i, j];
            valuef v2 = 0;

            {
                valuef sum = 0;

                auto raised = icY.raise(didjW, 0);

                for(int l=0; l < 3; l++)
                {
                    sum += raised[l, l];
                }

                v2 = args.cY[i, j] * sum;
            }

            valuef v3 = 0;

            {
                valuef sum = 0;

                for(int l=0; l < 3; l++)
                {
                    sum += icY.raise(derivs.dW)[l] * derivs.dW[l];
                }

                v3 = -2 * args.cY[i, j] * sum;
            }

            w2Rphiij[i, j] = args.W * (v1 + v2) + v3;
        }
    }

    pin(w2Rphiij);

    return w2Rphiij + calculate_cRij(args, derivs, d) * args.W * args.W;
}

tensor<valuef, 3, 3> calculate_adm_Rij(const tensor<valuef, 3, 3, 3>& christoff2, const derivative_data& d)
{
    tensor<valuef, 3, 3> ret;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            valuef s1 = 0;
            valuef s2 = 0;
            valuef s3 = 0;
            valuef s4 = 0;

            for(int k=0; k < 3; k++)
            {
                s1 += diff1(christoff2[k, i, j], k, d);

                s2 += -diff1(christoff2[k, k, j], i, d);

                for(int l=0; l < 3; l++)
                {
                    s3 += christoff2[l, i, j] * christoff2[k, l, k];
                    s4 += -christoff2[l, k, j] * christoff2[k, l, i];
                }
            }

            ret[i, j] = s1 + s2 + s3 + s4;
        }
    }

    return ret;
}

valuef calculate_hamiltonian_constraint_adm(bssn_args& args, bssn_derivatives& derivs, const derivative_data& d, valuef rho_s)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    auto christoff2 = christoffel_symbols_2(args.cY.invert(), derivs.dcY);
    auto fchristoff2 = get_full_christoffel2(args.W, derivs.dW, args.cY, args.cY.invert(), christoff2);

    auto W2Rij = args.W * args.W * calculate_adm_Rij(fchristoff2, d);

    valuef R = trace(W2Rij, icY);

    pin(icY);

    tensor<valuef, 3, 3> AMN = icY.raise(icY.raise(args.cA, 0), 1);

    valuef AMN_Amn = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            AMN_Amn += AMN[i, j] * args.cA[i, j];
        }
    }

    return R + (2.f/3.f) * args.K * args.K - AMN_Amn - 16 * M_PI * rho_s;
}

valuef calculate_hamiltonian_constraint(bssn_args& args, bssn_derivatives& derivs, const derivative_data& d, valuef rho_s)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    auto W2Rij = calculate_W2Rij(args, derivs, d);
    valuef R = trace(W2Rij, icY);

    pin(icY);

    tensor<valuef, 3, 3> AMN = icY.raise(icY.raise(args.cA, 0), 1);

    valuef AMN_Amn = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            AMN_Amn += AMN[i, j] * args.cA[i, j];
        }
    }

    return R + (2.f/3.f) * args.K * args.K - AMN_Amn - 16 * M_PI * rho_s;
}

#define BLACK_HOLE_GAUGE
#ifdef BLACK_HOLE_GAUGE
//#define SHOCK_AVOID
#define ONE_PLUS_LOG
#define GAMMA_DRIVER
#endif // BLACK_HOLE_GAUGE

//#define WAVE_TEST
#ifdef WAVE_TEST
#define HARMONIC_SLICING
#define ZERO_SHIFT
#endif // WAVE_TEST

//#define WAVE_TEST2
#ifdef WAVE_TEST2
#define ONE_LAPSE
#define ZERO_SHIFT
#endif

valuef get_dtgA(bssn_args& args, bssn_derivatives& derivs, const derivative_data& d, valuef total_elapsed, float lapse_damp_timescale)
{
    valuef bmdma = 0;

    for(int i=0; i < 3; i++)
    {
        bmdma += args.gB[i] * diff1(args.gA, i, d);
    }

    #define LAPSE_DAMPING
    #ifdef LAPSE_DAMPING
    valuef damp = 0.f;

    //45 for stationary matter sim tests
    if(lapse_damp_timescale > 0)
    {
        valuef sigma = lapse_damp_timescale;
        value h = (3.f/5.f);

        damp = args.W * (h * exp(-(total_elapsed*total_elapsed) / (2 * sigma * sigma))) * (args.gA - args.W);
    }
    #else
    valuef damp = 0;
    #endif

    #ifdef SHOCK_AVOID
    return -(8.f/3.f) * args.gA * args.K / (3 - args.gA) + bmdma - damp;
    #endif

    ///https://arxiv.org/pdf/gr-qc/0206072
    #ifdef ONE_PLUS_LOG
    return -2 * args.gA * args.K + bmdma * 1 - damp;
    #endif // ONE_PLUS_LOG

    ///https://arxiv.org/pdf/2201.08857
    #ifdef HARMONIC_SLICING
    return -args.gA * args.gA * args.K + bmdma;
    #endif // HARMONIC_SLICING

    #ifdef ONE_LAPSE
    return 0;
    #endif // ONE_LAPSE
}

///todo: https://arxiv.org/pdf/1003.4681
tensor<valuef, 3> get_dtgB(bssn_args& args, bssn_derivatives& derivs, const derivative_data& d, const initial_params& cfg)
{
    ///https://arxiv.org/pdf/gr-qc/0605030 (26)
    #ifdef GAMMA_DRIVER
    tensor<valuef, 3> djbjbi;

    for(int i=0; i < 3; i++)
    {
        valuef sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum += args.gB[j] * diff1(args.gB[i], j, d);
        }

        djbjbi[i] = sum;
    }

    ///gauge damping parameter, commonly set to 2
    valuef N = cfg.N;

    //#define VARIABLE_DAMP
    #ifdef VARIABLE_DAMP
    {
        N = 0.05f;

        using namespace single_source;

        auto icY = args.cY.invert();
        pin(icY);

        {
            float R0 = 1.31f;

            valuef W = clamp(args.W, 1e-3f, 0.95f);

            float a = 2;
            float b = 2;

            valuef sum = 0;

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    sum += icY.idx(i, j) * diff1(W, i, d) * diff1(W, j, d);
                }
            }

            valuef result = R0 * sqrt(sum) / pow(1 - pow(W, a), b);

            /*if_e(result >= N, [&]{
                print("N %f dW %f %f %f divisor %f\n", result, diff1(W, 0, d), diff1(W, 1, d), diff1(W, 2, d), pow(1 - pow(W, a), b));
            });*/

            N = max(N, result);
        }
    }
    #endif

    return (3/4.f) * args.cG_undiff(derivs) + djbjbi * 1 - N * args.gB;
    #endif // GAMMA_DRIVER

    #ifdef ZERO_SHIFT
    return {0,0,0};
    #endif // ZERO_SHIFT
}

tensor<valuef, 3, 3> get_dtcY(bssn_args& args, bssn_derivatives& derivs, const derivative_data& d, valuef rho_s)
{
    using namespace single_source;

    inverse_metric<valuef, 3, 3> icY = args.cY.invert();
    pin(icY);

    ///https://arxiv.org/pdf/1307.7391 specifically for why the trace free aspect
    ///https://arxiv.org/pdf/1106.2254 also see here, after 25
    auto dtcY = lie_derivative_weight(args.gB, args.cY.to_tensor(), d) - 2 * args.gA * args.cA;

    #define CY_STABILITY
    #ifdef CY_STABILITY
    tensor<valuef, 3, 3> d_cGi;

    for(int m=0; m < 3; m++)
    {
        tensor<dual<valuef>, 3, 3, 3> d_dcYij;
        unit_metric<dual<valuef>, 3, 3> d_cYij;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                d_cYij[i, j].real = args.cY[i, j];
                d_cYij[i, j].dual = derivs.dcY[m, i, j];
            }
        }

        pin(d_cYij);

        auto dicY = d_cYij.invert();

        pin(dicY);

        for(int k=0; k < 3; k++)
        {
            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    d_dcYij[k, i, j].real = derivs.dcY[k, i, j];
                    d_dcYij[k, i, j].dual = diff1(derivs.dcY[k, i, j], m, d);
                }
            }
        }

        pin(d_dcYij);

        tensor<dual<valuef>, 3> dcGi_G = calculate_cG(dicY, d_dcYij);
        pin(dcGi_G);

        for(int i=0; i < 3; i++)
        {
            d_cGi[m, i] = diff1(args.cG[i], m, d) - dcGi_G[i].dual;
        }
    }

    auto christoff2 = christoffel_symbols_2(icY, derivs.dcY);
    pin(christoff2);

    tensor<valuef, 3> calculated_cG = calculate_cG(icY, derivs.dcY);

    tensor<valuef, 3> Gi = args.cG - calculated_cG;

    tensor<valuef, 3, 3> cD = covariant_derivative_high_vec(Gi, d_cGi, christoff2);

    pin(cD);

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            valuef sum = 0;

            for(int k=0; k < 3; k++)
            {
                sum += 0.5f * (args.cY[k, i] * cD[k, j] + args.cY[k, j] * cD[k, i]);
            }

            float cK = -0.18f;

            dtcY.idx(i, j) += cK * args.gA * sum;
        }
    }
    #endif

    //this appears to do literally nothing
    //dtcY += 50.5f * args.gA * args.cY.to_tensor() * -calculate_hamiltonian_constraint(args, derivs, d, rho_s);

    return dtcY;
}

///https://iopscience.iop.org/article/10.1088/1361-6382/ac7e16/pdf 2.12 or
///https://arxiv.org/pdf/0709.2160
valuef get_dtW(bssn_args& args, bssn_derivatives& derivs, const derivative_data& d, const valuef& rho_s, valuef timestep)
{
    valuef dibi = 0;

    for(int i=0; i < 3; i++)
    {
        dibi += diff1(args.gB[i], i, d);
    }

    valuef dibiw = 0;

    for(int i=0; i < 3; i++)
    {
        dibiw += args.gB[i] * diff1(args.W, i, d);
    }

    return (1/3.f) * args.W * (args.gA * args.K - dibi) + dibiw + 0.15f * timestep * args.W * calculate_hamiltonian_constraint(args, derivs, d, rho_s);
}

tensor<valuef, 3, 3> calculate_W2DiDja(bssn_args& args, bssn_derivatives& derivs, const derivative_data& d)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    pin(icY);

    auto christoff2 = christoffel_symbols_2(icY, derivs.dcY);

    pin(christoff2);

    tensor<valuef, 3, 3> W2DiDja;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            valuef v1 = args.W * double_covariant_derivative(args.gA, derivs.dgA, christoff2, d)[i, j];

            valuef v2 = derivs.dW[i] * diff1(args.gA, j, d) + derivs.dW[j] * diff1(args.gA, i, d);

            valuef sum = 0;

            for(int m=0; m < 3; m++)
            {
                valuef lsum = 0;

                for(int n=0; n < 3; n++)
                {
                    lsum += icY[m, n] * diff1(args.gA, n, d);
                }

                sum += lsum * derivs.dW[m];
            }

            valuef v3 = -args.cY[i, j] * sum;

            W2DiDja[i, j] = args.W * (v1 + v2 + v3);
        }
    }

    return W2DiDja;
}

valuef get_dtK(bssn_args& args, bssn_derivatives& derivs, const derivative_data& d, valuef S, valuef rho_s)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    pin(icY);

    tensor<valuef, 3, 3> W2DiDja = calculate_W2DiDja(args, derivs, d);
    pin(W2DiDja);

    tensor<valuef, 3, 3> AMN = icY.raise(icY.raise(args.cA, 0), 1);
    pin(AMN);

    valuef v1 = 0;

    for(int m=0; m < 3; m++)
    {
        v1 += args.gB[m] * diff1(args.K, m, d);
    }

    return v1 - sum_multiply(icY.to_tensor(), W2DiDja)
              + args.gA * (sum_multiply(AMN, args.cA)
              + (1/3.f) * args.K * args.K
              + 4 * M_PI * (S + rho_s));
}

tensor<valuef, 3, 3> get_dtcA(bssn_args& args, bssn_derivatives& derivs, v3h momentum_constraint, const derivative_data& d, tensor<valuef, 3, 3> W2_Sij)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    pin(icY);

    auto W2DiDja = calculate_W2DiDja(args, derivs, d);
    pin(W2DiDja);

    tensor<valuef, 3, 3> with_trace = args.gA * calculate_W2Rij(args, derivs, d) - W2DiDja - 8 * M_PI * args.gA * W2_Sij;

    tensor<valuef, 3, 3> aij_amj;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            tensor<valuef, 3, 3> raised_Aij = icY.raise(args.cA, 0);

            valuef sum = 0;

            for(int m=0; m < 3; m++)
            {
                sum += args.cA[i, m] * raised_Aij[m, j];
            }

            aij_amj[i, j] = sum;
        }
    }

    tensor<valuef, 3, 3> dtcA = lie_derivative_weight(args.gB, args.cA, d)
                                + args.gA * (args.K * args.cA
                                - 2 * aij_amj)
                                + trace_free(with_trace, args.cY, icY);

    #ifdef MOMENTUM_CONSTRAINT_DAMPING
    auto christoff2 = christoffel_symbols_2(icY, derivs.dcY);
    pin(christoff2);

    tensor<valuef, 3, 3> dMi;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            dMi[i, j] = diff1((valuef)momentum_constraint[j], i, d);
        }
    }

    auto cd_low = covariant_derivative_low_vec((v3f)momentum_constraint, dMi, christoff2);
    pin(cd_low);

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            float Ka = 0.04f;

            dtcA[i, j] += Ka * args.gA * 0.5f *
                              (cd_low[i, j]
                             + cd_low[j, i]);
        }
    }
    #endif

    return dtcA;
}

tensor<valuef, 3> get_dtcG(bssn_args& args, bssn_derivatives& derivs, const derivative_data& d, v3f Si)
{
    using namespace single_source;

    inverse_metric<valuef, 3, 3> icY = args.cY.invert();
    pin(icY);

    tensor<valuef, 3, 3, 3> christoff2 = christoffel_symbols_2(icY, derivs.dcY);
    pin(christoff2);

    tensor<valuef, 3> calculated_cG = calculate_cG(icY, derivs.dcY);

    tensor<valuef, 3> Gi = args.cG - calculated_cG;

    tensor<valuef, 3> dtcG;

    ///dtcG
    {
        tensor<valuef, 3, 3> icAij = icY.raise(icY.raise(args.cA, 0), 1);

        pin(icAij);

        tensor<valuef, 3> Yij_Kj;

        //#define PAPER_CGI_DAMP
        #ifdef PAPER_CGI_DAMP
        {
            auto christoff2 = christoffel_symbols_2(icY, derivs.dcY);

            tensor<valuef, 3> calculated_cG;

            for(int i=0; i < 3; i++)
            {
                valuef sum = 0;

                for(int m=0; m < 3; m++)
                {
                    for(int n=0; n < 3; n++)
                    {
                        sum += icY[m, n] * christoff2[i, m, n];
                    }
                }

                calculated_cG[i] = sum;
            }

            tensor<dual<valuef>, 3, 3, 3> dicY;

            for(int k=0; k < 3; k++)
            {
                unit_metric<dual<valuef>, 3, 3> cYk;

                for(int i=0; i < 3; i++)
                {
                    for(int j=0; j < 3; j++)
                    {
                        dual<valuef> dl;
                        dl.real = args.cY.idx(i, j);
                        dl.dual = diff1(args.cY.idx(i, j), k, d);

                        cYk.idx(i, j) = dl;
                    }
                }

                inverse_metric<dual<valuef>, 3, 3> icYk = cYk.invert();

                for(int i=0; i < 3; i++)
                {
                    for(int j=0; j < 3; j++)
                    {
                        dicY.idx(k, i, j) = icYk.idx(i, j);
                    }
                }
            }

            for(int i=0; i < 3; i++)
            {
                valuef sum = 0;
                valuef sum2 = 0;

                for(int j=0; j < 3; j++)
                {
                    sum += icY.idx(i, j) * diff1(args.K, j, d);
                    sum2 += dicY.idx(j, i, j).dual;
                    //sum += diff1(ctx, littlekij.idx(i, j), j);
                }

                Yij_Kj.idx(i) = sum + args.K * (sum2 + calculated_cG[i]);
            }
        }
        #else
        for(int i=0; i < 3; i++)
        {
            valuef sum = 0;

            for(int j=0; j < 3; j++)
            {
                sum += icY[i, j] * diff1(args.K, j, d);
            }

            Yij_Kj[i] = sum;
        }
        #endif

        for(int i=0; i < 3; i++)
        {
            valuef s1 = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    s1 += 2 * args.gA * christoff2[i, j, k] * icAij[j, k];
                }
            }

            valuef s2 = 2 * args.gA * -(2.f/3.f) * Yij_Kj[i];

            valuef s3 = 0;

            for(int j=0; j < 3; j++)
            {
                s3 += icAij[i, j] * 2 * derivs.dW[j];
            }

            s3 = 2 * (-1.f/4.f) * args.gA / max(args.W, valuef(0.0001f)) * 6 * s3;

            valuef s4 = 0;

            for(int j=0; j < 3; j++)
            {
                s4 += icAij[i, j] * diff1(args.gA, j, d);
            }

            s4 = -2 * s4;

            valuef s5 = 0;

            for(int j=0; j < 3; j++)
            {
                s5 += args.gB[j] * diff1(args.cG[i], j, d);
            }

            valuef s6 = 0;

            for(int j=0; j < 3; j++)
            {
                s6 += -args.cG_undiff(derivs)[j] * derivs.dgB[j, i];
            }

            valuef s7 = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    s7 += icY[j, k] * diff2(args.gB[i], k, j, derivs.dgB[k, i], derivs.dgB[j, i], d);
                }
            }

            valuef s8 = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    //??
                    s8 += icY[i, j] * diff2(args.gB[k], k, j, derivs.dgB[k, k], derivs.dgB[j, k], d);
                }
            }

            s8 = (1.f/3.f) * s8;

            valuef s9 = 0;

            for(int k=0; k < 3; k++)
            {
                s9 += derivs.dgB[k, k];
            }

            s9 = (2.f/3.f) * s9 * args.cG_undiff(derivs)[i];

            dtcG[i] = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9;

            #ifdef PAPER_CGI_DAMP
            auto step = [](const valuef& in)
            {
                return ternary(in >= 0.f, value{1.f}, value{0.f});
            };

            valuef bkk = 0;

            for(int k=0; k < 3; k++)
            {
                bkk += derivs.dgB.idx(k, k);
            }

            float E = 1.f;

            valuef lambdai = (2.f/3.f) * (bkk - 2 * args.gA * args.K)
                            - derivs.dgB.idx(i, i)
                            - (2.f/5.f) * args.gA * raise_index(args.cA, icY, 1).idx(i, i);

            dtcG.idx(i) += -(1 + E) * step(lambdai) * lambdai * Gi.idx(i);
            #endif
        }

        valuef dmbm = 0;

        for(int m=0; m < 3; m++)
        {
            dmbm += derivs.dgB[m, m];
        }

        //#define STABILITY_SIGMA
        #ifdef STABILITY_SIGMA
        float lapse_cst = -0.1;

        dtcG += lapse_cst * args.gA * Gi;
        #endif // STABILITY_SIGMA

        #define STABILITY_SIGMA_2
        #ifdef STABILITY_SIGMA_2
        auto step = [](const valuef& in)
        {
            return ternary(in >= 0.f, value{1.f}, value{0.f});
        };

        ///https://arxiv.org/pdf/gr-qc/0209066
        //valuef s = sign(dmbm);
        valuef s = step(dmbm);

        value X = 0.9f;

        dtcG += -(X * s + 2.f/3.f) * Gi * dmbm;
        #endif
    }

    dtcG += -16 * M_PI * args.gA * icY.raise(Si);

    return dtcG;
}

valuef apply_evolution(const valuef& base, const valuef& dt, valuef timestep)
{
    return base + dt * timestep;
}

void make_momentum_constraint(cl::context ctx, const std::vector<plugin*>& plugins)
{
    auto func = [plugins](execution_context&,
                          bssn_args_mem<buffer<valuef>> in,
                          value_impl::builder::placeholder plugin_ph,
                          std::array<buffer_mut<momentum_t>, 3> momentum_constraint,
                          literal<v3i> ldim,
                          literal<valuef> scale,
                          literal<valuei> positions_length) {
        using namespace single_source;

        all_adm_args_mem plugin_data = make_arg_provider(plugins);
        plugin_ph.add(plugin_data);

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= positions_length.get(), [&] {
            return_e();
        });

        v3i pos = get_coordinate_including_boundary(lid, dim);
        pin(pos);

        valuei idx = pos.z() * dim.y() * dim.x() + pos.y() * dim.x() + pos.x();
        pin(idx);

        bssn_args args(pos, dim, in);

        derivative_data d;
        d.pos = pos;
        d.dim = dim;
        d.scale = scale.get();

        v3f Si_lower = plugin_data.adm_Si(args, d);
        pin(Si_lower);

        auto Mi = calculate_momentum_constraint(args, d, Si_lower);

        for(int i=0; i < 3; i++)
        {
            as_ref(momentum_constraint[i][idx]) = (momentum_t)Mi[i];
        }
    };

    cl::async_build_and_cache(ctx, [=]
    {
        return value_impl::make_function(func, "momentum_constraint");
    }, {"momentum_constraint"});
}

///https://arxiv.org/pdf/0709.3559 tested, appendix a.2
void make_bssn(cl::context ctx, const std::vector<plugin*>& plugins, const initial_params& cfg)
{
    auto bssn_function = [plugins, cfg]
            (execution_context&, bssn_args_mem<buffer<valuef>> base,
            bssn_args_mem<buffer<valuef>> in,
            bssn_args_mem<buffer_mut<valuef>> out,
            bssn_derivatives_mem<buffer<derivative_t>> derivatives,
            value_impl::builder::placeholder plugin_ph,
            std::array<buffer<momentum_t>, 3> momentum_constraint,
            literal<valuef> timestep,
            literal<valuef> scale,
            literal<valuef> total_elapsed,
            literal<v3i> idim,
            literal<valuei> positions_length) {
        using namespace single_source;

        all_adm_args_mem plugin_data = make_arg_provider(plugins);
        plugin_ph.add(plugin_data);

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = idim.get();

        if_e(lid >= positions_length.get(), []{
            return_e();
        });

        v3i pos = get_coordinate_including_boundary(lid, dim);
        pin(pos);

        bssn_args args(pos, dim, in);
        bssn_derivatives derivs(pos, dim, derivatives);

        derivative_data d;
        d.pos = pos;
        d.dim = dim;
        d.scale = scale.get();

        tensor<momentum_t, 3> Mi;

        for(int i=0; i < 3; i++)
        {
            Mi[i] = momentum_constraint[i][pos, dim];
        }

        valuef rho_s = plugin_data.adm_p(args, d);
        v3f Si = plugin_data.adm_Si(args, d);
        tensor<valuef, 3, 3> W2_Sij = plugin_data.adm_W2_Sij(args, d);

        valuef S = 0;

        {
            inverse_metric<valuef, 3, 3> icY = args.cY.invert();

            pin(icY);

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    S += icY[i, j] * W2_Sij[i, j];
                }
            }
        }


        tensor<valuef, 3, 3> dtcA = get_dtcA(args, derivs, Mi, d, W2_Sij);

        tensor<int, 2> index_table[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

        for(int i=0; i < 6; i++)
        {
            tensor<int, 2> idx = index_table[i];

            as_ref(out.cA[i][pos, dim]) = apply_evolution(base.cA[i][pos, dim], dtcA[idx.x(), idx.y()], timestep.get());
        }

        valuef dtW = get_dtW(args, derivs, d, rho_s, timestep.get());
        as_ref(out.W[pos, dim]) = apply_evolution(base.W[pos, dim], dtW, timestep.get());

        valuef dtK = get_dtK(args, derivs, d, S, rho_s);
        as_ref(out.K[pos, dim]) = apply_evolution(base.K[pos, dim], dtK, timestep.get());

        valuef dtgA = get_dtgA(args, derivs, d, total_elapsed.get(), cfg.lapse_damp_timescale);
        as_ref(out.gA[pos, dim]) = clamp(apply_evolution(base.gA[pos, dim], dtgA, timestep.get()), valuef(0.f), valuef(1.f));

        auto dtgB = get_dtgB(args, derivs, d, cfg);

        for(int i=0; i < 3; i++)
        {
            as_ref(out.gB[i][pos, dim]) = apply_evolution(base.gB[i][pos, dim], dtgB[i], timestep.get());
        }

        auto dtcY = get_dtcY(args, derivs, d, rho_s);

        for(int i=0; i < 6; i++)
        {
            tensor<int, 2> idx = index_table[i];

            as_ref(out.cY[i][pos, dim]) = apply_evolution(base.cY[i][pos, dim], dtcY[idx.x(), idx.y()], timestep.get());
        }

        auto dtcG = get_dtcG(args, derivs, d, Si);

        for(int i=0; i < 3; i++)
        {
            as_ref(out.cG[i][pos, dim]) = apply_evolution(base.cG[i][pos, dim], dtcG[i], timestep.get());
        }

        /*if_e(pos.x() == 2 && pos.y() == 128 && pos.z() == 128, [&]{
            value_base se;
            se.type = value_impl::op::SIDE_EFFECT;
            se.abstract_value = "printf(\"K: %f\\n\"," + value_to_string(out.W[pos, dim]) + ")";

            value_impl::get_context().add(se);
        });*/
    };

    cl::async_build_and_cache(ctx, [=]
    {
        return value_impl::make_function(bssn_function, "evolve");
    }, {"evolve"});
}

void enforce_algebraic_constraints(cl::context ctx)
{
    auto func = [](execution_context&, std::array<buffer_mut<valuef>, 6> mcY, std::array<buffer_mut<valuef>, 6> mcA, literal<valuei> positions_length, literal<v3i> idim)
    {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = {idim.get().x(), idim.get().y(), idim.get().z()};

        if_e(lid >= positions_length.get(), [&] {
            return_e();
        });

        v3i pos = get_coordinate_including_boundary(lid, dim, 1);
        pin(pos);

        metric<valuef, 3, 3> cY;
        tensor<valuef, 3, 3> cA;

        {
            int index_table[3][3] = {{0, 1, 2},
                                     {1, 3, 4},
                                     {2, 4, 5}};

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    int idx = index_table[i][j];

                    cY[i, j] = mcY[idx][pos, dim];
                    cA[i, j] = mcA[idx][pos, dim];
                }
            }
        }

        pin(cY);
        pin(cA);

        valuef det_cY = pow(cY.det(), 1.f/3.f);

        metric<valuef, 3, 3> fixed_cY = cY / det_cY;
        pin(fixed_cY);

        tensor<valuef, 3, 3> fixed_cA = trace_free(cA, fixed_cY, fixed_cY.invert());
        pin(fixed_cA);

        /*if_e(pos.x() == 128 && pos.y() == 128 && pos.z() == 128, [&]{
            value_base se;
            se.type = value_impl::op::SIDE_EFFECT;
            se.abstract_value = "printf(\"Trace: %.23f\\n\"," + value_to_string(trace(fixed_cA, fixed_cY.invert())) + ")";
            //se.abstract_value = "printf(\"Det: %.23f\\n\"," + value_to_string(fixed_cY.det()) + ")";

            value_impl::get_context().add(se);
        });*/

        tensor<int, 2> index_table[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

        for(int i=0; i < 6; i++)
        {
            tensor<int, 2> idx = index_table[i];

            as_ref(mcY[i][pos, dim]) = fixed_cY[idx.x(), idx.y()];
            as_ref(mcA[i][pos, dim]) = fixed_cA[idx.x(), idx.y()];
        }
    };

    cl::async_build_and_cache(ctx, [=] {
        return value_impl::make_function(func, "enforce_algebraic_constraints");
    }, {"enforce_algebraic_constraints"});
}

///todo: I think the hamiltonian constraint I'm using is expecting ADM, but we're using BSSN

void init_debugging(cl::context ctx, const std::vector<plugin*>& plugins)
{
    auto func = [plugins](execution_context&, bssn_args_mem<buffer<valuef>> in, bssn_derivatives_mem<buffer<derivative_t>> derivs_in, value_impl::builder::placeholder plugin_ph, literal<v3i> ldim, literal<valuef> scale, write_only_image<2> write) {
        using namespace single_source;

        all_adm_args_mem plugin_data = make_arg_provider(plugins);
        plugin_ph.add(plugin_data);

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        v3i pos = get_coordinate(lid, dim);

        /*if_e(pos.y() != valuei(dim.y()/2), [&] {
            return_e();
        });*/

        if_e(pos.z() != valuei(dim.z()/2), [&] {
            return_e();
        });

        bssn_args args(pos, dim, in);
        bssn_derivatives derivs(pos, dim, derivs_in);

        derivative_data d;
        d.pos = pos;
        d.dim = dim;
        d.scale = scale.get();

        //valuef rho_s = plugin_data.adm_p(args, d);
        //valuef ham = calculate_hamiltonian_constraint(args, derivs, d, rho_s);
        //valuef p = fabs(ham) * 1000;

        valuef momentum = calculate_momentum_constraint_summed(args, d, plugin_data.adm_Si(args, d));
        valuef p = fabs(momentum) * 10000;

        //valuef p = plugin_data.mem.adm_p(args, d);
        //valuef p = plugin_data.dbg(args, d);

        //valuef test_val = in.cY[0][lid];
        //valuef display = ((test_val - 1) / 0.1f) * 0.5f + 0.5f;

        //valuef display = fabs(in.gA[lid]);

        valuef display = p;

        v4f col = {display, display, display, 1.f};

        //col.x() = ternary(pos.x() == dim.x()/2 + 15, valuef(1.f), col.x());

        col = clamp(col, valuef(0.f), valuef(1.f));

        if_e(pos.x() >= 0 && pos.x() < dim.x() && pos.y() >= 0 && pos.y() < dim.y(), [&]()
        {
            write.write({pos.x(), pos.y()}, col);
        });


        #if 0
        if_e(pos.x() == 50 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
            using namespace single_source;

            auto icY = args.cY.invert();
            pin(icY);

            /*auto W2Rij = calculate_W2Rij(args, derivs, d);
            valuef R = trace(W2Rij, icY);*/

            auto christoff2 = christoffel_symbols_2(args.cY.invert(), derivs.dcY);
            auto fchristoff2 = get_full_christoffel2(args.W, derivs.dW, args.cY, args.cY.invert(), christoff2);

            auto W2Rij = args.W * args.W * calculate_adm_Rij(fchristoff2, d);

            valuef R = trace(W2Rij, icY);

            tensor<valuef, 3, 3> AMN = icY.raise(icY.raise(args.cA, 0), 1);

            valuef AMN_Amn = 0;

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    AMN_Amn += AMN[i, j] * args.cA[i, j];
                }
            }

            valuef cham = R + (2.f/3.f) * args.K * args.K - AMN_Amn - 16 * M_PI * rho_s;

            print("Ham %f\n", cham);

            print("Hd K %f R %.23f rho %f rho_all %f Asum %f", args.K, R, rho_s, -16 * M_PI * rho_s, AMN_Amn);

            write.write({pos.x(), pos.y()}, (v4f){1, 0, 0, 1});
        });
        #endif
    };

    cl::async_build_and_cache(ctx, [=] {
        return value_impl::make_function(func, "debug");
    }, {"debug"});
}

void make_sommerfeld(cl::context ctx)
{
    auto func = [](execution_context&, buffer<valuef> base, buffer<valuef> in, buffer_mut<valuef> out, literal<valuef> timestep,
                    literal<v3i> ldim,
                    literal<valuef> scale,
                    literal<valuef> wave_speed,
                    literal<valuef> asym,
                    buffer<v3i> positions,
                    literal<valuei> position_num) {

        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= position_num.get(), [&] {
            return_e();
        });

        v3i pos = positions[lid];

        pin(pos);

        v3f world_pos = grid_to_world((v3f)pos, dim, scale.get());

        valuef r = world_pos.length();

        auto sommerfeld = [&](single_source::buffer<valuef> f, const valuef& f0, const valuef& v)
        {
            valuef sum = 0;

            for(int i=0; i < 3; i++)
            {
                sum += world_pos[i] * diff1_boundary(f, i, scale.get(), pos, dim);
            }

            return (-sum - (f[pos, dim] - f0)) * (v/r);
        };

        valuef dt_boundary = sommerfeld(in, asym.get(), wave_speed.get());

        as_ref(out[pos, dim]) = apply_evolution(base[pos, dim], dt_boundary, timestep.get());

        /*if_e(pos.x() == 1 && pos.y() == 128 && pos.z() == 128, [&]{
            value_base se;
            se.type = value_impl::op::SIDE_EFFECT;
            se.abstract_value = "printf(\"%f\\n\"," + value_to_string(out[pos, dim]) + ")";

            value_impl::get_context().add(se);
        });*/
    };

    cl::async_build_and_cache(ctx, [=] {
        return value_impl::make_function(func, "sommerfeld");
    }, {"sommerfeld"});
}

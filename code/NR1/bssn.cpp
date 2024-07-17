#include "bssn.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include "tensor_algebra.hpp"
#include "derivatives.hpp"
#include <iostream>

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

tensor<valuef, 3> calculate_momentum_constraint(bssn_args& args, const valuef& scale)
{
    valuef X = args.W*args.W;

    tensor<valuef, 3> dW;

    for(int i=0; i < 3; i++)
        dW[i] = diff1(args.W, i, scale);

    tensor<valuef, 3> dX = 2 * args.W * dW;

    ///https://arxiv.org/pdf/1205.5111v1.pdf (54)
    tensor<valuef, 3, 3> aij_raised = raise_index(args.cA, args.cY.invert(), 1);

    tensor<valuef, 3> dPhi = -dX / (4 * max(X, valuef(0.0001f)));

    tensor<valuef, 3> Mi;

    for(int i=0; i < 3; i++)
    {
        valuef s1 = 0;

        for(int j=0; j < 3; j++)
        {
            s1 += diff1(aij_raised[i, j], j, scale);
        }

        valuef s2 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                s2 += -0.5f * args.cY.invert()[j, k] * diff1(args.cA[j, k], i, scale);
            }
        }

        valuef s3 = 0;

        for(int j=0; j < 3; j++)
        {
            s3 += 6 * dPhi[j] * aij_raised[i, j];
        }

        valuef p4 = -(2.f/3.f) * diff1(args.K, i, scale);

        Mi[i] = s1 + s2 + s3 + p4;
    }

    return Mi;
}

std::string make_derivatives()
{
    auto differentiate = [&](execution_context&, buffer<valuef> in, std::array<buffer_mut<derivative_t>, 3> out, literal<v3i> ldim, literal<valuef> scale)
    {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        v3i pos = get_coordinate(lid, dim);

        valuef v1 = in[pos, dim];

        as_ref(out[0][pos, dim]) = (derivative_t)diff1(v1, 0, scale.get());
        as_ref(out[1][pos, dim]) = (derivative_t)diff1(v1, 1, scale.get());
        as_ref(out[2][pos, dim]) = (derivative_t)diff1(v1, 2, scale.get());
    };

    std::string str = value_impl::make_function(differentiate, "differentiate");

    std::cout << str << std::endl;

    return str;
}

tensor<valuef, 3, 3> calculate_cRij(bssn_args& args, bssn_derivatives& derivs, const valuef& scale)
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
                    s1 += -0.5f * icY[l, m] * diff2(args.cY[i, j], m, l, derivs.dcY[m, i, j], derivs.dcY[l, i, j], scale);
                }
            }

            valuef s2 = 0;

            for(int k=0; k < 3; k++)
            {
                s2 += 0.5f * (args.cY[k, i] * diff1(args.cG[k], j, scale) + args.cY[k, j] * diff1(args.cG[k], i, scale));
            }

            valuef s3 = 0;

            for(int k=0; k < 3; k++)
            {
                s3 += 0.5f * args.cG[k] * (christoff1[i, j, k] + christoff1[j, i, k]);
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
tensor<valuef, 3, 3> calculate_W2Rij(bssn_args& args, bssn_derivatives& derivs, const valuef& scale)
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
            didjW[i, j] = double_covariant_derivative(args.W, derivs.dW, christoff2, scale)[j, i];
        }
    }

    tensor<valuef, 3, 3> w2Rphiij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            valuef v1 = args.W * didjW[i, j];
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

            w2Rphiij[i, j] = v1 + v2 + v3;
        }
    }

    pin(w2Rphiij);

    return w2Rphiij + calculate_cRij(args, derivs, scale) * args.W * args.W;
}

valuef calculate_hamiltonian_constraint(bssn_args& args, bssn_derivatives& derivs, const valuef& scale)
{
    using namespace single_source;

    auto W2Rij = calculate_W2Rij(args, derivs, scale);

    auto icY = args.cY.invert();
    pin(icY);

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

    return R + (2.f/3.f) * args.K * args.K - AMN_Amn;
}

//#define BLACK_HOLE_GAUGE
#ifdef BLACK_HOLE_GAUGE
#define ONE_PLUS_LOG
#define GAMMA_DRIVER
#endif // BLACK_HOLE_GAUGE

#define WAVE_TEST
#ifdef WAVE_TEST
#define HARMONIC_SLICING
#define ZERO_SHIFT
#endif // WAVE_TEST

//#define WAVE_TEST2
#ifdef WAVE_TEST2
#define ONE_LAPSE
#define ZERO_SHIFT
#endif

valuef get_dtgA(bssn_args& args, bssn_derivatives& derivs, valuef scale)
{
    valuef bmdma = 0;

    for(int i=0; i < 3; i++)
    {
        bmdma += args.gB[i] * diff1(args.gA, i, scale);
    }

    ///https://arxiv.org/pdf/gr-qc/0206072
    #ifdef ONE_PLUS_LOG
    return -2 * args.gA * args.K + bmdma;
    #endif // ONE_PLUS_LOG

    ///https://arxiv.org/pdf/2201.08857
    #ifdef HARMONIC_SLICING
    return -args.gA * args.gA * args.K + bmdma;
    #endif // HARMONIC_SLICING

    #ifdef ONE_LAPSE
    return 0;
    #endif // ONE_LAPSE
}

tensor<valuef, 3> get_dtgB(bssn_args& args, bssn_derivatives& derivs, valuef scale)
{
    ///https://arxiv.org/pdf/gr-qc/0605030 (26)
    #ifdef GAMMA_DRIVER
    tensor<valuef, 3> djbjbi;

    for(int i=0; i < 3; i++)
    {
        valuef sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum += args.gB[j] * diff1(args.gB[i], j, scale);
        }

        djbjbi[i] = sum;
    }

    ///gauge damping parameter, commonly set to 2
    float N = 2;

    return (3/4.f) * args.cG + djbjbi - N * args.gB;
    #endif // GAMMA_DRIVER

    #ifdef ZERO_SHIFT
    return {0,0,0};
    #endif // ZERO_SHIFT
}

tensor<valuef, 3, 3> get_dtcY(bssn_args& args, bssn_derivatives& derivs, valuef scale)
{
    using namespace single_source;

    inverse_metric<valuef, 3, 3> icY = args.cY.invert();
    pin(icY);

    ///https://arxiv.org/pdf/1307.7391 specifically for why the trace free aspect
    ///https://arxiv.org/pdf/1106.2254 also see here, after 25
    return lie_derivative_weight(args.gB, args.cY.to_tensor(), scale) - 2 * args.gA * trace_free(args.cA, args.cY, icY);
}

///https://iopscience.iop.org/article/10.1088/1361-6382/ac7e16/pdf 2.12 or
///https://arxiv.org/pdf/0709.2160
valuef get_dtW(bssn_args& args, bssn_derivatives& derivs, const valuef& scale)
{
    valuef dibi = 0;

    for(int i=0; i < 3; i++)
    {
        dibi += diff1(args.gB[i], i, scale);
    }

    valuef dibiw = 0;

    for(int i=0; i < 3; i++)
    {
        dibiw += args.gB[i] * diff1(args.W, i, scale);
    }

    return (1/3.f) * args.W * (args.gA * args.K - dibi) + dibiw;
}

tensor<valuef, 3, 3> calculate_W2DiDja(bssn_args& args, bssn_derivatives& derivs, const valuef& scale)
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
            valuef v1 = args.W * args.W * double_covariant_derivative(args.gA, derivs.dgA, christoff2, scale)[i, j];

            valuef v2 = args.W * (derivs.dW[i] * diff1(args.gA, j, scale) + derivs.dW[j] * diff1(args.gA, i, scale));

            valuef sum = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    sum += icY[m, n] * 2 * args.W * derivs.dW[m] * diff1(args.gA, n, scale);
                }
            }

            valuef v3 = -0.5f * args.cY[i, j] * sum;

            W2DiDja[i, j] = v1 + v2 + v3;
        }
    }

    return W2DiDja;
}

valuef get_dtK(bssn_args& args, bssn_derivatives& derivs, const valuef& scale)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    pin(icY);

    tensor<valuef, 3, 3> W2DiDja = calculate_W2DiDja(args, derivs, scale);
    pin(W2DiDja);

    tensor<valuef, 3, 3> AMN = icY.raise(icY.raise(args.cA, 0), 1);
    pin(AMN);

    valuef v1 = 0;

    for(int m=0; m < 3; m++)
    {
        v1 += args.gB[m] * diff1(args.K, m, scale);
    }

    return v1 - sum_multiply(icY.to_tensor(), W2DiDja)
              + args.gA * sum_multiply(AMN, args.cA)
              + (1/3.f) * args.gA * args.K * args.K;
}

tensor<valuef, 3, 3> get_dtcA(bssn_args& args, bssn_derivatives& derivs, v3f momentum_constraint, const valuef& scale)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    pin(icY);

    auto W2DiDja = calculate_W2DiDja(args, derivs, scale);
    pin(W2DiDja);

    tensor<valuef, 3, 3> with_trace = args.gA * calculate_W2Rij(args, derivs, scale) - W2DiDja;

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

    tensor<valuef, 3, 3> dtcA = lie_derivative_weight(args.gB, args.cA, scale)
                                + args.gA * args.K * args.cA
                                - 2 * args.gA * aij_amj
                                + trace_free(with_trace, args.cY, icY);

    #define MOMENTUM_CONSTRAINT_DAMPING
    #ifdef MOMENTUM_CONSTRAINT_DAMPING
    auto christoff2 = christoffel_symbols_2(icY, derivs.dcY);
    pin(christoff2);

    auto cd_low = covariant_derivative_low_vec(momentum_constraint, christoff2, scale);
    pin(cd_low);

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            float Ka = 0.1f;

            dtcA[i, j] += Ka * args.gA * 0.5f *
                              (cd_low[i, j]
                             + cd_low[j, i]);
        }
    }
    #endif

    return dtcA;
}

tensor<valuef, 3> get_dtcG(bssn_args& args, bssn_derivatives& derivs, const valuef& scale)
{
    using namespace single_source;

    inverse_metric<valuef, 3, 3> icY = args.cY.invert();
    pin(icY);

    tensor<valuef, 3, 3, 3> christoff2 = christoffel_symbols_2(icY, derivs.dcY);
    pin(christoff2);

    tensor<valuef, 3> dtcG;

    ///dtcG
    {
        tensor<valuef, 3, 3> icAij = icY.raise(icY.raise(args.cA, 0), 1);

        pin(icAij);

        tensor<valuef, 3> Yij_Kj;

        for(int i=0; i < 3; i++)
        {
            valuef sum = 0;

            for(int j=0; j < 3; j++)
            {
                sum += icY[i, j] * diff1(args.K, j, scale);
            }

            Yij_Kj[i] = sum;
        }

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
                s4 += icAij[i, j] * diff1(args.gA, j, scale);
            }

            s4 = -2 * s4;

            valuef s5 = 0;

            for(int j=0; j < 3; j++)
            {
                s5 += args.gB[j] * diff1(args.cG[i], j, scale);
            }

            valuef s6 = 0;

            for(int j=0; j < 3; j++)
            {
                s6 += -args.cG[j] * derivs.dgB[j, i];
            }

            valuef s7 = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    s7 += icY[j, k] * diff2(args.gB[i], k, j, derivs.dgB[k, i], derivs.dgB[j, i], scale);
                }
            }

            valuef s8 = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    //??
                    s8 += icY[i, j] * diff2(args.gB[k], k, j, derivs.dgB[k, k], derivs.dgB[j, k], scale);
                }
            }

            s8 = (1.f/3.f) * s8;

            valuef s9 = 0;

            for(int k=0; k < 3; k++)
            {
                s9 += derivs.dgB[k, k];
            }

            s9 = (2.f/3.f) * s9 * args.cG[i];

            dtcG[i] = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9;
        }

        //#define STABILITY_SIGMA
        #ifdef STABILITY_SIGMA
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

        tensor<valuef, 3> Gi = args.cG - calculated_cG;

        valuef dmbm = 0;

        for(int m=0; m < 3; m++)
        {
            dmbm += diff1(args.gB[m], m, scale);
        }

        float sigma = 1.333333f;

        dtcG += -sigma * Gi * dmbm;
        #endif // STABILITY_SIGMA
    }

    return dtcG;
}

valuef apply_evolution(const valuef& base, const valuef& dt, valuef timestep)
{
    return base + dt * timestep;
}

std::string make_momentum_constraint()
{
    auto cst = [&](execution_context&, bssn_args_mem<buffer<valuef>> in,
                                       std::array<buffer_mut<valuef>, 3> momentum_constraint,
                                       literal<v3i> ldim,
                                       literal<valuef> scale) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        v3i pos = get_coordinate(lid, dim);

        bssn_args args(pos, dim, in);

        auto Mi = calculate_momentum_constraint(args, scale.get());

        for(int i=0; i < 3; i++)
        {
            as_ref(momentum_constraint[i][lid]) = Mi[i];
        }
    };

    return value_impl::make_function(cst, "momentum_constraint");
}

///https://arxiv.org/pdf/0709.3559 tested, appendix a.2
std::string make_bssn(const tensor<int, 3>& idim)
{
    auto bssn_function = [&](execution_context&, bssn_args_mem<buffer<valuef>> base,
                                                 bssn_args_mem<buffer<valuef>> in,
                                                 bssn_args_mem<buffer_mut<valuef>> out,
                                                 bssn_derivatives_mem<buffer<derivative_t>> derivatives,
                                                 std::array<buffer<valuef>, 3> momentum_constraint,
                                                 literal<valuef> timestep,
                                                 literal<v3i> ldim,
                                                 literal<valuef> scale) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = {idim.x(), idim.y(), idim.z()};

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        v3i pos = get_coordinate(lid, dim);

        bssn_args args(pos, dim, in);
        bssn_derivatives derivs(pos, dim, derivatives);

        tensor<valuef, 3> Mi;

        for(int i=0; i < 3; i++)
        {
            Mi[i] = momentum_constraint[i][pos, dim];
        }

        tensor<valuef, 3, 3> dtcA = get_dtcA(args, derivs, Mi, scale.get());

        tensor<int, 2> index_table[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

        for(int i=0; i < 6; i++)
        {
            tensor<int, 2> idx = index_table[i];

            as_ref(out.cA[i][lid]) = apply_evolution(base.cA[i][lid], dtcA[idx.x(), idx.y()], timestep.get());
        }

        valuef dtW = get_dtW(args, derivs, scale.get());
        as_ref(out.W[lid]) = apply_evolution(base.W[lid], dtW, timestep.get());

        valuef dtK = get_dtK(args, derivs, scale.get());
        as_ref(out.K[lid]) = apply_evolution(base.K[lid], dtK, timestep.get());

        valuef dtgA = get_dtgA(args, derivs, scale.get());
        as_ref(out.gA[lid]) = apply_evolution(base.gA[lid], dtgA, timestep.get());

        auto dtgB = get_dtgB(args, derivs, scale.get());

        for(int i=0; i < 3; i++)
        {
            as_ref(out.gB[i][lid]) = apply_evolution(base.gB[i][lid], dtgB[i], timestep.get());
        }

        auto dtcY = get_dtcY(args, derivs, scale.get());

        for(int i=0; i < 6; i++)
        {
            tensor<int, 2> idx = index_table[i];

            as_ref(out.cY[i][lid]) = apply_evolution(base.cY[i][lid], dtcY[idx.x(), idx.y()], timestep.get());
        }

        auto dtcG = get_dtcG(args, derivs, scale.get());

        for(int i=0; i < 3; i++)
        {
            as_ref(out.cG[i][lid]) = apply_evolution(base.cG[i][lid], dtcG[i], timestep.get());
        }

    };

    std::string str = value_impl::make_function(bssn_function, "evolve");

    std::cout << str << std::endl;

    return str;
}

std::string enforce_algebraic_constraints()
{
    auto func = [&](execution_context&, std::array<buffer_mut<valuef>, 6> mcY, std::array<buffer_mut<valuef>, 6> mcA, literal<v3i> idim)
    {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = {idim.get().x(), idim.get().y(), idim.get().z()};

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        v3i pos = get_coordinate(lid, dim);

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

                    cY[i, j] = mcY[idx][lid];
                    cA[i, j] = mcA[idx][lid];
                }
            }
        }

        valuef det_cY_pow = pow(cY.det(), 1.f/3.f);

        metric<valuef, 3, 3> fixed_cY = cY / det_cY_pow;
        tensor<valuef, 3, 3> fixed_cA = trace_free(cA, fixed_cY, fixed_cY.invert());

        tensor<int, 2> index_table[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

        for(int i=0; i < 6; i++)
        {
            tensor<int, 2> idx = index_table[i];

            as_ref(mcY[i][lid]) = fixed_cY[idx.x(), idx.y()];
            as_ref(mcA[i][lid]) = fixed_cA[idx.x(), idx.y()];
        }
    };

    return value_impl::make_function(func, "enforce_algebraic_constraints");
}

std::string init_debugging()
{
    auto dbg = [&](execution_context&, bssn_args_mem<buffer_mut<valuef>> to_fill, literal<v3i> ldim, literal<valuef> scale, write_only_image<2> write) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        v3i pos = get_coordinate(lid, dim);

        if_e(pos.z() != valuei(dim.z()/2), [&] {
            return_e();
        });

        valuef test_val = to_fill.cY[0][lid];
        valuef display = ((test_val - 1) / 0.1f) * 0.5f + 0.5f;

        v4f col = {display, 0.f, 0.f, 1.f};

        col = clamp(col, valuef(0.f), valuef(1.f));

        write.write({pos.x(), pos.y()}, col);
    };

    return value_impl::make_function(dbg, "debug");
}

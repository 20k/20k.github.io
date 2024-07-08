#include "bssn.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include "../common/vec/dual.hpp"
#include <iostream>

template<typename T>
using dual = dual_types::dual_v<T>;

value_base positive_mod(const value_base& val, const value_base& n)
{
    return ((val % n) + n) % n;
}

template<typename T, int elements = 5>
struct differentiation_context
{
    std::array<T, elements> vars;

    differentiation_context(const T& in, int direction)
    {
        std::array<int, elements> offx = {};
        std::array<int, elements> offy = {};
        std::array<int, elements> offz = {};

        for(int i=0; i < elements; i++)
        {
            int offset = i - (elements - 1)/2;

            if(direction == 0)
                offx[i] = offset;
            if(direction == 1)
                offy[i] = offset;
            if(direction == 2)
                offz[i] = offset;
        }

        ///for each element, ie x-2, x-1, x, x+1, x+2
        for(int i=0; i < elements; i++)
        {
            ///assign to the original element, ie x
            vars[i] = in;

            //std::cout << "Pre " << value_to_string(in) << std::endl;

            vars[i].recurse([&i, &offx, &offy, &offz](value_base& v)
            {
                if(v.type == value_impl::op::BRACKET)
                {
                    auto get_substitution = [&i, &offx, &offy, &offz](const value_base& v)
                    {
                        assert(v.args.size() == 8);

                        auto buf = v.args[0];

                        value_base old_x = v.args[2];
                        value_base old_y = v.args[3];
                        value_base old_z = v.args[4];

                        value_base dx = v.args[5];
                        value_base dy = v.args[6];
                        value_base dz = v.args[7];

                        value_base next_x = old_x + valuei(offx[i]);
                        value_base next_y = old_y + valuei(offy[i]);
                        value_base next_z = old_z + valuei(offz[i]);

                        //std::cout << "Old " << value_to_string(old_x) << " dx " << value_to_string(dx) << std::endl;

                        #define PERIODIC_BOUNDARY
                        #ifdef PERIODIC_BOUNDARY
                        next_x = positive_mod(next_x, dx);
                        next_y = positive_mod(next_y, dy);
                        next_z = positive_mod(next_z, dz);
                        #endif // PERIODIC_BOUNDARY

                        value_base op;
                        op.type = value_impl::op::BRACKET;
                        op.args = {buf, value<int>(3), next_x, next_y, next_z, dx, dy, dz};
                        op.concrete = get_interior_type(T());

                        return op;
                    };

                    v = get_substitution(v);
                }
            });

            //std::cout << "post " << value_to_string(vars[i]) << std::endl;
        }
    }
};

valuef diff1(const valuef& val, int direction, valuef scale)
{
    ///second order derivatives
    differentiation_context dctx(val, direction);
    std::array<valuef, 5> vars = dctx.vars;

    valuef p1 = -vars[4] + vars[0];
    valuef p2 = valuef(8.f) * (vars[3] - vars[1]);

    return (p1 + p2) / (12.f * scale);
}

///this uses the commutativity of partial derivatives to lopsidedly prefer differentiating dy in the x direction
///as this is better on the memory layout
valuef diff2(const valuef& in, int idx, int idy, const valuef& dx, const valuef& dy, const valuef& scale)
{
    if(idx < idy)
    {
        return diff1(dy, idx, scale);
    }
    else
    {
        return diff1(dx, idy, scale);
    }
}


template<typename T, int N, typename S>
inline
tensor<T, N, N> lie_derivative_weight(const tensor<T, N>& B, const S& mT, const T& scale)
{
    tensor<T, N, N> lie;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            T sum = 0;
            T sum2 = 0;

            for(int k=0; k < N; k++)
            {
                sum += B[k] * diff1(mT[i, j], k, scale);
                sum += mT[i, k] * diff1(B[k], j, scale);
                sum += mT[j, k] * diff1(B[k], i, scale);
                sum2 += diff1(B[k], k, scale);
            }

            lie.idx(i, j) = sum - (2.f/3.f) * mT.idx(i, j) * sum2;
        }
    }

    return lie;
}

///derivatives are passed in like dkMij
///https://en.wikipedia.org/wiki/Christoffel_symbols#Christoffel_symbols_of_the_second_kind_(symmetric_definition)
///the christoffel symbols are calculated normally in bssn with the conformal metric tensor
template<typename T, int N>
inline
tensor<T, N, N, N> christoffel_symbols_2(const inverse_metric<T, N, N>& inverse, const tensor<T, N, N, N>& derivatives)
{
    tensor<T, N, N, N> christoff;

    for(int i=0; i < N; i++)
    {
        for(int k=0; k < N; k++)
        {
            for(int l=0; l < N; l++)
            {
                T sum = 0;

                for(int m=0; m < N; m++)
                {
                    T local = 0;

                    local += derivatives[l, m, k];
                    local += derivatives[k, m, l];
                    local += -derivatives[m, k, l];

                    sum += local * inverse[i, m];
                }

                christoff[i, k, l] = T(0.5f) * sum;
            }
        }
    }

    return christoff;
}

///todo: why is this like this? make it differentiable like cs2
template<typename T, int N>
inline
tensor<T, N, N, N> christoffel_symbols_1(const metric<T, N, N>& met, const valuef& scale)
{
    tensor<T, N, N, N> christoff;

    for(int c=0; c < N; c++)
    {
        for(int a=0; a < N; a++)
        {
            for(int b=0; b < N; b++)
            {
                christoff[c, a, b] = 0.5f * (diff1(met[c, a], b, scale) + diff1(met[c, b], a, scale) - diff1(met[a, b], c, scale));
            }
        }
    }

    return christoff;
}

///https://en.wikipedia.org/wiki/Covariant_derivative#Covariant_derivative_by_field_type
///a partial derivative is a lower index vector
template<typename T, int N>
inline
tensor<T, N, N> double_covariant_derivative(const T& in, const tensor<T, N>& first_derivatives,
                                            const tensor<T, N, N, N>& christoff2, const valuef& scale)
{
    tensor<T, N, N> lac;

    for(int a=0; a < N; a++)
    {
        for(int c=0; c < N; c++)
        {
            T sum = 0;

            for(int b=0; b < N; b++)
            {
                sum += christoff2[b, c, a] * diff1(in, b, scale);
            }

            lac.idx(a, c) = diff2(in, a, c, first_derivatives[a], first_derivatives[c], scale) - sum;
        }
    }

    return lac;
}

///https://arxiv.org/pdf/gr-qc/9810065.pdf
template<typename T, int N>
inline
T trace(const tensor<T, N, N>& mT, const inverse_metric<T, N, N>& inverse)
{
    T ret = 0;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            ret = ret + inverse.idx(i, j) * mT.idx(i, j);
        }
    }

    return ret;
}

template<typename T, int N>
inline
tensor<T, N, N> trace_free(const tensor<T, N, N>& mT, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    tensor<T, N, N> TF;
    T t = trace(mT, inverse);

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            TF.idx(i, j) = mT.idx(i, j) - (1/3.f) * met.idx(i, j) * t;
        }
    }

    return TF;
}
///https://en.wikipedia.org/wiki/Covariant_derivative#Covariant_derivative_by_field_type
///for the tensor DcDa, this returns idx(a, c)
template<typename T, int N>
inline
tensor<T, N, N> covariant_derivative_low_vec(const tensor<T, N>& v_in, const tensor<T, N, N, N>& christoff2, const valuef& scale)
{
    tensor<T, N, N> lac;

    for(int a=0; a < N; a++)
    {
        for(int c=0; c < N; c++)
        {
            T sum = 0;

            for(int b=0; b < N; b++)
            {
                sum += christoff2[b, c, a] * v_in[b];
            }

            lac[a, c] = diff1(v_in[a], c, scale) - sum;
        }
    }

    return lac;
}

template<typename T, int N>
inline
tensor<T, N, N> covariant_derivative_high_vec(const tensor<T, N>& v_in, const tensor<T, N, N>& derivatives, const tensor<T, N, N, N>& christoff2)
{
    tensor<T, N, N> lab;

    for(int a=0; a < N; a++)
    {
        for(int b=0; b < N; b++)
        {
            T sum = 0;

            for(int c=0; c < N; c++)
            {
                sum = sum + christoff2.idx(a, b, c) * v_in.idx(c);
            }

            lab.idx(a, b) = derivatives[b, a] + sum;
        }
    }

    return lab;
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
    auto differentiate = [&](execution_context&, buffer<valuef> in, std::array<buffer_mut<valuef>, 3> out, literal<v3i> ldim, literal<valuef> scale)
    {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        valuei x = lid % dim.x();
        valuei y = (lid / dim.x()) % dim.y();
        valuei z = lid / (dim.x() * dim.y());

        pin(x);
        pin(y);
        pin(z);

        v3i pos = {x, y, z};

        valuef v1 = in[pos, dim];

        as_ref(out[0][pos, dim]) = diff1(v1, 0, scale.get());
        as_ref(out[1][pos, dim]) = diff1(v1, 1, scale.get());
        as_ref(out[2][pos, dim]) = diff1(v1, 2, scale.get());
    };

    std::string str = value_impl::make_function(differentiate, "differentiate");

    std::cout << str << std::endl;

    return str;
}

struct time_derivatives
{
    tensor<valuef, 3, 3> dtcY;
    tensor<valuef, 3, 3> dtcA;
    valuef dtK;
    valuef dtW;
    tensor<valuef, 3> dtcG;

    valuef dtgA;
    tensor<valuef, 3> dtgB;
};

tensor<valuef, 3, 3> calculate_cRij(bssn_args& args, bssn_derivatives& derivs, const valuef& scale)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    pin(icY);

    auto christoff1 = christoffel_symbols_1(args.cY, scale);
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
tensor<valuef, 3, 3> calculate_W2_mult_Rij(bssn_args& args, bssn_derivatives& derivs, const valuef& scale)
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

    auto W2Rij = calculate_W2_mult_Rij(args, derivs, scale);

    auto icY = args.cY.invert();
    pin(icY);

    auto iYij = args.W * args.W * icY;

    value iW = 1/max(args.W, valuef(0.00001f));

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

float get_algebraic_damping_factor()
{
    return 1.f;
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

    tensor<valuef, 3, 3, 3> christoff2 = christoffel_symbols_2(icY, derivs.dcY);

    pin(christoff2);

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

    tensor<valuef, 3, 3> dtcY;

    ///dtcY
    {
        ///https://arxiv.org/pdf/1307.7391 specifically for why the trace free aspect
        ///https://arxiv.org/pdf/1106.2254 also see here, after 25
        dtcY = lie_derivative_weight(args.gB, args.cY, scale) - 2 * args.gA * trace_free(args.cA, args.cY, icY);

        ///https://arxiv.org/pdf/gr-qc/0204002
        dtcY += -get_algebraic_damping_factor() * args.gA * args.cY.to_tensor() * log(args.cY.det());

        dtcY += 0.00001f * args.gA * args.cY.to_tensor() * -calculate_hamiltonian_constraint(args, derivs, scale);

        /*tensor<valuef, 3, 3> cD = covariant_derivative_low_vec(args.cY.lower(Gi), christoff2, scale);

        pin(cD);

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                float cK = -0.035;

                ret.dtcY.idx(i, j) += cK * args.gA * 0.5f * (cD.idx(i, j) + cD.idx(j, i));
            }
        }*/

        #if 0
        tensor<valuef, 3, 3> d_cGi;

        for(int m=0; m < 3; m++)
        {
            tensor<dual<valuef>, 3, 3, 3> d_dcYij;

            metric<dual<valuef>, 3, 3> d_cYij;

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
                        d_dcYij[k, i, j].dual = diff1(derivs.dcY[k, i, j], m, scale);
                    }
                }
            }

            pin(d_dcYij);

            auto d_christoff2 = christoffel_symbols_2(dicY, d_dcYij);

            pin(d_christoff2);

            tensor<dual<valuef>, 3> dcGi_G;

            for(int i=0; i < 3; i++)
            {
                dual<valuef> sum = 0;

                for(int j=0; j < 3; j++)
                {
                    for(int k=0; k < 3; k++)
                    {
                        sum += dicY[j, k] * d_christoff2[i, j, k];
                    }
                }

                dcGi_G[i] = sum;
            }

            pin(dcGi_G);

            for(int i=0; i < 3; i++)
            {
                d_cGi[m, i] = diff1(args.cG[i], m, scale) - dcGi_G[i].dual;
            }
        }

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

                float cK = -0.055f;

                dtcY.idx(i, j) += cK * args.gA * sum;
            }
        }
        #endif
    }

    return dtcY;
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

tensor<valuef, 3, 3> calculate_DiDja(bssn_args& args, bssn_derivatives& derivs, const valuef& scale)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    pin(icY);

    auto christoff2 = christoffel_symbols_2(icY, derivs.dcY);

    pin(christoff2);

    ///W^2 = X
    valuef X = args.W * args.W;
    ///2 dW W = dX
    tensor<valuef, 3> dX = 2 * args.W * derivs.dW;

    value iX = 1/max(X, valuef(0.00001f));

    tensor<valuef, 3, 3> DiDja;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            valuef v1 = double_covariant_derivative(args.gA, derivs.dgA, christoff2, scale)[i, j];

            valuef v2 = 0.5f * iX * (dX[i] * diff1(args.gA, j, scale) + dX[j] * diff1(args.gA, i, scale));

            valuef sum = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    sum += icY[m, n] * dX[m] * diff1(args.gA, n, scale);
                }
            }

            valuef v3 = -0.5f * iX * args.cY[i, j] * sum;

            DiDja[i, j] = v1 + v2 + v3;
        }
    }

    return DiDja;
}

valuef get_dtK(bssn_args& args, bssn_derivatives& derivs, v3f momentum_constraint, const valuef& scale)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    pin(icY);

    valuef X = args.W * args.W;

    tensor<valuef, 3, 3> DiDja = calculate_DiDja(args, derivs, scale);

    pin(DiDja);

    valuef v1 = 0;

    for(int m=0; m < 3; m++)
    {
        v1 += args.gB[m] * diff1(args.K, m, scale);
    }

    valuef v2 = 0;

    {
        valuef sum = 0;

        for(int m=0; m < 3; m++)
        {
            for(int n=0; n < 3; n++)
            {
                sum += icY[m, n] * DiDja[m, n];
            }
        }

        v2 = -X * sum;
    }

    valuef v3 = 0;

    {
        valuef sum = 0;

        tensor<valuef, 3, 3> AMN = icY.raise(icY.raise(args.cA, 0), 1);

        pin(AMN);

        for(int m=0; m < 3; m++)
        {
            for(int n=0; n < 3; n++)
            {
                sum += AMN[m, n] * args.cA[m, n];
            }
        }

        v3 += args.gA * sum;
    }

    valuef v4 = (1/3.f) * args.gA * args.K * args.K;

    auto christoff2 = christoffel_symbols_2(icY, derivs.dcY);
    pin(christoff2);

    tensor<valuef, 3, 3> MkDj = covariant_derivative_low_vec(momentum_constraint, christoff2, scale);

    pin(MkDj);

    valuef sum = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            sum += icY[i, j] * MkDj[j, i];
        }
    }

    return v1 + v2 + v3 + v4;
}

tensor<valuef, 3, 3> get_dtcA(bssn_args& args, bssn_derivatives& derivs, v3f momentum_constraint, const valuef& scale)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    pin(icY);

    valuef X = args.W * args.W;

    auto DiDja = calculate_DiDja(args, derivs, scale);
    pin(DiDja);

    tensor<valuef, 3, 3> with_trace = args.gA * calculate_W2_mult_Rij(args, derivs, scale) - X * DiDja;

    tensor<valuef, 3, 3> dtcA;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            valuef v1 = lie_derivative_weight(args.gB, args.cA, scale)[i, j];

            valuef v2 = args.gA * args.K * args.cA[i, j];

            valuef v3 = 0;

            {
                valuef sum = 0;

                tensor<valuef, 3, 3> raised_Aij = icY.raise(args.cA, 0);

                for(int m=0; m < 3; m++)
                {
                    sum += args.cA[i, m] * raised_Aij[m, j];
                }

                v3 = -2 * args.gA * sum;
            }

            valuef v4 = trace_free(with_trace, args.cY, icY)[i, j];

            dtcA[i, j] = v1 + v2 + v3 + v4;
        }
    }

    dtcA += -get_algebraic_damping_factor() * args.gA * args.cY.to_tensor() * trace(args.cA, icY);

    //dtcA += -0.001f * args.gA * args.cA * calculate_hamiltonian_constraint(args, derivs, scale);

    //#define MOMENTUM_CONSTRAINT_DAMPING
    #ifdef MOMENTUM_CONSTRAINT_DAMPING
    auto christoff2 = christoffel_symbols_2(icY, derivs.dcY);
    pin(christoff2);

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            float Ka = 0.001f;

            dtcA[i, j] += Ka * args.gA * 0.5f *
                              (covariant_derivative_low_vec(momentum_constraint, christoff2, scale)[i, j]
                             + covariant_derivative_low_vec(momentum_constraint, christoff2, scale)[j, i]);
        }
    }
    #endif

    {
        auto christoff2 = christoffel_symbols_2(icY, derivs.dcY);
        pin(christoff2);

        tensor<valuef, 3, 3> dkmk;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                dkmk[i, j] = covariant_derivative_low_vec(momentum_constraint, christoff2, scale)[j, i];
            }
        }

        pin(dkmk);

        auto raised = icY.raise(dkmk, 0);

        valuef sum = 0;

        for(int k=0; k < 3; k++)
        {
            sum += raised[k, k];
        }

        //dtcA += 0.0002f * args.gA * args.cY.to_tensor() * sum;
    }

    {

        tensor<valuef, 3, 3> symmetric_momentum_deriv;

        {
            tensor<valuef, 3, 3> momentum_deriv;

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    momentum_deriv.idx(i, j) = diff1(momentum_constraint.idx(i), j, scale);
                }
            }

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    symmetric_momentum_deriv.idx(i, j) = 0.5f * (momentum_deriv.idx(i, j) + momentum_deriv.idx(j, i));
                }
            }

            pin(symmetric_momentum_deriv);
        }

        {
            valuef F_a = scale;

            //if(true)
            //    F_a = scale * args.gA;

                ///https://arxiv.org/pdf/1205.5111v1.pdf (56)
            //dtcA += 0.1f * F_a * trace_free(symmetric_momentum_deriv, args.cY, icY);
        }
    }

    return dtcA;
}

tensor<valuef, 3> get_dtcG(bssn_args& args, bssn_derivatives& derivs, const valuef& scale)
{
    using namespace single_source;

    inverse_metric<valuef, 3, 3> icY = args.cY.invert();
    pin(icY);

    tensor<valuef, 3, 3, 3> christoff2 = christoffel_symbols_2(icY, derivs.dcY);

    pin(christoff2);

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

    ///W^2 = X
    valuef X = args.W * args.W;
    ///2 dW W = dX
    tensor<valuef, 3> dX = 2 * args.W * derivs.dW;

    value iX = 1/max(X, valuef(0.00001f));

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
                    s7 += icY.idx(j, k) * diff2(args.gB[i], k, j, derivs.dgB[k, i], derivs.dgB[j, i], scale);
                }
            }

            valuef s8 = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
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
        valuef dmbm = 0;

        for(int m=0; m < 3; m++)
        {
            dmbm += diff1(args.gB[m], m, scale);
        }

        float sigma = 1.333333f;

        dtcG += -sigma * Gi * dmbm;

        /*{
            float mcGicst = -0.1f;

            ret.dtcG += mcGicst * args.gA * Gi;
        }*/

        /*if_e(args.pos.x() == 64 && args.pos.y() == 64 && args.pos.z() == 64, [&]{
            value_base se;
            se.type = value_impl::op::SIDE_EFFECT;
            se.abstract_value = "printf(\"w %.16f %.16f\\n\"," + value_to_string(Gi[0]) + "," + value_to_string(calculated_cG[0] - cG2[0]) + ")";

            value_impl::get_context().add(se);
        });*/
        #endif // STABILITY_SIGMA

        /*float mcGicst = -0.5f;

        dtcG += mcGicst * args.gA * Gi;*/
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

        valuei x = lid % dim.x();
        valuei y = (lid / dim.x()) % dim.y();
        valuei z = lid / (dim.x() * dim.y());

        pin(x);
        pin(y);
        pin(z);

        v3i pos = {x, y, z};

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
std::string make_bssn()
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

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        valuei x = lid % dim.x();
        valuei y = (lid / dim.x()) % dim.y();
        valuei z = lid / (dim.x() * dim.y());

        pin(x);
        pin(y);
        pin(z);

        v3i pos = {x, y, z};

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

        valuef dtK = get_dtK(args, derivs, Mi, scale.get());
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

std::string make_initial_conditions()
{
    auto init = [&](execution_context&, bssn_args_mem<buffer_mut<valuef>> to_fill, literal<v3i> ldim, literal<valuef> scale) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        valuei x = lid % dim.x();
        valuei y = (lid / dim.x()) % dim.y();
        valuei z = lid / (dim.x() * dim.y());

        pin(x);
        pin(y);
        pin(z);

        v3i pos = {x, y, z};

        v3i centre = dim / 2;

        v3f fpos = {x.to<float>(), y.to<float>(), z.to<float>()};
        v3f fcentre = {centre.x().to<float>(), centre.y().to<float>(), centre.z().to<float>()};

        v3f wpos = (fpos - fcentre) * scale.get();

        /*value_base se;
        se.type = value_impl::op::SIDE_EFFECT;
        se.abstract_value = "printf(\"w %.16f\\n\"," + value_to_string(wpos.x()) + ")";

        value_impl::get_context().add(se);*/

        #define GET_A 0.1f

        auto get_Guv = []<typename T>(const tensor<T, 4>& position)
        {
            float A = GET_A;
            float d = 1;

            auto H = A * sin(2 * std::numbers::pi_v<float> * (position.y() - position.x()) / d);

            metric<T, 4, 4> m;
            m[0, 0] = -1 * (1 - H);
            m[1, 1] = (1 - H);
            m[2, 2] = 1;
            m[3, 3] = 1;

            return m;
        };

        metric<valuef, 4, 4> Guv = get_Guv((v4f){0, wpos.x(), wpos.y(), wpos.z()});

        tensor<valuef, 4, 4, 4> dGuv;

        for(int k=0; k < 4; k++)
        {
            auto ldguv = diff_analytic(get_Guv, (v4f){0, wpos.x(), wpos.y(), wpos.z()}, k);

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

        for(int i=0; i < 6; i++)
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
        as_ref(to_fill.K[lid]) = K;
    };

    return value_impl::make_function(init, "init");
}

std::string init_christoffel()
{
     auto init = [&](execution_context&, bssn_args_mem<buffer_mut<valuef>> to_fill, literal<v3i> ldim, literal<valuef> scale) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        valuei x = lid % dim.x();
        valuei y = (lid / dim.x()) % dim.y();
        valuei z = lid / (dim.x() * dim.y());

        pin(x);
        pin(y);
        pin(z);

        v3i pos = {x, y, z};

        metric<valuef, 3, 3> cY;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                int index_table[3][3] = {{0, 1, 2},
                                         {1, 3, 4},
                                         {2, 4, 5}};

                cY[i, j] = to_fill.cY[index_table[i][j]][pos, dim];
            }
        }

        auto icY = cY.invert();

        tensor<valuef, 3, 3, 3> dcY;

        for(int k=0; k < 3; k++)
        {
            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    dcY[k, i, j] = diff1(cY[i, j], k, scale.get());
                }
            }
        }

        auto christoff2 = christoffel_symbols_2(icY, dcY);

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

        /*value_base se;
        se.type = value_impl::op::SIDE_EFFECT;
        se.abstract_value = "printf(\"%.16f\\n\"," + value_to_string(cG[0]) + ")";

        value_impl::get_context().add(se);*/

        for(int i=0; i < 3; i++)
        {
            as_ref(to_fill.cG[i][lid]) = calculated_cG[i];
        }
     };

     return value_impl::make_function(init, "init_christoffel");
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

        valuei x = lid % dim.x();
        valuei y = (lid / dim.x()) % dim.y();
        valuei z = lid / (dim.x() * dim.y());

        pin(x);
        pin(y);
        pin(z);

        v3i pos = {x, y, z};

        if_e(z != valuei(dim.z()/2), [&] {
            return_e();
        });

        /*value_base se;
        se.type = value_impl::op::SIDE_EFFECT;
        se.abstract_value = "printf(\"%.16f\\n\"," + value_to_string(to_fill.cY[3][lid]) + ")";

        value_impl::get_context().add(se);*/

        valuef test_val = to_fill.cY[0][lid];
        valuef display = ((test_val - 1) / GET_A) * 0.5f + 0.5f;

        //valuef test_val = to_fill.W[lid];

        /*value_base se;
        se.type = value_impl::op::SIDE_EFFECT;
        se.abstract_value = "printf(\"w %.16f %i\\n\"," + value_to_string(test_val) + "," + value_to_string(x) + ")";

        value_impl::get_context().add(se);*/

        //valuef display = clamp(test_val, 0.f,1.f);

        v4f col = {display, 0.f, 0.f, 1.f};

        col = clamp(col, valuef(0.f), valuef(1.f));

        write.write({pos.x(), pos.y()}, col);
    };

    return value_impl::make_function(dbg, "debug");
}

valuef diff6th(valuef in, int idx, valuef scale)
{
    differentiation_context<valuef, 7> dctx(in, idx);
    auto vars = dctx.vars;

    valuef p1 = vars[0] + vars[6];
    valuef p2 = -6 * (vars[1] + vars[5]);
    valuef p3 = 15 * (vars[2] + vars[4]);
    valuef p4 = -20 * vars[3];

    return (p1 + p2 + p3 + p4);
}


valuef diffnth(const valuef& in, int idx, int nth, const valuef& scale)
{
    ///1 with accuracy 2
    if(nth == 1)
    {
        assert(false);

        differentiation_context<valuef, 3> dctx(in, idx);
        auto vars = dctx.vars;

        return (vars[2] - vars[0]) / 2;
    }

    ///2 with accuracy 2
    if(nth == 2)
    {
        differentiation_context<valuef, 3> dctx(in, idx);
        auto vars = dctx.vars;

        value p1 = vars[0] + vars[2];
        value p2 = -2 * vars[1];

        return (p1 + p2);
    }


    ///4 with accuracy 2
    if(nth == 4)
    {
        differentiation_context<valuef, 5> dctx(in, idx);
        auto vars = dctx.vars;

        value p1 = vars[0] + vars[4];
        value p2 = -4 * (vars[1] + vars[3]);
        value p3 = 6 * vars[2];

        return (p1 + p2 + p3);
    }

    ///6 with accuracy 2
    if(nth == 6)
    {
        differentiation_context<valuef, 7> dctx(in, idx);
        auto vars = dctx.vars;

        value p1 = vars[0] + vars[6];
        value p2 = -6 * (vars[1] + vars[5]);
        value p3 = 15 * (vars[2] + vars[4]);
        value p4 = -20 * vars[3];

        return (p1 + p2 + p3 + p4);
    }

    assert(false);
}

valuef kreiss_oliger_interior(valuef in, valuef scale)
{
    valuef val = 0;

    int n = 2;

    for(int i=0; i < 3; i++)
    {
        val += diffnth(in, i, n, scale);
    }

    float p = n - 1;

    int sign = pow(-1, (p + 3)/2);

    int divisor = pow(2, p+1);

    float prefix = (float)sign / divisor;

    return (prefix / scale) * val;
}

std::string make_kreiss_oliger()
{
     auto func = [&](execution_context&, buffer<valuef> in, buffer_mut<valuef> inout, literal<valuef> timestep, literal<v3i> ldim, literal<valuef> scale, literal<valuef> eps) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        valuei x = lid % dim.x();
        valuei y = (lid / dim.x()) % dim.y();
        valuei z = lid / (dim.x() * dim.y());

        pin(x);
        pin(y);
        pin(z);

        v3i pos = {x, y, z};

        as_ref(inout[lid]) = declare_e(in[lid]) + eps.get() * timestep.get() * kreiss_oliger_interior(in[pos, dim], scale.get());
     };

     return value_impl::make_function(func, "kreiss_oliger");
}

#include "bssn.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include <iostream>

using valuef = value<float>;
using valuei = value<int>;

using v2f = tensor<valuef, 2>;
using v3f = tensor<valuef, 3>;
using v4f = tensor<valuef, 4>;
using v2i = tensor<valuei, 2>;
using v3i = tensor<valuei, 3>;
using v4i = tensor<valuei, 4>;
using m44f = metric<valuef, 4, 4>;

using mut_v4f = tensor<mut<valuef>, 4>;
using mut_v3f = tensor<mut<valuef>, 3>;

using derivative_t = valuef;

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

            #if 0
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
            #endif
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

///thoughts: its a lot easier to get my hands on equations with X
///but W^2 is clearly a better choice
///I think there are two options:
///1. Implement the X formalism, in terms of the W formalism
///2. Implement the cBSSN W formalism
///3. Put in more legwork and find a good W reference when I'm more awake
template<typename T>
struct bssn_args_mem : value_impl::single_source::argument_pack
{
    std::array<T, 6> cY;
    std::array<T, 6> cA;
    T K;
    T W;
    std::array<T, 3> cG;

    T gA;
    std::array<T, 3> gB;

    void build(auto& in)
    {
        using namespace value_impl::builder;

        add(cY, in);
        add(cA, in);
        add(K, in);
        add(W, in);
        add(cG, in);

        add(gA, in);
        add(gB, in);
    }
};

template<typename T>
struct bssn_derivatives_mem : value_impl::single_source::argument_pack
{
    ///todo: swapsies?
    std::array<std::array<T, 3>, 6> dcY;
    std::array<T, 3> dgA;
    std::array<std::array<T, 3>, 3> dgB;
    std::array<T, 3> dW;

    void build(auto& in)
    {
        using namespace value_impl::builder;

        add(dcY, in);
        add(dgA, in);
        add(dgB, in);
        add(dW, in);
    }
};

struct bssn_args
{
    metric<valuef, 3, 3> cY;
    tensor<valuef, 3, 3> cA;
    valuef K;
    valuef W;
    tensor<valuef, 3> cG;

    valuef gA;
    tensor<valuef, 3> gB;

    ///diYjk
    tensor<derivative_t, 3, 3, 3> dcY;
    tensor<derivative_t, 3> dgA;
    ///digBj
    tensor<derivative_t, 3, 3> dgB;
    tensor<derivative_t, 3> dW;

    bssn_args(v3i pos, v3i dim,
              bssn_args_mem<buffer<valuef>>& in, bssn_derivatives_mem<buffer<derivative_t>>& derivatives)
    {
        int index_table[3][3] = {{0, 1, 2},
                                 {1, 3, 4},
                                 {2, 4, 5}};

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                cY[i, j] = in.cY[index_table[i][j]][pos, dim];
                cA[i, j] = in.cA[index_table[i][j]][pos, dim];
            }
        }

        K = in.K[pos, dim];
        W = in.W[pos, dim];

        for(int i=0; i < 3; i++)
            cG[i] = in.cG[i][pos, dim];

        gA = in.gA[pos, dim];

        for(int i=0; i < 3; i++)
            gB[i] = in.gB[i][pos, dim];


        for(int k=0; k < 3; k++)
        {
            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    int index = index_table[i][j];

                    dcY[k, i, j] = derivatives.dcY[index][k][pos, dim];
                }

                dgB[k, i] = derivatives.dgB[i][k][pos, dim];
            }

            dgA[k] = derivatives.dgA[k][pos, dim];
            dW[k] = derivatives.dW[k][pos, dim];
        }
    }
};

std::string make_derivatives()
{
    /*auto differentiate = [&](execution_context&, buffer<valuef> in, std::array<buffer_mut<valuef>, 3> out, literal<v3i> dim, literal<valuef> scale)
    {
        using namespace single_source;

        valuei x = value_impl::get_global_id(0);
        valuei y = value_impl::get_global_id(1);
        valuei z = value_impl::get_global_id(2);

        pin(x);
        pin(y);
        pin(z);

        if_e(x >= dim.get().x() || y >= dim.get().y() || z >= dim.get().z(), [&] {
            return_e();
        });

        v3i pos = {x, y, z};

        valuei index = pos.z() * dim.get().y() * dim.get().x() + pos.y() * dim.get().x() + pos.x();

        pin(index);

        valuef v1 = in[index];

        as_ref(out[0][index]) = valuef(0);

        //as_ref(out[0][pos, dim.get().xyz()]) = diff1(v1, 0, scale.get());
        //as_ref(out[1][pos, dim.get()]) = diff1(v1, 1, scale.get());
        //as_ref(out[2][pos, dim.get()]) = diff1(v1, 2, scale.get());
    };

    std::string str = value_impl::make_function(differentiate, "differentiate");

    std::cout << str << std::endl;

    return str;*/

    return R"(
__kernel void differentiate(global float* o1, int3 dim, float scale)
{
    int lid = get_global_id(0);
    //int y = get_global_id(1);
    //int z = get_global_id(2);

    int x = lid % dim.x;
    int y = (lid - z * dim.x * dim.y) / dim.x;
    int z = lid / (dim.x * dim.y);

    if(lid >= dim.x * dim.y * dim.z)
        return;

    //printf("X %i %i %i\n", x, y, z);

    o1[z * dim.x * dim.y + y * dim.x + x] = 0.f;
}

              )";
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

tensor<valuef, 3, 3> calculate_cRij(bssn_args& args, const valuef& scale)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    pin(icY);

    auto christoff1 = christoffel_symbols_1(args.cY, scale);
    auto christoff2 = christoffel_symbols_2(icY, args.dcY);

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
                    s1 += -0.5f * icY[l, m] * diff2(args.cY[i, j], m, l, args.dcY[m, i, j], args.dcY[l, i, j], scale);
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
tensor<valuef, 3, 3> calculate_W2_mult_Rij(bssn_args& args, valuef scale)
{
    using namespace single_source;

    auto icY = args.cY.invert();
    pin(icY);

    auto christoff2 = christoffel_symbols_2(icY, args.dcY);

    pin(christoff2);

    tensor<valuef, 3, 3> didjW;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            didjW[i, j] = double_covariant_derivative(args.W, args.dW, christoff2, scale)[j, i];
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
                    sum += icY.raise(args.dW)[l] * args.dW[l];
                }

                v3 = -2 * args.cY[i, j] * sum;
            }

            w2Rphiij[i, j] = v1 + v2 + v3;
        }
    }

    pin(w2Rphiij);

    return w2Rphiij + calculate_cRij(args, scale) * args.W * args.W;
}

float get_algebraic_damping_factor()
{
    return 3.f;
}

time_derivatives get_evolution_variables(bssn_args& args, const valuef& scale)
{
    using namespace single_source;

    time_derivatives ret;

    inverse_metric<valuef, 3, 3> icY = args.cY.invert();
    pin(icY);

    ///dtcY
    {
        ///https://arxiv.org/pdf/1307.7391 specifically for why the trace free aspect
        ///https://arxiv.org/pdf/1106.2254 also see here, after 25
        ret.dtcY = lie_derivative_weight(args.gB, args.cY, scale) - 2 * args.gA * trace_free(args.cA, args.cY, icY);

        ret.dtcY += -get_algebraic_damping_factor() * args.gA * args.cY.to_tensor() * log(args.cY.det());
    }

    ///https://iopscience.iop.org/article/10.1088/1361-6382/ac7e16/pdf 2.12 or
    ///https://arxiv.org/pdf/0709.2160
    ///dtW
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

        ret.dtW = (1/3.f) * args.W * (args.gA * args.K - dibi) + dibiw;
    }

    tensor<valuef, 3, 3, 3> christoff2 = christoffel_symbols_2(icY, args.dcY);

    pin(christoff2);

    ///W^2 = X
    valuef X = args.W * args.W;
    ///2 dW W = dX
    tensor<valuef, 3> dX = 2 * args.W * args.dW;

    value iX = 1/max(X, valuef(0.00001f));

    tensor<valuef, 3, 3> DiDja;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            valuef v1 = double_covariant_derivative(args.gA, args.dgA, christoff2, scale)[i, j];

            valuef v2 = 0.5f * iX * (dX[i] * diff1(args.gA, j, scale) + dX[j] * diff1(args.gA, i, scale));

            valuef sum = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    sum += icY[m, n] * dX[m] * diff1(args.gA, n, scale);
                }
            }

            valuef v3 = 0.5f * iX * args.cY[i, j] * sum;

            DiDja[i, j] = v1 + v2 + v3;
        }
    }

    pin(DiDja);

    {
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

        ret.dtK = v1 + v2 + v3 + v4;
    }

    ///dtcA
    {
        tensor<valuef, 3, 3> with_trace = args.gA * calculate_W2_mult_Rij(args, scale) - X * DiDja;

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

                ret.dtcA[i, j] = v1 + v2 + v3 + v4;
            }
        }

        ret.dtcA += -get_algebraic_damping_factor() * args.gA * args.cY.to_tensor() * trace(args.cA, icY);
    }

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
                s3 += icAij[i, j] * 2 * args.dW[j];
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
                s6 += -args.cG[j] * args.dgB[j, i];
            }

            valuef s7 = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    s7 += icY.idx(j, k) * diff2(args.gB[i], k, j, args.dgB[k, i], args.dgB[j, i], scale);
                }
            }

            valuef s8 = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    s8 += icY[i, j] * diff2(args.gB[k], k, j, args.dgB[k, k], args.dgB[j, k], scale);
                }
            }

            s8 = (1.f/3.f) * s8;

            valuef s9 = 0;

            for(int k=0; k < 3; k++)
            {
                s9 += args.dgB[k, k];
            }

            s9 = (2.f/3.f) * s9 * args.cG[i];

            ret.dtcG[i] = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9;
        }

        #define STABILITY_SIGMA
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

        float sigma = 0.25f;

        ret.dtcG += sigma * Gi * dmbm;
        #endif // STABILITY_SIGMA
    }

    #ifdef BLACK_HOLE_GAUGE
    #define ONE_PLUS_LOG
    #define GAMMA_DRIVER
    #endif // BLACK_HOLE_GAUGE

    #define WAVE_TEST
    #ifdef WAVE_TEST
    #define HARMONIC_SLICING
    #define ZERO_SHIFT
    #endif // WAVE_TEST

    {
        valuef bmdma = 0;

        for(int i=0; i < 3; i++)
        {
            bmdma += args.gB[i] * diff1(args.gA, i, scale);
        }

        ///https://arxiv.org/pdf/gr-qc/0206072
        #ifdef ONE_PLUS_LOG
        ret.dtgA = -2 * args.gA * args.K + bmdma;
        #endif // ONE_PLUS_LOG

        ///https://arxiv.org/pdf/2201.08857
        #ifdef HARMONIC_SLICING
        ret.dtgA = -args.gA * args.gA * args.K + bmdma;
        #endif // HARMONIC_SLICING
    }

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

        ret.dtgB = (3/4.f) * args.cG + djbjbi - N * args.gB;
        #endif // GAMMA_DRIVER

        #ifdef ZERO_SHIFT
        ret.dtgB = {0,0,0};
        #endif // ZERO_SHIFT
    }

    return ret;
}

valuef apply_evolution(const valuef& base, const valuef& dt, valuef timestep)
{
    return base + dt * timestep;
}

std::string make_bssn()
{
    auto bssn_function = [&](execution_context&, bssn_args_mem<buffer<valuef>> base,
                                                 bssn_args_mem<buffer<valuef>> in,
                                                 bssn_args_mem<buffer_mut<valuef>> out,
                                                 bssn_derivatives_mem<buffer<derivative_t>> derivatives,
                                                 literal<valuef> timestep,
                                                 literal<v3i> dim,
                                                 literal<valuef> scale) {
        using namespace single_source;

        valuei x = value_impl::get_global_id(0);
        valuei y = value_impl::get_global_id(1);
        valuei z = value_impl::get_global_id(2);

        pin(x);
        pin(y);
        pin(z);

        if_e(x >= dim.get().x() || y >= dim.get().y() || z >= dim.get().z(), [&] {
            return_e();
        });

        v3i pos = {x, y, z};

        valuei linear_index = pos.z() * dim.get().y() * dim.get().x() + pos.y() * dim.get().x() + pos.x();

        bssn_args args(pos, dim.get(), in, derivatives);

        time_derivatives in_time = get_evolution_variables(args, scale.get());

        tensor<int, 2> index_table[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

        for(int i=0; i < 6; i++)
        {
            tensor<int, 2> idx = index_table[i];

            as_ref(out.cY[i][linear_index]) = apply_evolution(base.cY[i][linear_index], in_time.dtcY[idx.x(), idx.y()], timestep.get());
            as_ref(out.cA[i][linear_index]) = apply_evolution(base.cA[i][linear_index], in_time.dtcA[idx.x(), idx.y()], timestep.get());
        }

        for(int i=0; i < 3; i++)
        {
            as_ref(out.gB[i][linear_index]) = apply_evolution(base.gB[i][linear_index], in_time.dtgB[i], timestep.get());
            as_ref(out.cG[i][linear_index]) = apply_evolution(base.cG[i][linear_index], in_time.dtcG[i], timestep.get());
        }

        as_ref(out.gA[linear_index]) = apply_evolution(base.gA[linear_index], in_time.dtgA, timestep.get());
        as_ref(out.W[linear_index]) = apply_evolution(base.W[linear_index], in_time.dtW, timestep.get());
        as_ref(out.K[linear_index]) = apply_evolution(base.K[linear_index], in_time.dtK, timestep.get());
    };

    return value_impl::make_function(bssn_function, "evolve");
}

/*
///https://hal.archives-ouvertes.fr/hal-00569776/document this paper implies you simply sum the directions
///https://en.wikipedia.org/wiki/Finite_difference_coefficient according to wikipedia, this is the 6th derivative with 2nd order accuracy. I am confused, but at least I know where it came from
value kreiss_oliger_dissipate(equation_context& ctx, const value& in, const value_i& order)
{
    value_i n = get_maximum_differentiation_derivative(order);

    n = min(n, value_i{6});

    value fin = 0;

    for(int i=0; i < 3; i++)
    {
        fin += diffnth(ctx, in, i, n, value{1.f});
    }

    value scale = "scale";

    value p = n.convert<float>() - 1;

    value sign = pow(value{-1}, (p + 3)/2);

    value divisor = pow(value{2}, p+1);

    value prefix = sign / divisor;

    return prefix * fin / scale;
}

void kreiss_oliger_unidir(equation_context& ctx, buffer<tensor<value_us, 4>> points, literal<value_i> point_count,
                          buffer<value> buf_in, buffer<value_mut> buf_out,
                          literal<value> eps, single_source::named_literal<value, "scale"> scale, single_source::named_literal<tensor<value_i, 4>, "dim"> idim, literal<value> timestep,
                          buffer<value_us> order_ptr)
{
    using namespace dual_types::implicit;

    value_i local_idx = declare(ctx, value_i{"get_global_id(0)"}, "local_idx");

    if_e(local_idx >= point_count, [&]()
    {
        return_e();
    });

    value_i ix = declare(ctx, points[local_idx].x().convert<int>(), "ix");
    value_i iy = declare(ctx, points[local_idx].y().convert<int>(), "iy");
    value_i iz = declare(ctx, points[local_idx].z().convert<int>(), "iz");

    v3i pos = {ix, iy, iz};
    v3i dim = {idim.get().x(), idim.get().y(), idim.get().z()};

    value_i order = declare(ctx, order_ptr[(v3i){ix, iy, iz}, dim].convert<int>(), "order");

    ///note to self we're not actually doing this correctly
    value_i is_valid_point = ((order & value_i{(int)D_LOW}) > 0) || ((order & value_i{(int)D_FULL}) > 0);

    assert(buf_out.storage.is_mutable);

    if_e(!is_valid_point, [&]()
    {
        mut(buf_out[pos, dim]) = buf_in[pos, dim];
        return_e();
    });

    value buf_v = bidx(ctx, buf_in.name, false, false);

    value v = buf_v + timestep * eps * kreiss_oliger_dissipate(ctx, buf_v, order);

    mut(buf_out[(v3i){ix, iy, iz}, dim]) = v;
}
*/

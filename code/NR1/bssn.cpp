#include "bssn.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"

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

using derivative_t = valuef;

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

                        value_base op;
                        op.type = value_impl::op::BRACKET;
                        op.args = {buf, value<int>(3), next_x, next_y, next_z, dx, dy, dz};
                        op.concrete = get_interior_type(T());

                        return op;
                    };

                    v = get_substitution(v);
                }
            });
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
    auto differentiate = [&](execution_context&, buffer<valuef> in, std::array<buffer_mut<valuef>, 3> out, literal<v3i> dim, literal<valuef> scale)
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

        valuef v1 = in[pos, dim.get()];

        as_ref(out[0][pos, dim.get()]) = diff1(v1, 0, scale.get());
        as_ref(out[1][pos, dim.get()]) = diff1(v1, 1, scale.get());
        as_ref(out[2][pos, dim.get()]) = diff1(v1, 2, scale.get());
    };

    return value_impl::make_function(differentiate, "differentiate");
}

struct evolution_variables
{
    tensor<valuef, 3, 3> dtcY;
    tensor<valuef, 3, 3> dtcA;
    valuef dtK;
    valuef dtW;
    tensor<valuef, 3> dtcG;

    valuef dtgA;
    tensor<valuef, 3> dtgB;
};

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

evolution_variables get_evolution_variables(bssn_args& args, const valuef& scale)
{
    evolution_variables ret;

    ///dtcY
    {
        ret.dtcY = lie_derivative_weight(args.gB, args.cY, scale) - 2 * args.gA * args.cA;
    }

    ///https://iopscience.iop.org/article/10.1088/1361-6382/ac7e16/pdf 2.12 or
    ///https://arxiv.org/pdf/0709.2160
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

    return ret;
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

        bssn_args args(pos, dim.get(), in, derivatives);
    };

    return value_impl::make_function(bssn_function, "evolve");
}

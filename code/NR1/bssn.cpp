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

                        value_base next_x = old_x + offx[i];
                        value_base next_y = old_y + offy[i];
                        value_base next_z = old_z + offz[i];

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

template<typename T>
valuef diff1(const valuef& val, int direction, valuef scale)
{
    ///second order derivatives
    differentiation_context dctx(val, direction);
    std::array<valuef, 5> vars = dctx.vars;

    valuef p1 = -vars[4] + vars[0];
    valuef p2 = valuef(8.f) * (vars[3] - vars[1]);

    return (p1 + p2) / (12.f * scale);
}

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
    std::array<std::array<T, 3>, 6> dcY;
    std::array<T, 3> dcA;
    std::array<std::array<T, 3>, 3> dgB;
    std::array<T, 3> dW;

    void build(auto& in)
    {
        using namespace value_impl::builder;

        add(dcY, in);
        add(dcA, in);
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

    bssn_args(v3i pos, v3i dim,
              bssn_args_mem<buffer<valuef>>& in, bssn_derivatives_mem<buffer<valuef>>& derivs)
    {
        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                int index_table[3][3] = {{0, 1, 2},
                                         {1, 3, 4},
                                         {2, 4, 5}};

                cY[i, j] = in.cY[index_table[i][j]][pos, dim];
                cA[i, j] = in.cA[index_table[i][j]][pos, dim];
            }
        }

        ///todo: full 3d index
        K = in.K[pos, dim];
        W = in.W[pos, dim];

        for(int i=0; i < 3; i++)
            cG[i] = in.cG[i][pos, dim];

        gA = in.gA[pos, dim];

        for(int i=0; i < 3; i++)
            gB[i] = in.gB[i][pos, dim];
    }
};

std::string make_derivatives()
{
    auto differentiate = [&](execution_context&, buffer<valuef> in, std::array<buffer_mut<valuef>, 3> out, literal<v3i> dim)
    {

    };

    return value_impl::make_function(differentiate, "differentiate");
}

std::string make_bssn()
{
    auto bssn_function = [&](execution_context&, bssn_args_mem<buffer<valuef>> base,
                                                 bssn_args_mem<buffer<valuef>> in,
                                                 bssn_args_mem<buffer_mut<valuef>> out,
                                                 bssn_derivatives_mem<buffer<valuef>> derivatives,
                                                 literal<valuef> timestep,
                                                 literal<v3i> dim) {

    };

    return value_impl::make_function(bssn_function, "evolve");
}

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

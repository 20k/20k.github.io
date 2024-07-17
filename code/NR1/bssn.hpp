#ifndef BSSN_HPP_INCLUDED
#define BSSN_HPP_INCLUDED

#include <string>
#include <array>
#include "../common/value2.hpp"
#include "../common/single_source.hpp"

using derivative_t = value<float16>;
using valuef = value<float>;
using valued = value<double>;
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

v3i get_coordinate(valuei id, v3i dim);

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

    bssn_args(v3i pos, v3i dim,
              bssn_args_mem<buffer<value<float>>>& in)
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
    }
};

struct bssn_derivatives
{
    ///diYjk
    tensor<valuef, 3, 3, 3> dcY;
    tensor<valuef, 3> dgA;
    ///digBj
    tensor<valuef, 3, 3> dgB;
    tensor<valuef, 3> dW;

    bssn_derivatives(v3i pos, v3i dim, bssn_derivatives_mem<buffer<derivative_t>>& derivatives)
    {
        int index_table[3][3] = {{0, 1, 2},
                                 {1, 3, 4},
                                 {2, 4, 5}};

        for(int k=0; k < 3; k++)
        {
            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    int index = index_table[i][j];

                    dcY[k, i, j] = (valuef)derivatives.dcY[index][k][pos, dim];
                }

                dgB[k, i] = (valuef)derivatives.dgB[i][k][pos, dim];
            }

            dgA[k] = (valuef)derivatives.dgA[k][pos, dim];
            dW[k] = (valuef)derivatives.dW[k][pos, dim];
        }
    }
};

valuef calculate_hamiltonian_constraint(bssn_args& args, bssn_derivatives& derivs, const valuef& scale);
tensor<valuef, 3> calculate_momentum_constraint(bssn_args& args, const valuef& scale);

std::string make_derivatives();
std::string make_bssn(const tensor<int, 3>& dim);
std::string enforce_algebraic_constraints();
std::string init_debugging();
std::string make_momentum_constraint();

#endif // BSSN_HPP_INCLUDED

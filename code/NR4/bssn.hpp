#ifndef BSSN_HPP_INCLUDED
#define BSSN_HPP_INCLUDED

#include <string>
#include <array>
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include <toolkit/opencl.hpp>
#include "value_alias.hpp"

#define MOMENTUM_CONSTRAINT_DAMPING

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

struct bssn_derivatives;

struct bssn_args
{
    unit_metric<valuef, 3, 3> cY;
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
        W = max(in.W[pos, dim], valuef(1e-4f));

        for(int i=0; i < 3; i++)
            cG[i] = in.cG[i][pos, dim];

        gA = max(in.gA[pos, dim], valuef(1e-4f));

        for(int i=0; i < 3; i++)
            gB[i] = in.gB[i][pos, dim];
    }

    tensor<valuef, 3> cG_undiff(const bssn_derivatives& derivs);
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

struct bssn_buffer_pack
{
    std::array<cl::buffer, 6> cY;
    std::array<cl::buffer, 6> cA;
    cl::buffer K;
    cl::buffer W;
    std::array<cl::buffer, 3> cG;

    cl::buffer gA;
    std::array<cl::buffer, 3> gB;

    //lovely
    bssn_buffer_pack(cl::context& ctx) :
        cY{ctx, ctx, ctx, ctx, ctx, ctx},
        cA{ctx, ctx, ctx, ctx, ctx, ctx},
        K{ctx},
        W{ctx},
        cG{ctx, ctx, ctx},
        gA{ctx},
        gB{ctx, ctx, ctx}
    {

    }

    void allocate(t3i size)
    {
        int64_t linear_size = int64_t{size.x()} * size.y() * size.z();

        for(auto& i : cY)
            i.alloc(sizeof(cl_float) * linear_size);
        for(auto& i : cA)
            i.alloc(sizeof(cl_float) * linear_size);
        for(auto& i : cG)
            i.alloc(sizeof(cl_float) * linear_size);
        for(auto& i : gB)
            i.alloc(sizeof(cl_float) * linear_size);

        K.alloc(sizeof(cl_float) * linear_size);
        W.alloc(sizeof(cl_float) * linear_size);
        gA.alloc(sizeof(cl_float) * linear_size);
    }

    template<typename T>
    void for_each(T&& func)
    {
        for(auto& i : cY)
            func(i);

        for(auto& i : cA)
            func(i);

        func(K);
        func(W);

        for(auto& i : cG)
            func(i);

        func(gA);

        for(auto& i : gB)
            func(i);
    }

    void append_to(cl::args& args)
    {
        for(auto& i : cY)
            args.push_back(i);

        for(auto& i : cA)
            args.push_back(i);

        args.push_back(K);
        args.push_back(W);

        for(auto& i : cG)
            args.push_back(i);

        args.push_back(gA);

        for(auto& i : gB)
            args.push_back(i);
    }
};

struct derivative_data;

valuef calculate_hamiltonian_constraint(bssn_args& args, bssn_derivatives& derivs, const derivative_data& d);
tensor<valuef, 3> calculate_momentum_constraint(bssn_args& args, const derivative_data& d);

struct all_adm_args_mem;

void make_derivatives(cl::context ctx);
void make_bssn(cl::context ctx, const tensor<int, 3>& dim);
void enforce_algebraic_constraints(cl::context ctx);
void init_debugging(cl::context ctx);
void make_momentum_constraint(cl::context ctx, const all_adm_args_mem& args_mem);
void make_sommerfeld(cl::context ctx);

template<typename T, typename U, typename V>
inline
tensor<T, 3> grid_to_world(const tensor<T, 3>& pos, const tensor<V, 3>& dim, const U& scale)
{
    tensor<T, 3> centre = (tensor<T, 3>{(T)dim.x(), (T)dim.y(), (T)dim.z()} - 1) / 2;

    tensor<T, 3> diff = pos - centre;

    return diff * scale;
}

template<typename T, typename U, typename V>
inline
tensor<T, 3> world_to_grid(const tensor<T, 3>& pos, const tensor<V, 3>& dim, const U& scale)
{
    tensor<T, 3> centre = (tensor<T, 3>{(T)dim.x(), (T)dim.y(), (T)dim.z()} - 1) / 2;

    return (pos / scale) + centre;
}

template<typename T>
inline
std::array<T, 6> extract_symmetry(const tensor<T, 3, 3>& in)
{
    return {in[0, 0], in[1, 0], in[2, 0], in[1, 1], in[2, 1], in[2, 2]};
}

template<typename T>
inline
tensor<T, 3, 3> make_symmetry(const std::array<T, 6>& in)
{
    tensor<T, 3, 3> ret;
    ret[0, 0] = in[0];
    ret[1, 0] = in[1];
    ret[2, 0] = in[2];

    ret[1, 1] = in[3];
    ret[2, 1] = in[4];
    ret[2, 2] = in[5];

    ret[0, 1] = in[1];
    ret[0, 2] = in[2];
    ret[1, 2] = in[4];

    return ret;
}

#endif // BSSN_HPP_INCLUDED

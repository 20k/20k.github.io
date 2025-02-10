#ifndef INIT_NEUTRON_STAR_HPP_INCLUDED
#define INIT_NEUTRON_STAR_HPP_INCLUDED

#include <vec/tensor.hpp>
#include <vector>
#include "tov.hpp"
#include <toolkit/opencl.hpp>

///end goal: calculate conformal ppw2p and conformal aij_aij
///take in a tov solution
namespace neutron_star
{
    ///so, I'm going to crack all the constants in here, then perform the discretisation on the GPU
    struct solution
    {
        std::vector<double> mu_cfl;
        std::vector<double> pressure_cfl;
        //std::vector<tensor<double, 3, 3>> AIJ_cfl;

        std::vector<double> N;
        std::vector<double> Q;
        std::vector<double> C;
    };

    solution solve(const tov::integration_solution& sol);

    struct discretised_solution
    {
        cl::buffer mu_cfl;
        cl::buffer pressure_cfl;
        std::array<cl::buffer, 6> Aij_cfl;

        discretised_solution(cl::context& ctx) : mu_cfl(ctx), pressure_cfl(ctx), Aij_cfl{ctx, ctx, ctx, ctx, ctx, ctx}{}
    };

    void add_to_solution(cl::context& ctx, cl::command_queue& cqueue, discretised_solution& dsol, const solution& nsol, const tov::integration_solution& tsol);
}

#endif // INIT_NEUTRON_STAR_HPP_INCLUDED

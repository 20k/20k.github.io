#ifndef INIT_NEUTRON_STAR_HPP_INCLUDED
#define INIT_NEUTRON_STAR_HPP_INCLUDED

#include <vec/tensor.hpp>
#include <vector>
#include "tov.hpp"
#include <toolkit/opencl.hpp>

///end goal: calculate conformal ppw2p and conformal aij_aij
///take in a tov solution
///todo: unify all the params structs
namespace neutron_star
{
    void boot_solver(cl::context ctx);

    struct params
    {
        tensor<float, 3> position;
        tensor<float, 3> linear_momentum;
        tensor<float, 3> angular_momentum;
    };

    struct discretised_solution
    {
        cl::buffer mu_cfl;
        cl::buffer mu_h_cfl;
        cl::buffer pressure_cfl;
        std::array<cl::buffer, 6> AIJ_cfl;
        std::array<cl::buffer, 3> Si;

        discretised_solution(cl::context& ctx) : mu_cfl(ctx), mu_h_cfl(ctx), pressure_cfl(ctx), AIJ_cfl{ctx, ctx, ctx, ctx, ctx, ctx}, Si{ctx, ctx, ctx}{}
    };

    void add_to_solution(cl::context& ctx, cl::command_queue& cqueue,
                         discretised_solution& dsol, const params& phys_params, const tov::integration_solution& sol,
                         tensor<int, 3> dim, float scale);
}

#endif // INIT_NEUTRON_STAR_HPP_INCLUDED

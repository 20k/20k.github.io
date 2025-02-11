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

        double K = 100;
        double M_sol = 1.4;
        double Gamma = 2;
    };

    struct discretised_solution
    {
        cl::buffer mu_cfl;
        cl::buffer mu_h_cfl;
        cl::buffer pressure_cfl;
        std::array<cl::buffer, 6> AIJ_cfl;
        std::array<cl::buffer, 3> Si;

        discretised_solution(cl::context& ctx) : mu_cfl(ctx), mu_h_cfl(ctx), pressure_cfl(ctx), AIJ_cfl{ctx, ctx, ctx, ctx, ctx, ctx}, Si{ctx, ctx, ctx}{}

        void init(cl::command_queue& cqueue, t3i dim)
        {
            int64_t cells = int64_t{dim.x()} * dim.y() * dim.z();

            mu_cfl.alloc(sizeof(cl_float) * cells);
            mu_h_cfl.alloc(sizeof(cl_float) * cells);
            pressure_cfl.alloc(sizeof(cl_float) * cells);

            mu_cfl.set_to_zero(cqueue);
            mu_h_cfl.set_to_zero(cqueue);
            pressure_cfl.set_to_zero(cqueue);

            for(auto& i : AIJ_cfl)
            {
                i.alloc(sizeof(cl_float) * cells);
                i.set_to_zero(cqueue);
            }

            for(auto& i : Si)
            {
                i.alloc(sizeof(cl_float) * cells);
                i.set_to_zero(cqueue);
            }
        }
    };

    void add_to_solution(cl::context& ctx, cl::command_queue& cqueue,
                         discretised_solution& dsol, const params& phys_params, const tov::integration_solution& sol,
                         tensor<int, 3> dim, float scale);
}

#endif // INIT_NEUTRON_STAR_HPP_INCLUDED

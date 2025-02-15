#ifndef INIT_NEUTRON_STAR_HPP_INCLUDED
#define INIT_NEUTRON_STAR_HPP_INCLUDED

#include <vec/tensor.hpp>
#include <vector>
#include "tov.hpp"
#include <toolkit/opencl.hpp>

struct discretised_initial_data;

///end goal: calculate conformal ppw2p and conformal aij_aij
///take in a tov solution
///todo: unify all the params structs
namespace neutron_star
{
    void boot_solver(cl::context ctx);

    struct parameters
    {
        tensor<float, 3> position;
        tensor<float, 3> linear_momentum;
        tensor<float, 3> angular_momentum;

        double K = 123.741;
        double Gamma = 2;
        double p0_c_kg_m3 = 6.235 * pow(10., 17.);
    };

    struct data
    {
        parameters params;
        tov::integration_solution sol;

        data(const parameters& p) : params(p)
        {
            tov::parameters tov_params;
            tov_params.K = params.K;
            tov_params.Gamma = params.Gamma;

            auto start = tov::make_integration_state_si(params.p0_c_kg_m3, 1e-6, tov_params);
            sol = tov::solve_tov(start, tov_params, 1e-6, 0);
        }

        void add_to_solution(cl::context& ctx, cl::command_queue& cqueue,
                             discretised_initial_data& dsol,
                             tensor<int, 3> dim, float scale, int star_index);
    };
}

#endif // INIT_NEUTRON_STAR_HPP_INCLUDED

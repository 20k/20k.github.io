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

    struct params
    {
        tensor<float, 3> position;
        tensor<float, 3> linear_momentum;
        tensor<float, 3> angular_momentum;

        double K = 100;
        double M_sol = 1.4;
        double Gamma = 2;
    };

    void add_to_solution(cl::context& ctx, cl::command_queue& cqueue,
                         discretised_initial_data& dsol, const params& phys_params, const tov::integration_solution& sol,
                         tensor<int, 3> dim, float scale);
}

#endif // INIT_NEUTRON_STAR_HPP_INCLUDED

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

    struct numerical_eos
    {
        ///linear map from rest mass density -> pressure
        std::vector<float> pressure;
        float max_density = 0;
    };

    struct all_numerical_eos_gpu
    {
        cl::buffer pressures;
        cl::buffer max_densities;
        cl_int stride = 0;
        cl_int count = 0;

        all_numerical_eos_gpu(cl::context ctx) : pressures(ctx), max_densities(ctx){}

        void init(cl::command_queue cqueue, const std::vector<numerical_eos>& eos);
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

        numerical_eos get_eos();
    };
}

#endif // INIT_NEUTRON_STAR_HPP_INCLUDED

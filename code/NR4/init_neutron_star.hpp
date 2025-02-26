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

    struct param_K
    {
        std::optional<double> msols;
    };

    struct param_rest_mass
    {
        double mass = 0;
        int result_index = 0;
    };

    struct param_mass
    {
        std::optional<double> p0_kg_m3;
        std::optional<double> p0_geometric;
        std::optional<double> p0_msols;

        std::optional<param_rest_mass> rest_mass;
    };

    struct dimensionless_linear_momentum
    {
        t3f axis = {0, 0, 1};
        ///M is baryonic mass
        ///P / M = x
        double x = 0;
    };

    struct param_linear_momentum
    {
        std::optional<t3f> momentum;
        std::optional<dimensionless_linear_momentum> dimensionless;
    };

    struct dimensionless_angular_momentum
    {
        t3f axis = {0, 0, 1};
        ///M is baryonic mass
        ///J / M^2 = x
        double x = 0;
    };

    struct param_angular_momentum
    {
        std::optional<t3f> momentum;
        std::optional<dimensionless_angular_momentum> dimensionless;
    };

    struct parameters
    {
        //linear colour
        std::optional<t3f> colour;
        tensor<float, 3> position;
        param_linear_momentum linear_momentum;
        param_angular_momentum angular_momentum;

        double Gamma = 2;

        //123.741 is a good number
        param_K K;
        //6.235 * pow(10., 17.) kg/m3 is a good number
        param_mass mass;
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
        double p0_msols = 0;
        double total_mass = 0;
        numerical_eos stored;

        data(const parameters& p);

        void add_to_solution(cl::context& ctx, cl::command_queue& cqueue,
                             discretised_initial_data& dsol,
                             tensor<int, 3> dim, float scale, int star_index);

        numerical_eos get_eos();
    };
}

#endif // INIT_NEUTRON_STAR_HPP_INCLUDED

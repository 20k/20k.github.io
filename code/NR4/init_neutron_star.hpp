#ifndef INIT_NEUTRON_STAR_HPP_INCLUDED
#define INIT_NEUTRON_STAR_HPP_INCLUDED

#include <vec/tensor.hpp>
#include <vector>
#include "tov.hpp"

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
}

#endif // INIT_NEUTRON_STAR_HPP_INCLUDED

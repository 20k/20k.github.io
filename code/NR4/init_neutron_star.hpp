#ifndef INIT_NEUTRON_STAR_HPP_INCLUDED
#define INIT_NEUTRON_STAR_HPP_INCLUDED

#include <vec/tensor.hpp>
#include <vector>
#include "tov.hpp"

///end goal: calculate conformal ppw2p and conformal aij_aij
///take in a tov solution
namespace neutron_star
{
    struct solution
    {
        std::vector<double> ph_cfl;
        std::vector<tensor<double, 3, 3>> AIJ_cfl;
    };

    solution solve(const tov::integration_solution& sol);
}

#endif // INIT_NEUTRON_STAR_HPP_INCLUDED

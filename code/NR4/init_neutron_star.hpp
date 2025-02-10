#ifndef INIT_NEUTRON_STAR_HPP_INCLUDED
#define INIT_NEUTRON_STAR_HPP_INCLUDED

#include <vec/tensor.hpp>
#include <vector>
#include "tov.hpp"

///end goal: calculate ppw2p, provide a discretisation facility
///take in a tov solution

namespace neutron_star
{
    struct neutron_star_solution
    {
        std::vector<double> ph;
        std::vector<tensor<double, 3, 3>> Aij;
    };

    neutron_star_solution solve(const tov::integration_solution& sol);
}

#endif // INIT_NEUTRON_STAR_HPP_INCLUDED

#ifndef GALAXY_MODEL_HPP_INCLUDED
#define GALAXY_MODEL_HPP_INCLUDED

#include <vector>
#include <vec/tensor.hpp>

using t3f = tensor<float, 3>;
using t2f = tensor<float, 2>;

struct galaxy_data
{
    std::vector<t3f> positions;
    std::vector<t3f> velocities;
    std::vector<float> masses;

    std::vector<float> debug_velocities;
    std::vector<float> debug_analytic_mass;
    std::vector<float> debug_real_mass;
};

galaxy_data build_galaxy(float simulation_width);

#endif // GALAXY_MODEL_HPP_INCLUDED

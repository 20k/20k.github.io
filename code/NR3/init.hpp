#ifndef INIT_HPP_INCLUDED
#define INIT_HPP_INCLUDED

#include <string>
#include <array>
#include "../common/value2.hpp"
#include "../common/vec/tensor.hpp"
#include <toolkit/opencl.hpp>

using valuef = value<float>;
using valued = value<double>;
using valuei = value<int>;
using valueh = value<float16>;

using v2f = tensor<valuef, 2>;
using v3f = tensor<valuef, 3>;
using v4f = tensor<valuef, 4>;
using v2i = tensor<valuei, 2>;
using v3i = tensor<valuei, 3>;
using v4i = tensor<valuei, 4>;
using m44f = metric<valuef, 4, 4>;
using v3h = tensor<valueh, 3>;

using mut_v4f = tensor<mut<valuef>, 4>;
using mut_v3f = tensor<mut<valuef>, 3>;

using t3i = tensor<int, 3>;
using t3f = tensor<float, 3>;

struct adm_variables
{
    metric<valuef, 3, 3> Yij;
    tensor<valuef, 3, 3> Kij;
    valuef gA;
    tensor<valuef, 3> gB;
};

struct bssn_args;

adm_variables bssn_to_adm(const bssn_args& args);

void make_initial_conditions(cl::context ctx);
void init_christoffel(cl::context ctx);

#endif // INIT_HPP_INCLUDED

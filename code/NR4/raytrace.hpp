#ifndef RAYTRACE_HPP_INCLUDED
#define RAYTRACE_HPP_INCLUDED

#include "../common/vec/tensor.hpp"
#include "../common/value2.hpp"
#include <toolkit/opencl.hpp>

using valuef = value<float>;
using valuei = value<int>;

using v2f = tensor<valuef, 2>;
using v3f = tensor<valuef, 3>;
using v4f = tensor<valuef, 4>;
using v2i = tensor<valuei, 2>;
using v3i = tensor<valuei, 3>;
using v4i = tensor<valuei, 4>;
using m44f = metric<valuef, 4, 4>;

using mut_v4f = tensor<mut<valuef>, 4>;
using mut_v3f = tensor<mut<valuef>, 3>;

using block_precision_t = valuef;

struct plugin;

void build_raytrace_kernels(cl::context ctx, const std::vector<plugin*>& plugins);

#endif // RAYTRACE_HPP_INCLUDED

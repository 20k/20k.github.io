#ifndef RAYTRACE_HPP_INCLUDED
#define RAYTRACE_HPP_INCLUDED

#include "../common/vec/tensor.hpp"
#include "../common/value2.hpp"
#include <toolkit/opencl.hpp>
#include "value_alias.hpp"

struct plugin;

void build_raytrace_kernels(cl::context ctx, const std::vector<plugin*>& plugins, bool use_matter, bool use_colour);

#endif // RAYTRACE_HPP_INCLUDED

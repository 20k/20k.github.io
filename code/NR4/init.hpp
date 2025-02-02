#ifndef INIT_HPP_INCLUDED
#define INIT_HPP_INCLUDED

#include <toolkit/opencl.hpp>

void make_initial_conditions(cl::context ctx);
void init_christoffel(cl::context ctx);

#endif // INIT_HPP_INCLUDED

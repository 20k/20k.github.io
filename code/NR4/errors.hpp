#ifndef ERRORS_HPP_INCLUDED
#define ERRORS_HPP_INCLUDED

#include <toolkit/opencl.hpp>

void make_cG_error(cl::context ctx, int idx);
//void make_momentum_error(cl::context ctx, int idx);
void make_hamiltonian_error(cl::context ctx);
void make_global_sum(cl::context ctx);

#endif // ERRORS_HPP_INCLUDED

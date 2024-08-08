#ifndef ERRORS_HPP_INCLUDED
#define ERRORS_HPP_INCLUDED

#include <string>

std::string make_cG_error(int idx);
std::string make_momentum_error(int idx);
std::string make_hamiltonian_error();
std::string make_global_sum();

#endif // ERRORS_HPP_INCLUDED

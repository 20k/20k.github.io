#ifndef BSSN_HPP_INCLUDED
#define BSSN_HPP_INCLUDED

#include <string>

std::string make_derivatives();
std::string make_bssn();
std::string make_initial_conditions();
std::string init_christoffel();
std::string init_debugging();
std::string make_momentum_constraint();
std::string make_kreiss_oliger();

#endif // BSSN_HPP_INCLUDED

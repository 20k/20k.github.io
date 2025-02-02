#ifndef FORMALISMS_HPP_INCLUDED
#define FORMALISMS_HPP_INCLUDED

#include "../common/value2.hpp"
#include "../common/vec/tensor.hpp"
#include "bssn.hpp"

struct adm_variables
{
    metric<valuef, 3, 3> Yij;
    tensor<valuef, 3, 3> Kij;
    valuef gA;
    tensor<valuef, 3> gB;
};

struct bssn_args;

adm_variables bssn_to_adm(const bssn_args& args);

adm_variables adm_at(v3i pos, v3i dim, bssn_args_mem<buffer<valuef>> in);
bssn_args bssn_at(v3i pos, v3i dim, bssn_args_mem<buffer<valuef>> in);
adm_variables admf_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in);

using block_precision_t = valuef;

metric<valuef, 4, 4> Guv_at(v4i grid_pos, v3i dim, std::array<buffer<block_precision_t>, 10> Guv_buf, valuei last_slice);

#endif // FORMALISMS_HPP_INCLUDED

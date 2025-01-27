#include "formalisms.hpp"

adm_variables bssn_to_adm(const bssn_args& args)
{
    adm_variables ret;
    tensor<valuef, 3, 3> met = args.cY.to_tensor() / (args.W * args.W);

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            ret.Yij[i, j] = met[i, j];
        }
    }

    ret.Kij = (args.cA / (args.W * args.W)) + ret.Yij.to_tensor() * (args.K / 3.f);

    ret.gA = args.gA;
    ret.gB = args.gB;

    return ret;
}

adm_variables adm_at(v3i pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    bssn_args args(pos, dim, in);

    return bssn_to_adm(args);
}

bssn_args bssn_at(v3i pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    bssn_args args(pos, dim, in);
    return args;
}

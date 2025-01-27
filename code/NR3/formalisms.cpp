#include "formalisms.hpp"
#include "interpolation.hpp"

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

adm_variables admf_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    auto Yij_at = [&](v3i pos)
    {
        return adm_at(pos, dim, in).Yij;
    };

    auto Kij_at = [&](v3i pos)
    {
        return adm_at(pos, dim, in).Kij;
    };

    auto gA_at = [&](v3i pos)
    {
        return adm_at(pos, dim, in).gA;
    };

    auto gB_at = [&](v3i pos)
    {
        return adm_at(pos, dim, in).gB;
    };

    adm_variables out;
    out.Yij = function_trilinear(Yij_at, pos);
    out.Kij = function_trilinear(Kij_at, pos);
    out.gA = function_trilinear(gA_at, pos);
    out.gB = function_trilinear(gB_at, pos);

    pin(out.Yij);
    pin(out.Kij);
    pin(out.gA);
    pin(out.gB);

    return out;
}

metric<valuef, 4, 4> Guv_at(v4i grid_pos, v3i dim, std::array<buffer<block_precision_t>, 10> Guv_buf, valuei last_slice)
{
    grid_pos.x() = clamp(grid_pos.x(), valuei(0), last_slice - 1);

    tensor<value<uint64_t>, 3> p = (tensor<value<uint64_t>, 3>)grid_pos.yzw();
    tensor<value<uint64_t>, 3> d = (tensor<value<uint64_t>, 3>)dim;

    ///this might be the problem?
    value<uint64_t> idx = ((value<uint64_t>)grid_pos.x()) * d.x() * d.y() * d.z() + p.z() * d.x() * d.y() + p.y() * d.x() + p.x();

    int indices[16] = {
        0, 1, 2, 3,
        1, 4, 5, 6,
        2, 5, 7, 8,
        3, 6, 8, 9,
    };

    metric<valuef, 4, 4> met;

    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            met[i, j] = (valuef)Guv_buf[indices[j * 4 + i]][idx];
        }
    }

    return met;
}

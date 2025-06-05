#include "init_black_hole.hpp"
#include "tensor_algebra.hpp"
#include "bssn.hpp"
#include "../common/single_source.hpp"
#include "value_alias.hpp"
#include "init_general.hpp"

tensor<valuef, 3, 3> get_pointlike_aIJ(v3f world_pos, v3f pos, v3f angular_momentum, v3f momentum)
{
    tensor<valuef, 3, 3, 3> eijk = get_eijk();

    tensor<valuef, 3, 3> aij;

    metric<valuef, 3, 3> flat;

    for(int i=0; i < 3; i++)
    {
        flat[i, i] = 1;
    }

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            valuef r = (world_pos - pos).length();

            r = max(r, valuef(1e-6f));

            tensor<valuef, 3> n = (world_pos - pos) / r;

            tensor<valuef, 3> momentum_lo = flat.lower(momentum);
            tensor<valuef, 3> n_lo = flat.lower(n);

            aij[i, j] += (3 / (2.f * r * r)) * (momentum_lo[i] * n_lo[j] + momentum_lo[j] * n_lo[i] - (flat[i, j] - n_lo[i] * n_lo[j]) * sum_multiply(momentum, n_lo));

            ///spin
            valuef s1 = 0;
            valuef s2 = 0;

            for(int k=0; k < 3; k++)
            {
                for(int l=0; l < 3; l++)
                {
                    s1 += eijk[k, i, l] * angular_momentum[l] * n[k] * n_lo[j];
                    s2 += eijk[k, j, l] * angular_momentum[l] * n[k] * n_lo[i];
                }
            }

            aij[i, j] += (3 / (r*r*r)) * (s1 + s2);
        }
    }

    return aij;
}

valuef get_conformal_guess(v3f world_pos, v3f bh_pos, valuef bare_mass)
{
    return bare_mass / (2 * (world_pos - bh_pos).length());
}

black_hole_data init_black_hole(cl::context& ctx, cl::command_queue& cqueue, black_hole_params params, tensor<int, 3> dim, float scale)
{
    black_hole_data dat(ctx);
    tensor<int, 2> index_table[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    auto grid_pos = world_to_grid(params.position, dim, scale);

    std::cout << "w2g " << grid_pos[0] << " " << grid_pos[1] << " " << grid_pos[2] << std::endl;

    params.position = grid_to_world(round(world_to_grid(params.position, dim, scale)), dim, scale);

    for(int i=0; i < 6; i++)
    {
        tensor<int, 2> idx = index_table[i];

        auto func = [&](v3i pos, v3i dim, valuef scale)
        {
            using namespace single_source;

            v3f world_pos = grid_to_world((v3f)pos, dim, scale);

            /*if_e(pos.x() == 106 && pos.y() == 106 && pos.z() == 106, [&]{
                value_base se;
                se.type = value_impl::op::SIDE_EFFECT;
                se.abstract_value = "printf(\"K: %i %i %i %f %f %f\\n\"," + value_to_string(pos.x()) + "," + value_to_string(pos.y()) + "," + value_to_string(pos.z()) + "," + value_to_string(world_pos.x()) + "," + value_to_string(world_pos.y()) + "," + value_to_string(world_pos.z()) + ")";

                value_impl::get_context().add(se);
            });*/

            tensor<valuef, 3, 3> aij = get_pointlike_aIJ(world_pos, (v3f)params.position, (v3f)params.angular_momentum, (v3f)params.linear_momentum);

            return aij[idx.x(), idx.y()];
        };

        dat.aij[i] = discretise<float>(ctx, cqueue, func, dim, scale);
    }

    auto cfl = [&](v3i pos, v3i dim, valuef scale)
    {
        v3f world_pos = grid_to_world((v3f)pos, dim, scale);

        return get_conformal_guess(world_pos, (v3f)params.position, (valuef)params.bare_mass);
    };

    dat.conformal_guess = discretise<float>(ctx, cqueue, cfl, dim, scale);

    return dat;
}

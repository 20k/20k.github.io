#include "init_black_hole.hpp"
#include "tensor_algebra.hpp"

using valuef = value<float>;
using v3f = tensor<valuef, 3>;

tensor<valuef, 3, 3> get_aIJ(v3f world_pos, v3f bh_pos, v3f angular_momentum, v3f momentum)
{
    ///todo: fixme
    tensor<valuef, 3, 3, 3> eijk = get_eijk();

    tensor<valuef, 3, 3> bcAij;

    metric<valuef, 3, 3> flat;

    for(int i=0; i < 3; i++)
    {
        flat[i, i] = 1;
    }

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            valuef r = (world_pos - bh_pos).length();

            r = max(r, valuef(1e-6f));

            tensor<valuef, 3> n = (world_pos - bh_pos) / r;

            tensor<valuef, 3> momentum_lo = flat.lower(momentum);
            tensor<valuef, 3> n_lo = flat.lower(n);

            bcAij[i, j] += (3 / (2.f * r * r)) * (momentum_lo[i] * n_lo[j] + momentum_lo[j] * n_lo[i] - (flat[i, j] - n_lo[i] * n_lo[j]) * sum_multiply(momentum, n_lo));

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

            bcAij[i, j] += (3 / (r*r*r)) * (s1 + s2);
        }
    }

    return bcAij;
}

black_hole_data init_black_hole(const black_hole_params& params, float scale, tensor<int, 3> grid_size)
{

}

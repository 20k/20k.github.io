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
            tensor<valuef, 3> momentum_tensor = {momentum.x(), momentum.y(), momentum.z()};

            tensor<valuef, 3> vri = {bh_pos.x(), bh_pos.y(), bh_pos.z()};

            valuef ra = (world_pos - vri).length();

            ra = max(ra, valuef(1e-6f));

            tensor<valuef, 3> nia = (world_pos - vri) / ra;

            tensor<valuef, 3> momentum_lower = lower_index(momentum_tensor, flat, 0);
            tensor<valuef, 3> nia_lower = lower_index(tensor<valuef, 3>{nia.x(), nia.y(), nia.z()}, flat, 0);

            bcAij[i, j] += (3 / (2.f * ra * ra)) * (momentum_lower[i] * nia_lower[j] + momentum_lower[j] * nia_lower[i] - (flat[i, j] - nia_lower[i] * nia_lower[j]) * sum_multiply(momentum_tensor, nia_lower));

            ///spin
            valuef s1 = 0;
            valuef s2 = 0;

            for(int k=0; k < 3; k++)
            {
                for(int l=0; l < 3; l++)
                {
                    s1 += eijk[k, i, l] * angular_momentum[l] * nia[k] * nia_lower[j];
                    s2 += eijk[k, j, l] * angular_momentum[l] * nia[k] * nia_lower[i];
                }
            }

            bcAij[i, j] += (3 / (ra*ra*ra)) * (s1 + s2);
        }
    }

    return bcAij;
}

black_hole_data init_black_hole(const black_hole_params& params, float scale, tensor<int, 3> grid_size)
{

}

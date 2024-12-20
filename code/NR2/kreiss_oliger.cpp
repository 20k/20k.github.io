#include "kreiss_oliger.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include "tensor_algebra.hpp"
#include "derivatives.hpp"
#include "bssn.hpp"

/*valuef diffnth(const valuef& in, int idx, valuei n)
{
    valuef v2 = diff2nd(in, idx);
    valuef v4 = diff4th(in, idx);
    valuef v6 = diff6th(in, idx);

    return ternary(n == 6, v6, ternary(n == 4, v4, v2));
}*/

valuef kreiss_oliger_interior(valuef in, int order)
{
    ///boundary is at 1 and dim - 2
    valuef val = 0;

    for(int i=0; i < 3; i++)
    {
        if(order == 2)
            val += diff2nd(in, i);
        if(order == 4)
            val += diff4th(in, i);
        if(order == 6)
            val += diff6th(in, i);
        if(order == 8)
            val += diff8th(in, i);
        if(order == 10)
            val += diff10th(in, i);
    }

    int n = order;
    float p = n - 1;

    int sign = pow(-1, (p + 3)/2);

    int divisor = pow(2, p+1);

    float prefix = (float)sign / divisor;

    return prefix * val;
}

valuei distance_to_boundary(valuei pos, valuei dim)
{
    ///so. Pos == 0 is out of the question
    ///pos == 1 is the boundary
    ///similarly pos == dim-1 is out of the question
    ///pos == dim-2 is the boundary
    valuei distance_from_left = pos - 1;
    valuei distance_from_right = dim - 2 - pos;

    return min(distance_from_left, distance_from_right);
}

valuei distance_to_boundary(v3i pos, v3i dim)
{
    return min(min(distance_to_boundary(pos[0], dim[0]), distance_to_boundary(pos[1], dim[1])), distance_to_boundary(pos[2], dim[2]));
}

std::string make_kreiss_oliger()
{
     auto func = [&](execution_context&, buffer<valuef> in, buffer_mut<valuef> out, buffer<valuef> W, literal<v3i> ldim, literal<valuef> scale, literal<valuef> eps) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        v3i pos = get_coordinate(lid, dim);

        valuei boundary_distance = distance_to_boundary(pos, dim);

        if_e(boundary_distance == 0, [&] {
            as_ref(out[lid]) = in[lid];

            return_e();
        });

        #define CAKO
        #ifdef CAKO
        auto do_kreiss = [&](int order)
        {
            as_ref(out[lid]) = in[lid] + eps.get() * kreiss_oliger_interior(in[pos, dim], order) * max(W[lid], valuef(0.1));
        };
        #else
        auto do_kreiss = [&](int order)
        {
            as_ref(out[lid]) = in[lid] + eps.get() * kreiss_oliger_interior(in[pos, dim], order);
        };
        #endif

        int max_kreiss = 4;

        for(int i=1; i < max_kreiss; i++)
        {
            if_e(boundary_distance == i, [&] {
                do_kreiss(i * 2);
            });
        }

        if_e(boundary_distance >= max_kreiss, [&] {
            do_kreiss(max_kreiss * 2);
        });

        /*if_e(pos.x() == 2 && pos.y() == 128 && pos.z() == 128, [&]{
            value_base se;
            se.type = value_impl::op::SIDE_EFFECT;
            se.abstract_value = "printf(\"hello %f %i\\n\"," + value_to_string(in[pos, dim]) + ", " + value_to_string(boundary_distance) + ")";

            value_impl::get_context().add(se);
        });*/
     };

     return value_impl::make_function(func, "kreiss_oliger");
}

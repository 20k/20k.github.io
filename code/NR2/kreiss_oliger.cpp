#include "kreiss_oliger.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include "tensor_algebra.hpp"
#include "derivatives.hpp"
#include "bssn.hpp"

valuef diffnth(const valuef& in, int idx, valuei n)
{
    valuef v2 = diff2nd(in, idx);
    valuef v4 = diff4th(in, idx);
    valuef v6 = diff6th(in, idx);

    return ternary(n == 6, v6, ternary(n == 4, v4, v2));
}

valuef kreiss_oliger_interior(valuef in, valuef scale, int order)
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
    }

    int n = order;
    float p = n - 1;

    int sign = pow(-1, (p + 3)/2);

    int divisor = pow(2, p+1);

    float prefix = (float)sign / divisor;

    return (prefix / scale) * val;
}

valuei distance_to_boundary(v3i pos, v3i dim)
{
    using namespace single_source;

    mut<valuei> out = declare_mut_e(valuei(3));

    if_e(pos.x() == 3 || pos.y() == 3 || pos.z() == 3
         || pos.x() == dim.x() - 4 || pos.y() == dim.y() - 4 || pos.z() == dim.z() - 4, [&] {
        as_ref(out) = valuei(2);
    });

    if_e(pos.x() == 2 || pos.y() == 2 || pos.z() == 2
         || pos.x() == dim.x() - 3 || pos.y() == dim.y() - 3 || pos.z() == dim.z() - 3, [&] {
        as_ref(out) = valuei(1);
    });

    if_e(pos.x() <= 1 || pos.y() <= 1 || pos.z() <= 1
         || pos.x() >= dim.x() - 2 || pos.y() >= dim.y() - 2 || pos.z() >= dim.z() - 2, [&] {
        as_ref(out) = valuei(0);
    });

    return declare_e(out);
}

std::string make_kreiss_oliger()
{
     auto func = [&](execution_context&, buffer<valuef> in, buffer_mut<valuef> out, buffer<valuef> W, literal<valuef> timestep, literal<v3i> ldim, literal<valuef> scale, literal<valuef> eps) {
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

        auto do_kreiss = [&](int order)
        {
            as_ref(out[lid]) = in[lid] + eps.get() * timestep.get() * kreiss_oliger_interior(in[pos, dim], scale.get(), order) * max(W[lid], valuef(0.01f));
        };

        if_e(boundary_distance == 1, [&]{
            do_kreiss(2);
        });

        if_e(boundary_distance == 2, [&]{
            do_kreiss(4);
        });

        if_e(boundary_distance == 3, [&]{
            do_kreiss(6);
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

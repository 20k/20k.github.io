#include "kreiss_oliger.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include "tensor_algebra.hpp"
#include "derivatives.hpp"
#include "bssn.hpp"

valuef kreiss_oliger_interior(valuef in, valuef scale)
{
    ///boundary is at 1 and dim - 2
    valuef val = 0;

    for(int i=0; i < 3; i++)
    {
        val += diff6th(in, i);
    }

    int n = 6;
    float p = n - 1;

    int sign = pow(-1, (p + 3)/2);

    int divisor = pow(2, p+1);

    float prefix = (float)sign / divisor;

    return (prefix / scale) * val;
}

std::string make_kreiss_oliger()
{
     auto func = [&](execution_context&, buffer<valuef> in, buffer_mut<valuef> out, literal<valuef> timestep, literal<v3i> ldim, literal<valuef> scale, literal<valuef> eps) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        v3i pos = get_coordinate(lid, dim);

        as_ref(out[lid]) = in[lid] + eps.get() * timestep.get() * kreiss_oliger_interior(in[pos, dim], scale.get());
     };

     return value_impl::make_function(func, "kreiss_oliger");
}

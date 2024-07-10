#include "derivatives.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"

using valuei = value<int>;

template<typename T, int elements = 5>
struct differentiation_context
{
    std::array<T, elements> vars;

    differentiation_context(const T& in, int direction)
    {
        std::array<int, elements> offx = {};
        std::array<int, elements> offy = {};
        std::array<int, elements> offz = {};

        for(int i=0; i < elements; i++)
        {
            int offset = i - (elements - 1)/2;

            if(direction == 0)
                offx[i] = offset;
            if(direction == 1)
                offy[i] = offset;
            if(direction == 2)
                offz[i] = offset;
        }

        ///for each element, ie x-2, x-1, x, x+1, x+2
        for(int i=0; i < elements; i++)
        {
            ///assign to the original element, ie x
            vars[i] = in;

            vars[i].recurse([&i, &offx, &offy, &offz](value_base& v)
            {
                if(v.type == value_impl::op::BRACKET)
                {
                    auto get_substitution = [&i, &offx, &offy, &offz](const value_base& v)
                    {
                        assert(v.args.size() == 8);

                        auto buf = v.args[0];

                        value_base old_x = v.args[2];
                        value_base old_y = v.args[3];
                        value_base old_z = v.args[4];

                        value_base dx = v.args[5];
                        value_base dy = v.args[6];
                        value_base dz = v.args[7];

                        value_base next_x = old_x + valuei(offx[i]);
                        value_base next_y = old_y + valuei(offy[i]);
                        value_base next_z = old_z + valuei(offz[i]);

                        #define PERIODIC_BOUNDARY
                        #ifdef PERIODIC_BOUNDARY
                        if(offx[i] > 0)
                            next_x = ternary(next_x >= dx, next_x - dx, next_x);

                        if(offy[i] > 0)
                            next_y = ternary(next_y >= dy, next_y - dy, next_y);

                        if(offz[i] > 0)
                            next_z = ternary(next_z >= dz, next_z - dz, next_z);

                        if(offx[i] < 0)
                            next_x = ternary(next_x < valuei(0), next_x + dx, next_x);

                        if(offy[i] < 0)
                            next_y = ternary(next_y < valuei(0), next_y + dy, next_y);

                        if(offz[i] < 0)
                            next_z = ternary(next_z < valuei(0), next_z + dz, next_z);
                        #endif // PERIODIC_BOUNDARY

                        value_base op;
                        op.type = value_impl::op::BRACKET;
                        op.args = {buf, value<int>(3), next_x, next_y, next_z, dx, dy, dz};
                        op.concrete = get_interior_type(T());

                        return op;
                    };

                    v = get_substitution(v);
                }
            });
        }
    }
};

valuef diff1(const valuef& val, int direction, const valuef& scale)
{
    ///second order derivatives
    differentiation_context dctx(val, direction);
    std::array<valuef, 5> vars = dctx.vars;

    /*valuef p1 = -vars[4] + vars[0];
    valuef p2 = valuef(8.f) * (vars[3] - vars[1]);

    return (p1 + p2) / (12.f * scale);*/

    return 0.5f * (vars[3] - vars[1]) / scale;
}

///this uses the commutativity of partial derivatives to lopsidedly prefer differentiating dy in the x direction
///as this is better on the memory layout
valuef diff2(const valuef& in, int idx, int idy, const valuef& dx, const valuef& dy, const valuef& scale)
{
    using namespace single_source;

    if(idx < idy)
    {
        ///we must use dy, therefore swap all instances of diff1 in idy -> dy
        alias(diff1(in, idy, scale), dy);

        return diff1(dy, idx, scale);
    }
    else
    {
        ///we must use dx, therefore swap all instances of diff1 in idx -> dx
        alias(diff1(in, idx, scale), dx);

        return diff1(dx, idy, scale);
    }
}

valuef diff6th(const valuef& in, int idx, const valuef& scale)
{
    differentiation_context<valuef, 7> dctx(in, idx);
    auto vars = dctx.vars;

    valuef p1 = vars[0] + vars[6];
    valuef p2 = -6 * (vars[1] + vars[5]);
    valuef p3 = 15 * (vars[2] + vars[4]);
    valuef p4 = -20 * vars[3];

    return (p1 + p2 + p3 + p4);
}

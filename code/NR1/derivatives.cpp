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
        std::array<int, elements> offsets = {};

        for(int i=0; i < elements; i++)
        {
            int offset = i - (elements - 1)/2;

            offsets[i] = offset;
        }

        ///for each element, ie x-2, x-1, x, x+1, x+2
        for(int i=0; i < elements; i++)
        {
            ///assign to the original element, ie x
            vars[i] = in;

            vars[i].recurse([&i, &offsets, direction](value_base& v)
            {
                if(v.type == value_impl::op::BRACKET)
                {
                    auto get_substitution = [&i, &offsets, direction](const value_base& v)
                    {
                        assert(v.args.size() == 8);

                        auto buf = v.args[0];

                        std::array<value_base, 3> pos = {v.args[2], v.args[3], v.args[4]};
                        std::array<value_base, 3> dim = {v.args[5], v.args[6], v.args[7]};

                        pos[direction] = pos[direction] + valuei(offsets[i]);

                        if(offsets[i] > 0)
                            pos[direction] = ternary(pos[direction] >= dim[direction], pos[direction] - dim[direction], pos[direction]);

                        if(offsets[i] < 0)
                            pos[direction] = ternary(pos[direction] < valuei(0), pos[direction] + dim[direction], pos[direction]);

                        value_base op;
                        op.type = value_impl::op::BRACKET;
                        op.args = {buf, value<int>(3), pos[0], pos[1], pos[2], dim[0], dim[1], dim[2]};
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

    valuef p1 = -vars[4] + vars[0];
    valuef p2 = valuef(8.f) * (vars[3] - vars[1]);

    return (p1 + p2) / (12.f * scale);

    //return 0.5f * (vars[3] - vars[1]) / scale;
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

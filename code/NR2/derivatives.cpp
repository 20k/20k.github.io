#include "derivatives.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"

using valuei = value<int>;

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

template<std::size_t elements, typename T>
auto get_differentiation_variables(const T& in, int direction)
{
    std::array<T, elements> vars;

    ///for each element, ie x-2, x-1, x, x+1, x+2
    for(int i=0; i < elements; i++)
    {
        ///assign to the original element, ie x
        vars[i] = in;

        vars[i].recurse([&i, direction](value_base& v)
        {
            if(v.type == value_impl::op::BRACKET)
            {
                auto get_substitution = [&i, direction](const value_base& v)
                {
                    assert(v.args.size() == 8);

                    auto buf = v.args[0];

                    std::array<value_base, 3> pos = {v.args[2], v.args[3], v.args[4]};
                    std::array<value_base, 3> dim = {v.args[5], v.args[6], v.args[7]};

                    int offset = i - (elements - 1)/2;

                    pos[direction] = pos[direction] + valuei(offset);

                    #ifdef PERIODIC
                    if(offset > 0)
                        pos[direction] = ternary(pos[direction] >= dim[direction], pos[direction] - dim[direction], pos[direction]);

                    if(offset < 0)
                        pos[direction] = ternary(pos[direction] < valuei(0), pos[direction] + dim[direction], pos[direction]);
                    #endif

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

    return vars;
}

valuef diff1(const valuef& in, int direction, const derivative_data& d)
{
    valuef second;

    {
        ///4th order derivatives
        std::array<valuef, 5> vars = get_differentiation_variables<5>(in, direction);

        valuef p1 = -vars[4] + vars[0];
        valuef p2 = 8.f * (vars[3] - vars[1]);

        second = (p1 + p2) / (12.f * d.scale);
    }

    valuef first;

    {
        std::array<valuef, 3> vars = get_differentiation_variables<3>(in, direction);

        first = (vars[2] - vars[0]) / (2.f * d.scale);
    }

    valuei width = distance_to_boundary(d.pos[direction], d.dim[direction]);

    return ternary(width >= 2, second, first);
}

///this uses the commutativity of partial derivatives to lopsidedly prefer differentiating dy in the x direction
///as this is better on the memory layout
valuef diff2(const valuef& in, int idx, int idy, const valuef& dx, const valuef& dy, const derivative_data& d)
{
    using namespace single_source;

    if(idx < idy)
    {
        ///we must use dy, therefore swap all instances of diff1 in idy -> dy
        alias(diff1(in, idy, d), dy);

        return diff1(dy, idx, d);
    }
    else
    {
        ///we must use dx, therefore swap all instances of diff1 in idx -> dx
        alias(diff1(in, idx, d), dx);

        return diff1(dx, idy, d);
    }
}

valuef diff2nd(const valuef& in, int idx)
{
    auto vars = get_differentiation_variables<3>(in, idx);

    valuef p1 = vars[0] + vars[2];

    return p1 - 2 * vars[1];
}

valuef diff4th(const valuef& in, int idx)
{
    auto vars = get_differentiation_variables<5>(in, idx);

    valuef p1 = vars[0] + vars[4];
    valuef p2 = -4.f * vars[1] -4.f * vars[3];
    valuef p3 = 6.f * vars[2];

    return p1 + p2 + p3;
}

valuef diff6th(const valuef& in, int idx)
{
    auto vars = get_differentiation_variables<7>(in, idx);

    valuef p1 = vars[0] + vars[6];
    valuef p2 = -6 * (vars[1] + vars[5]);
    valuef p3 = 15 * (vars[2] + vars[4]);
    valuef p4 = -20 * vars[3];

    return p1 + p2 + p3 + p4;
}

valuef diff8th(const valuef& in, int idx)
{
    auto vars = get_differentiation_variables<9>(in, idx);

    valuef p1 = vars[0] + vars[8];
    valuef p2 = -8.f * vars[1] - 8.f * vars[7];
    valuef p3 = 28.f * vars[2] + 28.f * vars[6];
    valuef p4 = -56.f * vars[3] - 56.f * vars[5];
    valuef p5 = 70.f * vars[4];

    return p1 + p2 + p3 + p4 + p5;
}

valuef diff10th(const valuef& in, int idx)
{
    auto vars = get_differentiation_variables<11>(in, idx);

    valuef p1 = vars[0] + vars[10];
    valuef p2 = -10.f * vars[1] - 10.f * vars[9];
    valuef p3 = 45.f * vars[2] + 45.f * vars[8];
    valuef p4 = -120.f * vars[3] - 120.f * vars[7];
    valuef p5 = 210.f * vars[4] + 210.f * vars[6];
    valuef p6 = -252.f * vars[5];

    return p1 + p2 + p3 + p4 + p5 + p6;
}

valuef diff1_boundary(single_source::buffer<valuef> in, int direction, const valuef& scale, v3i pos, v3i dim)
{
    using namespace single_source;

    v3i offset;
    offset[direction] = 1;

    derivative_data d;
    d.scale = scale;
    d.pos = pos;
    d.dim = dim;

    ///ok so. If we're at the boundary, do one sided derivatives
    ///otherwise shell out to diff1, which will also handle near boundary points correctly
    mut<valuef> val = declare_mut_e(valuef(0.f));

    value<bool> left = pos[direction] == 1;
    value<bool> right = pos[direction] == dim[direction] - 2;

    if_e(left, [&]{
        as_ref(val) = (-3.f * in[pos, dim] + 4.f * in[pos + offset, dim] - in[pos + 2 * offset, dim]) / (2.f * scale);
    });

    if_e(right, [&] {
        as_ref(val) = (3.f * in[pos, dim] - 4.f * in[pos - offset, dim] + in[pos - 2 * offset, dim]) / (2.f * scale);
    });

    if_e(!(left || right), [&]{
        as_ref(val) = diff1(in[pos, dim], direction, d);
    });

    return declare_e(val);
}

valuef diff1_boundary(const valuef& in, int direction, const derivative_data& d)
{
    /*valuef second;

    {
        ///4th order derivatives
        std::array<valuef, 5> vars = get_differentiation_variables<5>(in, direction);

        valuef p1 = -vars[4] + vars[0];
        valuef p2 = 8.f * (vars[3] - vars[1]);

        second = (p1 + p2) / (12.f * d.scale);
    }

    valuef first;

    {
        std::array<valuef, 3> vars = get_differentiation_variables<3>(in, direction);

        first = (vars[2] - vars[0]) / (2.f * d.scale);
    }

    valuei width = distance_to_boundary(d.pos[direction], d.dim[direction]);

    return ternary(width >= 2, second, first);*/

    using namespace single_source;

    mut<valuef> val = declare_mut_e(valuef(0.f));

    value<bool> left = d.pos[direction] == 1;
    value<bool> right = d.pos[direction] == d.dim[direction] - 2;

    std::array<valuef, 5> vars = get_differentiation_variables<5>(in, direction);

    if_e(left, [&]{
        as_ref(val) = (-3.f * vars[2] + 4 * vars[3] - vars[4]) / (2 * d.scale);
    });

    if_e(right, [&] {
        as_ref(val) = (3.f * vars[2] - 4 * vars[1] + vars[0]) / (2 * d.scale);
    });

    if_e(!(left || right), [&]{
        as_ref(val) = diff1(in, direction, d);
    });

    return declare_e(val);
}

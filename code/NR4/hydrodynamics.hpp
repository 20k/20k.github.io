#ifndef HYDRODYNAMICS_HPP_INCLUDED
#define HYDRODYNAMICS_HPP_INCLUDED

#include <toolkit/opencl.hpp>
#include "../common/single_source.hpp"
#include <array>
#include "plugin.hpp"

///so. We have to sort out the classic problem, which is that
///we need to triple buffer the hydrodynamic state, and solve in lockstep with the classic bssn equation
///while also using backwards euler correctly. I'm going to copy the plugin architecture from the other project
struct hydrodynamic_args : adm_args_mem
{
    /*cl::buffer p_star;
    cl::buffer e_star;
    std::array<cl::buffer, 3> Si;

    hydrodynamic_buffers(cl::context ctx) : p_star(ctx), e_star(ctx), Si{ctx, ctx, ctx}{}*/

    buffer<valuef> p_star;
    buffer<valuef> e_star;
    std::array<buffer<valuef>, 3> Si;

    virtual void build(value_impl::type_storage& in) override
    {
        using namespace value_impl::builder;

        add(p_star, in);
        add(e_star, in);
        add(Si, in);
    }

    virtual void add_adm_S(bssn_args& args, valuef& in) override
    {

    }
};

struct hydrodynamic_buffers : buffer_provider
{
    cl::buffer p_star;
    cl::buffer e_star;

    std::array<cl::buffer, 3> Si;

    hydrodynamic_buffers(cl::context ctx) : p_star(ctx), e_star(ctx), Si{ctx, ctx, ctx}
    {

    }

    std::vector<cl::buffer> buffers;

    virtual std::vector<buffer_descriptor> get_description() override
    {
        buffer_descriptor p;
        p.name = "p*";

        buffer_descriptor e;
        e.name = "e*";

        buffer_descriptor s0;
        s0.name = "cs0";

        buffer_descriptor s1;
        s1.name = "cs1";

        buffer_descriptor s2;
        s2.name = "cs2";

        return {p, e, s0, s1, s2};
    }

    virtual std::vector<cl::buffer> get_buffers() override
    {
        return {p_star, e_star, Si[0], Si[1], Si[2]};
    }

    virtual void allocate(cl::context ctx, cl::command_queue cqueue, t3i size)
    {
        int64_t cells = int64_t{size.x()} * size.y() * size.z();

        p_star.alloc(sizeof(cl_float) * cells);
        e_star.alloc(sizeof(cl_float) * cells);

        p_star.set_to_zero(cqueue);
        e_star.set_to_zero(cqueue);

        for(auto& i : Si)
        {
            i.alloc(sizeof(cl_float) * cells);
            i.set_to_zero(cqueue);
        }
    }
};

#endif // HYDRODYNAMICS_HPP_INCLUDED

#ifndef HYDRODYNAMICS_HPP_INCLUDED
#define HYDRODYNAMICS_HPP_INCLUDED

#include <toolkit/opencl.hpp>
#include "../common/single_source.hpp"
#include <array>
#include "plugin.hpp"

///so. We have to sort out the classic problem, which is that
///we need to triple buffer the hydrodynamic state, and solve in lockstep with the classic bssn equation
///while also using backwards euler correctly. I'm going to copy the plugin architecture from the other project
template<typename T>
struct hydrodynamic_args : adm_args_mem
{
    T p_star;
    T e_star;
    std::array<T, 3> Si;

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

    virtual std::vector<buffer_descriptor> get_description() override;
    virtual std::vector<cl::buffer> get_buffers() override;
    virtual void allocate(cl::context ctx, cl::command_queue cqueue, t3i size) override;
};

struct hydrodynamic_plugin : plugin
{
    hydrodynamic_plugin(cl::context ctx);

    virtual buffer_provider* get_buffer_factory(cl::context ctx) override;
    virtual void init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, buffer_provider* to_init) override;
};

#endif // HYDRODYNAMICS_HPP_INCLUDED

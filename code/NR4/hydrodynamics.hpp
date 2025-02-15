#ifndef HYDRODYNAMICS_HPP_INCLUDED
#define HYDRODYNAMICS_HPP_INCLUDED

#include <toolkit/opencl.hpp>
#include "../common/single_source.hpp"
#include <array>
#include "plugin.hpp"

///so. We have to sort out the classic problem, which is that
///we need to triple buffer the hydrodynamic state, and solve in lockstep with the classic bssn equation
///while also using backwards euler correctly. I'm going to copy the plugin architecture from the other project
///this is templated for buffer_mut vs buffer
template<typename T>
struct hydrodynamic_base_args : value_impl::single_source::argument_pack
{
    T p_star;
    T e_star;
    std::array<T, 3> Si;

    void build(value_impl::type_storage& in)
    {
        using namespace value_impl::builder;

        add(p_star, in);
        add(e_star, in);
        add(Si, in);
    }
};

template<typename T>
struct hydrodynamic_utility_args : value_impl::single_source::argument_pack
{
    T w;
    T P;
    std::array<T, 3> vi;

    void build(value_impl::type_storage& in)
    {
        using namespace value_impl::builder;

        add(w, in);
        add(P, in);
        add(vi, in);
    }
};

template<typename T>
struct full_hydrodynamic_args : adm_args_mem
{
    T p_star;
    T e_star;
    std::array<T, 3> Si;
    T w;
    T P;
    std::array<T, 3> vi;

    virtual void build(value_impl::type_storage& in) override
    {
        using namespace value_impl::builder;

        add(p_star, in);
        add(e_star, in);
        add(Si, in);
        add(w, in);
        add(P, in);
        add(vi, in);
    }

    ///hmm. the issue if we have buffer<valuef> is that we need a position and a dim
    virtual valuef adm_p(bssn_args& args, const derivative_data& d) override;
    virtual tensor<valuef, 3> adm_Si(bssn_args& args, const derivative_data& d) override;
    virtual tensor<valuef, 3, 3> adm_W2_Sij(bssn_args& args, const derivative_data& d) override;
    virtual valuef dbg(bssn_args& args, const derivative_data& d) override;
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

struct hydrodynamic_utility_buffers : buffer_provider
{
    cl::buffer w;
    cl::buffer P;
    std::array<cl::buffer, 3> vi;

    hydrodynamic_utility_buffers(cl::context ctx) : w(ctx), P(ctx), vi{ctx, ctx, ctx}{}

    virtual std::vector<buffer_descriptor> get_description() override;
    virtual std::vector<cl::buffer> get_buffers() override;
    virtual void allocate(cl::context ctx, cl::command_queue cqueue, t3i size) override;
};

struct hydrodynamic_plugin : plugin
{
    hydrodynamic_plugin(cl::context ctx);

    ///we get three copies of these
    virtual buffer_provider* get_buffer_factory(cl::context ctx) override;
    ///you only get one copy of this
    virtual buffer_provider* get_utility_buffer_factory(cl::context ctx) override;
    virtual void init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, buffer_provider* to_init, buffer_provider* to_init_utility) override;
    virtual void step(cl::context ctx, cl::command_queue cqueue, const plugin_step_data& sdata) override;

    virtual void add_args_provider(all_adm_args_mem& mem) override;
};

#endif // HYDRODYNAMICS_HPP_INCLUDED

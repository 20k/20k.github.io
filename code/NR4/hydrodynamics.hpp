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
struct hydrodynamic_args : adm_args_mem
{
    T p_star;
    T e_star;
    std::array<T, 3> Si;

    ///so, ideally we'd have a w and a P in here
    ///do I just establish a convention?
    ///hang on. I am in charge of this, i just get a buffer pack to mess with

    T w;
    T P;

    virtual void build(value_impl::type_storage& in) override
    {
        using namespace value_impl::builder;

        add(p_star, in);
        add(e_star, in);
        add(Si, in);
        add(w, in);
        add(P, in);
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

struct hydrodynamic_utility_buffers : buffer_provider
{
    cl::buffer w;
    cl::buffer P;

    hydrodynamic_utility_buffers(cl::context ctx) : w(ctx), P(ctx){}

    virtual std::vector<buffer_descriptor> get_description() override;
    virtual std::vector<cl::buffer> get_buffers() override;
    virtual void allocate(cl::context ctx, cl::command_queue cqueue, t3i size) override;
};

/*struct hydrodynamic_adm : adm_args_mem
{
    virtual void add_adm_S(bssn_args& args, valuef& in) override;
    virtual void add_adm_p(bssn_args& args, valuef& in) override;
    virtual void add_adm_Si(bssn_args& args, tensor<valuef, 3>& in) override;
    virtual void add_adm_W2_Sij(bssn_args& args, tensor<valuef, 3, 3>& in) override;
};*/

struct hydrodynamic_plugin : plugin
{
    hydrodynamic_plugin(cl::context ctx);

    ///we get three copies of these
    virtual buffer_provider* get_buffer_factory(cl::context ctx) override;
    ///you only get one copy of this
    virtual buffer_provider* get_utility_buffer_factory(cl::context ctx) override;
    virtual void init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, buffer_provider* to_init, buffer_provider* to_init_utility) override;
};

#endif // HYDRODYNAMICS_HPP_INCLUDED

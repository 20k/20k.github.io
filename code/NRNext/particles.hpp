#ifndef PARTICLES_HPP_INCLUDED
#define PARTICLES_HPP_INCLUDED

#include <toolkit/opencl.hpp>
#include "../common/single_source.hpp"
#include <array>
#include "plugin.hpp"

//https://arxiv.org/abs/2404.03722 - star cluster
template<typename T>
struct particle_base_args : virtual value_impl::single_source::argument_pack
{
    std::array<T, 3> pos;
    std::array<T, 3> vel;
    T mass;

    v3f get_pos(valuei idx)
    {
        return {pos[0][idx], pos[1][idx], pos[2][idx]};
    }

    v3f get_vel(valuei idx)
    {
        return {vel[0][idx], vel[1][idx], vel[2][idx]};
    }

    valuef get_mass(valuei idx)
    {
        return mass[idx];
    }

    void build(value_impl::type_storage& in)
    {
        using namespace value_impl::builder;

        add(pos);
        add(vel);
    }
};

template<typename T>
struct particle_utility : virtual value_impl::single_source::argument_pack
{

};

struct particle_buffers : buffer_provider
{
    std::array<cl::buffer, 3> pos;
    std::array<cl::buffer, 3> vel;
    cl::buffer mass;

    particle_buffers(cl::context ctx) : pos{ctx, ctx, ctx}, vel{ctx, ctx, ctx}, mass{ctx}{}

    virtual std::vector<buffer_descriptor> get_description() override;
    virtual std::vector<cl::buffer> get_buffers() override;
    virtual void allocate(cl::context ctx, cl::command_queue cqueue, t3i size) override;
};

struct particle_utility_buffers :  buffer_provider
{

};

template<typename T>
struct full_particle_args : adm_args_mem, particle_base_args<T>, particle_utility<T>
{
    virtual void build(value_impl::type_storage& in) override
    {
        using namespace value_impl::builder;

        hydrodynamic_base_args<T>::build(in);
        hydrodynamic_utility_args<T>::build(in);
    }

    virtual valuef adm_p(bssn_args& args, const derivative_data& d) override;
    virtual tensor<valuef, 3> adm_Si(bssn_args& args, const derivative_data& d) override;
    virtual tensor<valuef, 3, 3> adm_W2_Sij(bssn_args& args, const derivative_data& d) override;
};

struct particle_plugin : plugin
{
    particle_plugin(cl::context ctx);

    ///we get three copies of these
    virtual buffer_provider* get_buffer_factory(cl::context ctx) override;
    ///you only get one copy of this
    virtual buffer_provider* get_utility_buffer_factory(cl::context ctx) override;
    virtual void init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, cl::buffer u, buffer_provider* to_init, buffer_provider* to_init_utility) override;
    virtual void step(cl::context ctx, cl::command_queue cqueue, const plugin_step_data& sdata) override;
    virtual void finalise(cl::context ctx, cl::command_queue cqueue, const finalise_data& sdata) override;

    virtual void add_args_provider(all_adm_args_mem& mem) override;
};

#endif // PARTICLES_HPP_INCLUDED

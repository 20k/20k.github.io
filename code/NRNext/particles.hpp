#ifndef PARTICLES_HPP_INCLUDED
#define PARTICLES_HPP_INCLUDED

#include <toolkit/opencl.hpp>
#include "../common/single_source.hpp"
#include <array>
#include "plugin.hpp"

//https://arxiv.org/abs/2404.03722 - star cluster
//https://arxiv.org/pdf/1208.3927.pdf - eom

//relevant resources, copied over from old project
//https://arxiv.org/pdf/1611.07906.pdf 16
//https://artscimedia.case.edu/wp-content/uploads/sites/213/2018/08/18010345/Mertens_SestoGR18.pdf
//https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses 3.81
//https://einsteinrelativelyeasy.com/index.php/fr/einstein/9-general-relativity/78-the-energy-momentum-tensor
//https://arxiv.org/pdf/1905.08890.pdf
//https://en.wikipedia.org/wiki/Stress%E2%80%93energy_tensor#Stress%E2%80%93energy_in_special_situations
//https://arxiv.org/pdf/1904.07841.pdf so, have to discretise the dirac delta. This paper gives explicit version

/**
Old design: In the original particle dynamics design
Each particle loops over its dirac discretisation, and then accumulates the fact that it exists there
Then, for every cell, I loop over the constituents and sum them
Suboptimal parts:
1. The giant linear memory allocator, requires huge amounts of memory. Allocation itself was fast
Solution: Could do it in chunks?
Solution: Fixed point accumulation? Removes the need for allocation entirely
2. Particles iterating over dirac discretisation is slow, because its entirely random
Solution: GPU sorting

Possible combo solution: Imagine we divvy up the larger cube into 8 smaller chunks, and assign particles. Then do that recursively
hmm. not the best memory locality. Could use that to do sorting though?
In fact, could do the memory allocator technique for sorting particles into cubes, then dirac delta iterate *afterwards*
hmmmm that has a perf factor of ~25x better than the older particle technique, which i like. we could directly iterate over the cells and accumulate fixed point
or could write out and then do another pass to accumulate in flops at the expense of memory

bam, sold to the highest bidder

Integration: Verlet? Is there any need for an implicit integrator? I could

1. Step forwards in time verlet style
2. Interpolate

Or

1. Unconditionally step in lockstep

#2 has the benefit of absolute correctness. #1 has the benefit of perf. But the perf tradeoff may already have been semi solved

*/
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
        add(mass);
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

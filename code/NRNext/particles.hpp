#ifndef PARTICLES_HPP_INCLUDED
#define PARTICLES_HPP_INCLUDED

#include <toolkit/opencl.hpp>
#include "../common/single_source.hpp"
#include <array>
#include "plugin.hpp"
#include "value_alias.hpp"
#include "bssn.hpp"

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

struct particle_data;
struct discretised_initial_data;

void dirac_test();
//todo: split out into particles_init.cpp? only if sufficiently complicated
void particle_initial_conditions(cl::context& ctx, cl::command_queue& cqueue, discretised_initial_data& to_fill, particle_data& data,t3i dim, float scale);

struct particle_params
{
    std::array<std::vector<float>, 3> positions;
    std::array<std::vector<float>, 3> velocities;
    std::vector<float> masses;
    double total_mass = 0;

    int64_t size() const
    {
        return positions[0].size();
    }

    void add(t3f position, t3f velocity, float mass)
    {
        for(int i=0; i < 3; i++)
        {
            positions[i].push_back(position[i]);
            velocities[i].push_back(velocity[i]);
        }

        masses.push_back(mass);
        total_mass += mass;
    }
};

struct particle_data
{
    double total_mass = 0;
    int64_t count = 0;
    std::array<cl::buffer, 3> positions;
    std::array<cl::buffer, 3> velocities;
    cl::buffer masses;
    cl::buffer lorentzs;

    particle_data(cl::context& ctx) : positions{ctx, ctx, ctx}, velocities{ctx, ctx, ctx}, masses(ctx), lorentzs(ctx){}

    void add(cl::command_queue& cqueue, const particle_params& params)
    {
        total_mass = params.total_mass;
        count = params.size();

        for(int i=0; i < 3; i++)
        {
            positions[i].alloc(sizeof(cl_float) * params.positions[i].size());
            velocities[i].alloc(sizeof(cl_float) * params.velocities[i].size());

            positions[i].write(cqueue, params.positions[i]);
            velocities[i].write(cqueue, params.velocities[i]);
        }

        masses.alloc(sizeof(cl_float) * params.masses.size());
        masses.write(cqueue, params.masses);

        lorentzs.alloc(sizeof(cl_float) * params.masses.size());
        lorentzs.fill(cqueue, cl_float{1});
    }
};

template<typename T>
struct particle_base_args : virtual value_impl::single_source::argument_pack
{
    std::array<T, 3> positions;
    std::array<T, 3> velocities;
    T masses;
    T lorentzs;

    v3f get_position(value<size_t> idx)
    {
        return {positions[0][idx], positions[1][idx], positions[2][idx]};
    }

    v3f get_velocity(value<size_t> idx)
    {
        return {velocities[0][idx], velocities[1][idx], velocities[2][idx]};
    }

    valuef get_mass(value<size_t> idx)
    {
        return masses[idx];
    }

    valuef get_lorentz(value<size_t> idx)
    {
        return lorentzs[idx];
    }

    void build(value_impl::type_storage& in)
    {
        using namespace value_impl::builder;

        add(positions, in);
        add(velocities, in);
        add(masses, in);
        add(lorentzs, in);
    }
};

template<typename T>
struct particle_utility_args : virtual value_impl::single_source::argument_pack
{
    T E;
    std::array<T, 3> Si_raised;
    std::array<T, 6> Sij_raised;

    T::value_type get_E(v3i pos, v3i dim)
    {
        return E[pos, dim];
    }

    tensor<typename T::value_type, 3> get_Si(v3i pos, v3i dim)
    {
        return {Si_raised[0][pos, dim], Si_raised[1][pos, dim], Si_raised[2][pos, dim]};
    }

    tensor<typename T::value_type, 3, 3> get_Sij(v3i pos, v3i dim)
    {
        std::array<typename T::value_type, 6> indexed;

        for(int i=0; i < 6; i++)
        {
            indexed[i] = Sij_raised[i][pos, dim];
        }

        return make_symmetry(indexed);
    }

    void build(value_impl::type_storage& in)
    {
        using namespace value_impl::builder;

        add(E, in);
        add(Si_raised, in);
        add(Sij_raised, in);
    }
};

struct particle_buffers : buffer_provider
{
    std::array<cl::buffer, 3> positions;
    std::array<cl::buffer, 3> velocities;
    cl::buffer masses;
    cl::buffer lorentzs;
    uint64_t particle_count = 0;

    particle_buffers(cl::context ctx, uint64_t _particle_count) : positions{ctx, ctx, ctx}, velocities{ctx, ctx, ctx}, masses{ctx}, lorentzs(ctx), particle_count(_particle_count){}

    virtual std::vector<buffer_descriptor> get_description() override;
    virtual std::vector<cl::buffer> get_buffers() override;
    virtual void allocate(cl::context ctx, cl::command_queue cqueue, t3i size) override;
};

struct particle_utility_buffers :  buffer_provider
{
    cl::buffer E;
    std::array<cl::buffer, 3> Si_raised;
    std::array<cl::buffer, 6> Sij_raised;

    particle_utility_buffers(cl::context ctx) : E{ctx}, Si_raised{ctx, ctx, ctx}, Sij_raised{ctx, ctx, ctx, ctx, ctx, ctx}{}

    virtual std::vector<buffer_descriptor> get_description() override;
    virtual std::vector<cl::buffer> get_buffers() override;
    virtual void allocate(cl::context ctx, cl::command_queue cqueue, t3i size) override;
};

template<typename T>
struct full_particle_args : adm_args_mem, particle_base_args<T>, particle_utility_args<T>
{
    virtual void build(value_impl::type_storage& in) override
    {
        using namespace value_impl::builder;

        particle_base_args<T>::build(in);
        particle_utility_args<T>::build(in);
    }

    virtual valuef adm_p(bssn_args& args, const derivative_data& d) override;
    virtual tensor<valuef, 3> adm_Si(bssn_args& args, const derivative_data& d) override;
    virtual tensor<valuef, 3, 3> adm_W2_Sij(bssn_args& args, const derivative_data& d) override;
};

struct particle_plugin : plugin
{
    cl::buffer lorentz_storage;
    std::vector<cl::buffer> particle_temp;

    //sizeof(cl_int)
    cl::buffer memory_allocation_count;
    //sizeof(cl_int) * dim^3
    cl::buffer memory_ptrs;
    //sizeof(cl_int) * dim^3
    cl::buffer memory_counts;

    double total_mass = 0;
    uint64_t particle_count = 0;

    particle_plugin(cl::context ctx, uint64_t _particle_count);

    ///we get three copies of these
    virtual buffer_provider* get_buffer_factory(cl::context ctx) override;
    ///you only get one copy of this
    virtual buffer_provider* get_utility_buffer_factory(cl::context ctx) override;
    virtual void init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, cl::buffer u, buffer_provider* to_init, buffer_provider* to_init_utility) override;
    virtual void step(cl::context ctx, cl::command_queue cqueue, const plugin_step_data& sdata) override;
    virtual void finalise(cl::context ctx, cl::command_queue cqueue, const finalise_data& sdata) override{}

    virtual void add_args_provider(all_adm_args_mem& mem) override;

    virtual ~particle_plugin(){}

    void calculate_intermediates(cl::context ctx, cl::command_queue cqueue, std::vector<cl::buffer> bssn_in, particle_buffers& p_in, particle_utility_buffers& util_out, t3i dim, float scale);
};

#endif // PARTICLES_HPP_INCLUDED

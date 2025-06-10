#ifndef PLUGIN_HPP_INCLUDED
#define PLUGIN_HPP_INCLUDED

#include <vector>
#include <assert.h>
#include <string>
#include <toolkit/opencl.hpp>
#include "value_alias.hpp"
#include "../common/single_source.hpp"

struct mesh;
struct thin_intermediates_pool;
struct buffer_pool;
struct bssn_args;
struct bssn_buffer_pack;
struct initial_pack;
struct derivative_data;

struct buffer_descriptor
{
    std::string name;
    float dissipation_coeff = 0.f;
    float asymptotic_value = 0;
    float wave_speed = 1;
    bool has_boundary_condition = true;
    int dissipation_order = 4;
    bool sommerfeld_enabled = true;
};

struct adm_args_mem : virtual value_impl::single_source::argument_pack
{
    virtual void build(value_impl::type_storage& store){assert(false);}

    ///rest mass
    virtual valuef get_density(bssn_args& args, const derivative_data& d)
    {
        return adm_p(args, d);
    }

    ///rest energy density
    virtual valuef get_energy(bssn_args& args, const derivative_data& d)
    {
        return adm_p(args, d);
    }

    v4f get_4_momentum(bssn_args& args, const derivative_data& d)
    {
        return get_4_velocity(args, d) * get_density(args, d);
    }

    ///velocity: upper indices, affine parameterised, spatial components
    virtual v4f get_4_velocity(bssn_args& args, const derivative_data& d){return {};}

    //todo: needs to be transformed into units of.. rest mass? energy? brightness? help
    virtual v3f get_colour(bssn_args& args, const derivative_data& d){return {};}
    virtual valuef adm_p(bssn_args& args, const derivative_data& d) {return valuef();};
    virtual tensor<valuef, 3> adm_Si(bssn_args& args, const derivative_data& d) {return tensor<valuef, 3>();}
    virtual tensor<valuef, 3, 3> adm_W2_Sij(bssn_args& args, const derivative_data& d) {return tensor<valuef, 3, 3>();}
    virtual valuef dbg(bssn_args& args, const derivative_data& d) {return {};}
};

struct all_adm_args_mem : value_impl::single_source::argument_pack
{
    std::vector<adm_args_mem*> all_mem;

    void build(value_impl::type_storage& in)
    {
        for(auto& i : all_mem)
            i->build(in);
    }

    v4f get_4_velocity(bssn_args& args, const derivative_data& d);
    v4f get_4_momentum(bssn_args& args, const derivative_data& d);
    valuef get_density(bssn_args& args, const derivative_data& d);
    valuef get_energy(bssn_args& args, const derivative_data& d);

    v3f get_colour(bssn_args& args, const derivative_data& d)
    {
        v3f col;

        for(auto& i : all_mem)
            col += i->get_colour(args, d);

        return col;
    }

    valuef adm_p(bssn_args& args, const derivative_data& d) const
    {
        valuef p = 0;

        for(auto& i : all_mem)
            p += i->adm_p(args, d);

        return p;
    }

    ///Si *lower*
    tensor<valuef, 3> adm_Si(bssn_args& args, const derivative_data& d) const
    {
        tensor<valuef, 3> Si;

        for(auto& i : all_mem)
            Si += i->adm_Si(args, d);

        return Si;
    }

    tensor<valuef, 3, 3> adm_W2_Sij(bssn_args& args, const derivative_data& d) const
    {
        tensor<valuef, 3, 3> W2_Sij;

        for(auto& i : all_mem)
            W2_Sij += i->adm_W2_Sij(args, d);

        return W2_Sij;
    }

    valuef dbg(bssn_args& args, const derivative_data& d) const
    {
        valuef dbg;

        for(auto& i : all_mem)
            dbg += i->dbg(args, d);

        return dbg;
    }

    template<typename T>
    void add(T&& mem)
    {
        adm_args_mem* ptr = new T(std::move(mem));
        all_mem.push_back(ptr);
    }

    template<typename T>
    T* get()
    {
        for(adm_args_mem* ptr : all_mem)
        {
            auto cst = dynamic_cast<T*>(ptr);

            if(cst)
                return cst;
        }

        return nullptr;
    }
};

struct buffer_provider
{
    virtual std::vector<buffer_descriptor> get_description(){assert(false);}
    virtual std::vector<cl::buffer> get_buffers() {assert(false);}
    virtual void allocate(cl::context ctx, cl::command_queue cqueue, t3i size){assert(false);}
};

struct plugin_step_data
{
    std::array<buffer_provider*, 3> buffers;
    buffer_provider* utility_buffers;

    std::vector<cl::buffer> base_bssn_buffers;
    std::vector<cl::buffer> bssn_buffers;

    cl_int evolve_length = 0;

    t3i dim;
    float scale = 0;
    float timestep = 0;
    float total_elapsed = 0;

    int in_idx = 0;
    int out_idx = 0;
    int base_idx = 0;
    int iteration = 0;

    plugin_step_data(cl::context ctx){}
};

struct finalise_data
{
    buffer_provider* inout;
    buffer_provider* utility_buffers;

    std::vector<cl::buffer> bssn_buffers;

    cl_int evolve_length = 0;

    t3i dim;
    float scale = 0;
    float timestep = 0;
    float total_elapsed = 0;
};

struct plugin
{
    virtual buffer_provider* get_buffer_factory(cl::context ctx){return nullptr;}
    ///long term: take a buffer pool. we're going to have to ref count manually
    virtual buffer_provider* get_utility_buffer_factory(cl::context ctx){return nullptr;}
    ///pass the discretised state into here
    virtual void init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, cl::buffer u, buffer_provider* to_init, buffer_provider* to_init_utility){assert(false);}
    //virtual void pre_step(mesh& m, cl::context& ctx, cl::command_queue& mqueue, thin_intermediates_pool& pool, buffer_set& buffers, float timestep){}
    virtual void step(cl::context ctx, cl::command_queue cqueue, const plugin_step_data& sdata){assert(false);}
    virtual void finalise(cl::context ctx, cl::command_queue cqueue, const finalise_data& sdata) {}
    virtual void save(cl::command_queue& cqueue, const std::string& directory, buffer_provider* buf){assert(false);}
    virtual void load(cl::command_queue& cqueue, const std::string& directory, buffer_provider* buf){assert(false);}

    virtual void add_args_provider(all_adm_args_mem& mem){};

    virtual ~plugin(){}
};

inline
all_adm_args_mem make_arg_provider(const std::vector<plugin*>& plugins)
{
    all_adm_args_mem ret;

    for(auto p : plugins)
        p->add_args_provider(ret);

    return ret;
}

#endif // PLUGIN_HPP_INCLUDED

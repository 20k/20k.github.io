#ifndef PLUGIN_HPP_INCLUDED
#define PLUGIN_HPP_INCLUDED

#include <vector>
#include <assert.h>
#include <string>
#include <toolkit/opencl.hpp>
#include "value_alias.hpp"

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
};

///hmm. What if we stick a bssn buffer pack in here? or even inherit?
///this project is in a unique position where I don't have to be so
///careless about passing in all-buffers due to the hideousousness of opencl
///lets think about this. We have a bssn_buffer_pack, which contains our named bssn args
///we also have our individual structs for each of our sim parts, eg we'll have a hydro_buffer_pack
///so: problem one. The bssn kernel will have to accept an unlimited number of arguments
///this is because it has to take in hydro components
///taking in tonnes of buffers may be slower than accumulating into the default ones, except a priori for a fact
///the #1 use case is neutron stars, when it is DEFINITELY faster not to do this due to Sij
///for particle dynamics, it may be better to accumulate directly into sij. There's no way to do this generically, except if i use heuristics (?)
///but eh maybe this isn't worth worrying about

///decision: bssn buffers are their own thing. Nobody is going to attach new buffers to the bssn variables
///decision: no plugin can accept adm matter variables in, I think, because its inherently ill formed. If matter mutually interacts, I'll shim it on
///adm variables: composite of N+ buffers. So the adm buffer set will be tricky to bolt on

///I need to pass in the BSSN buffers by *value*, as well as our buffer struct by *value*, or at least by pointer
///make build a virtual fucntion, overload, and add an overloaded push
struct adm_args_mem : virtual value_impl::single_source::argument_pack
{
    virtual void build(value_impl::type_storage& store){assert(false);}

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

    std::vector<cl::buffer> bssn_buffers;

    cl::buffer evolve_points;
    cl_int evolve_length = 0;

    t3i dim;
    float scale = 0;
    float timestep = 0;
    float total_elapsed = 0;

    int in_idx = 0;
    int out_idx = 0;
    int base_idx = 0;

    plugin_step_data(cl::context ctx) : evolve_points(ctx){}
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
    virtual void finalise(cl::context ctx, cl::command_queue cqueue, std::vector<cl::buffer> bssn_buffers, buffer_provider* out, t3i dim, cl::buffer evolve_points, cl_int evolve_length) {}
    //virtual void save(cl::command_queue& cqueue, const std::string& directory){assert(false);}
    //virtual void load(cl::command_queue& cqueue, const std::string& directory){assert(false);}

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

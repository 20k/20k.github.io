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

struct buffer_descriptor
{
    std::string name;
    float dissipation_coeff = 0.f;
    float asymptotic_value = 0;
    float wave_speed = 1;
    bool has_boundary_condition = true;
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
struct adm_args_mem : value_impl::single_source::argument_pack
{
    virtual void build(value_impl::type_storage& store){assert(false);}

    virtual void add_adm_S(bssn_args& args, valuef& in){}
    virtual void add_adm_p(bssn_args& args, valuef& in){}
    virtual void add_adm_Si(bssn_args& args, tensor<valuef, 3>& in){}
    virtual void add_adm_W2_Sij(bssn_args& args, tensor<valuef, 3, 3>& in){}
};

struct all_adm_args_mem : adm_args_mem
{
    std::vector<adm_args_mem*> all_mem;

    virtual void build(value_impl::type_storage& in) override
    {
        for(auto& i : all_mem)
            i->build(in);
    }

    virtual void add_adm_S(bssn_args& args, valuef& in) override
    {
        for(auto& i : all_mem)
            i->add_adm_S(args, in);
    }

    virtual void add_adm_p(bssn_args& args, valuef& in) override
    {
        for(auto& i : all_mem)
            i->add_adm_p(args, in);
    }

    virtual void add_adm_Si(bssn_args& args, tensor<valuef, 3>& in) override
    {
        for(auto& i : all_mem)
            i->add_adm_Si(args, in);
    }

    virtual void add_adm_W2_Sij(bssn_args& args,tensor<valuef, 3, 3>& in) override
    {
        for(auto& i : all_mem)
            i->add_adm_W2_Sij(args, in);
    }

    template<typename T>
    void add(T&& mem)
    {
        adm_args_mem* ptr = new T(std::move(mem));
        all_mem.push_back(ptr);
    }
};

template<all_adm_args_mem& arg>
struct arg_data : value_impl::single_source::argument_pack
{
    all_adm_args_mem mem = arg;

    void build(value_impl::type_storage& in)
    {
        return mem.build(in);
    }
};

struct buffer_provider
{
    virtual std::vector<buffer_descriptor> get_description(){assert(false);}
    virtual std::vector<cl::buffer> get_buffers() {assert(false);}
    virtual void allocate(cl::context ctx, cl::command_queue cqueue, t3i size){assert(false);}
};

struct plugin
{
    virtual buffer_provider* get_buffer_factory(cl::context ctx){return nullptr;}
    ///long term: take a buffer pool. we're going to have to ref count manually
    virtual buffer_provider* get_utility_buffer_factory(cl::context ctx){return nullptr;}
    //virtual std::vector<buffer_descriptor> get_utility_buffers(){return std::vector<buffer_descriptor>();}
    ///pass the discretised state into here
    virtual void init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, buffer_provider* to_init, buffer_provider* to_init_utility){assert(false);}
    //virtual void pre_step(mesh& m, cl::context& ctx, cl::command_queue& mqueue, thin_intermediates_pool& pool, buffer_set& buffers, float timestep){}
    //virtual void step(mesh& m, cl::context& ctx, cl::command_queue& mqueue, buffer_pack& pack, float timestep, int iteration, int max_iteration){assert(false);}
    //virtual void finalise(mesh& m, cl::context& ctx, cl::command_queue& mqueue, float timestep) {}
    //virtual void save(cl::command_queue& cqueue, const std::string& directory){assert(false);}
    //virtual void load(cl::command_queue& cqueue, const std::string& directory){assert(false);}

    virtual void add_args_provider(all_adm_args_mem& mem){};

    virtual ~plugin(){}
};

#endif // PLUGIN_HPP_INCLUDED

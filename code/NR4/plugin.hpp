#ifndef PLUGIN_HPP_INCLUDED
#define PLUGIN_HPP_INCLUDED

#include <vector>
#include <assert.h>
#include <string>
#include <toolkit/opencl.hpp>
#include "value_alias.hpp"

struct bssn_args;
struct mesh;
struct thin_intermediates_pool;
struct buffer_pool;

struct buffer_descriptor
{
    std::string name;
    float dissipation_coeff = 0.f;
    float asymptotic_value = 0;
    float wave_speed = 1;
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
/*struct buffer_set
{
    std::vector<cl::buffer> bufs;
    std::vector<buffer_descriptor> desc;

    std::pair<cl::buffer, buffer_descriptor> lookup_by_name(const std::string& name)
    {
        for(int i=0; i < (int)bufs.size(); i++)
        {
            if(desc[i].name == name)
                return {bufs[i], desc[i]};
        }

        assert(false);
    }

    void append_to(cl::args& args)
    {
        for(auto& i : bufs)
            args.push_back(i);
    }
};*/


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

struct buffer_provider
{
    virtual std::vector<buffer_descriptor> get_description(){assert(false);}
    virtual std::vector<cl::buffer> get_buffers() {assert(false);}
    virtual void allocate(cl::context ctx, cl::command_queue cqueue, t3i size){assert(false);}
};

struct plugin
{
    virtual buffer_provider* get_buffer_factory(cl::context ctx){return nullptr;}
    //virtual std::vector<buffer_descriptor> get_utility_buffers(){return std::vector<buffer_descriptor>();}
    virtual void init(mesh& m, cl::context& ctx, cl::command_queue& cqueue, bssn_args& to_init){assert(false);}
    //virtual void pre_step(mesh& m, cl::context& ctx, cl::command_queue& mqueue, thin_intermediates_pool& pool, buffer_set& buffers, float timestep){}
    //virtual void step(mesh& m, cl::context& ctx, cl::command_queue& mqueue, buffer_pack& pack, float timestep, int iteration, int max_iteration){assert(false);}
    virtual void finalise(mesh& m, cl::context& ctx, cl::command_queue& mqueue, float timestep) {}
    //virtual void save(cl::command_queue& cqueue, const std::string& directory){assert(false);}
    //virtual void load(cl::command_queue& cqueue, const std::string& directory){assert(false);}

    virtual ~plugin(){}
};

/*struct matter_provider
{
    virtual value               calculate_adm_S(equation_context& ctx, standard_arguments& bssn_args) const{return 0;};
    virtual value               calculate_adm_p(equation_context& ctx, standard_arguments& bssn_args) const{return 0;};
    virtual tensor<value, 3, 3> calculate_adm_X_Sij(equation_context& ctx, standard_arguments& bssn_args) const{return {0,0,0,0,0,0,0,0,0};};
    virtual tensor<value, 3>    calculate_adm_Si(equation_context& ctx, standard_arguments& bssn_args) const{return {0,0,0};};

    virtual ~matter_interop(){}
};

struct matter_meta_provider : matter_provider
{
    std::vector<matter_provider*> all;

    virtual value               calculate_adm_S(equation_context& ctx, standard_arguments& bssn_args) const override;
    virtual value               calculate_adm_p(equation_context& ctx, standard_arguments& bssn_args) const override;
    virtual tensor<value, 3, 3> calculate_adm_X_Sij(equation_context& ctx, standard_arguments& bssn_args) const override;
    virtual tensor<value, 3>    calculate_adm_Si(equation_context& ctx, standard_arguments& bssn_args) const override;
};*/

#endif // PLUGIN_HPP_INCLUDED

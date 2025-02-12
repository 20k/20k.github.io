#ifndef PLUGIN_HPP_INCLUDED
#define PLUGIN_HPP_INCLUDED

#include <vector>
#include <assert.h>
#include <string>
#include <toolkit/opencl.hpp>

struct mesh
struct thin_intermediates_pool;
struct buffer_pool;;

struct buffer_descriptor
{
    std::string name;
    float dissipation_coeff = 0.f;
    float asymptotic_value = 0;
    float wave_speed = 1;
};

struct buffer_set
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
};

struct all_buffers
{
    std::array<buffers, 3> all_bufs;
};

struct plugin
{
    virtual std::vector<buffer_descriptor> get_buffers(){return std::vector<buffer_descriptor>();}
    //virtual std::vector<buffer_descriptor> get_utility_buffers(){return std::vector<buffer_descriptor>();}
    virtual void init(mesh& m, cl::context& ctx, cl::command_queue& cqueue, buffer_set& to_init){assert(false);}
    //virtual void pre_step(mesh& m, cl::context& ctx, cl::command_queue& mqueue, thin_intermediates_pool& pool, buffer_set& buffers, float timestep){}
    virtual void step(mesh& m, cl::context& ctx, cl::command_queue& mqueue, buffer_pack& pack, float timestep, int iteration, int max_iteration){assert(false);}
    virtual void finalise(mesh& m, cl::context& ctx, cl::command_queue& mqueue, float timestep) {}
    //virtual void save(cl::command_queue& cqueue, const std::string& directory){assert(false);}
    //virtual void load(cl::command_queue& cqueue, const std::string& directory){assert(false);}

    virtual ~plugin(){}
};

#endif // PLUGIN_HPP_INCLUDED

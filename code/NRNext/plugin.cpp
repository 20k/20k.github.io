#include "plugin.hpp"
#include "bssn.hpp"
#include <toolkit/fs_helpers.hpp>

void buffer_provider::save(cl::command_queue& cqueue, const std::string& directory)
{
    auto bufs = get_buffers();
    auto desc = get_description();

    for(int i=0; i < (int)bufs.size(); i++)
    {
        std::string name = desc.at(i).name;
        std::vector<uint8_t> data = bufs[i].read<uint8_t>(cqueue);

        file::write(directory + name + ".bin", std::string(data.begin(), data.end()), file::mode::BINARY);
    }
}

void buffer_provider::load(cl::command_queue& cqueue, const std::string& directory)
{
    auto bufs = get_buffers();
    auto desc = get_description();

    for(int i=0; i < (int)bufs.size(); i++)
    {
        std::string name = desc.at(i).name;
        std::string data = file::read(directory + name + ".bin", file::mode::BINARY);

        bufs[i].write(cqueue, std::span<char>(data.begin(), data.end()));
    }
}

v4f all_adm_args_mem::get_4_velocity(bssn_args& args, const derivative_data& d)
{
    if(all_mem.size() == 0)
        return {};

    if(all_mem.size() == 1)
        return all_mem[0]->get_4_velocity(args, d);

    v4f momentum = get_4_momentum(args, d);

    return momentum / max(get_density(args, d), 1e-6f);
}

v4f all_adm_args_mem::get_4_momentum(bssn_args& args, const derivative_data& d)
{
    v4f sum;

    for(auto& i : all_mem)
    {
        sum += i->get_4_momentum(args, d);
    }

    return sum;
}

///i'm overcomplicating this
///4 momentum is conserved yes, but that's only relevant when talking about distributed particles
///the total rest mass is just the sum of rest masses, the trick is: is this the quantity I want?
valuef all_adm_args_mem::get_density(bssn_args& args, const derivative_data& d)
{
    valuef sum = 0;

    for(auto& i : all_mem)
        sum += i->get_density(args, d);

    return sum;

    /*metric<valuef, 3, 3> Yij = args.cY / max(args.W*args.W, 1e-3f);

    metric<valuef, 4, 4> met = calculate_real_metric(Yij, args.gA, args.gB);

    valuef momentum = get_4_momentum(args, d);

    //the minus sign is because of the sign convention
    //https://arxiv.org/pdf/1208.3927 see before (7) if you really want a reference
    valuef rest_density = sqrt(max(-met.dot(momentum, momentum), 0.f));

    return rest_density;*/
}

///todo: I am super incredibly not sure on this function
valuef all_adm_args_mem::get_energy(bssn_args& args, const derivative_data& d)
{
    if(all_mem.size() == 0)
        return {};

    if(all_mem.size() == 1)
        return all_mem[0]->get_energy(args, d);

    ///similar, I think the total energy is just the density weighted energies
    valuef sum_energy = 0;
    valuef sum_density = 0;

    for(int i=0; i < (int)all_mem.size(); i++)
    {
        valuef density = all_mem[i]->get_density(args, d);
        valuef energy_density = all_mem[i]->get_energy(args, d);

        sum_energy += density * energy_density;
        sum_density += density;
    }

    return sum_energy / max(sum_density, 1e-6f);
}

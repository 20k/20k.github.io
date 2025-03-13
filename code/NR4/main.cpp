#include <iostream>
#include <toolkit/render_window.hpp>
#include <imgui/misc/freetype/imgui_freetype.h>
#include "bssn.hpp"
#include <vec/tensor.hpp>
#include <toolkit/texture.hpp>
#include <toolkit/fs_helpers.hpp>
#include "errors.hpp"
#include "init.hpp"
#include "kreiss_oliger.hpp"
#include "init_black_hole.hpp"
#include "init_general.hpp"
#include <thread>
#include "raytrace.hpp"
#include <SFML/Graphics/Image.hpp>
#include <GLFW/glfw3.h>
#include "raytrace_init.hpp"
#include "tov.hpp"
#include "laplace.hpp"
#include "init_neutron_star.hpp"
#include "plugin.hpp"
#include "hydrodynamics.hpp"

using t3i = tensor<int, 3>;

float get_scale(float simulation_width, t3i dim)
{
    return simulation_width / (dim.x() - 1);
}

int get_evolve_size_with_boundary(t3i dim, int boundary)
{
    t3i real_evolve_size = dim - (t3i){boundary*2, boundary*2, boundary*2};
    return real_evolve_size.x() * real_evolve_size.y() * real_evolve_size.z();
}

struct mesh
{
    std::vector<plugin*> plugins;
    std::array<bssn_buffer_pack, 3> buffers;
    std::array<std::vector<buffer_provider*>, 3> plugin_buffers;
    std::vector<buffer_provider*> plugin_utility_buffers;
    t3i dim;

    std::vector<cl::buffer> derivatives;

    bool using_momentum_constraint = false;
    std::vector<cl::buffer> momentum_constraint;
    cl::buffer temporary_buffer;
    cl::buffer temporary_single;
    std::vector<double> hamiltonian_error;
    //std::vector<double> Mi_error;
    //std::vector<double> cG_error;

    cl::buffer sommerfeld_points;
    cl_int sommerfeld_length = 0;
    float total_elapsed = 0;
    float simulation_width = 0;

    cl_int evolve_length;
    int valid_derivative_buffer = 0;

    ///strictly only for rendering
    mesh(cl::context& ctx, t3i _dim, float _simulation_width) : buffers{ctx, ctx, ctx}, sommerfeld_points(ctx), temporary_buffer(ctx), temporary_single(ctx)
    {
        dim = _dim;
        simulation_width = _simulation_width;

        #ifdef MOMENTUM_CONSTRAINT_DAMPING
        using_momentum_constraint = true;
        #endif // MOMENTUM_CONSTRAINT_DAMPING
    }

    void calculate_derivatives_for(cl::command_queue cqueue, bssn_buffer_pack& pack, std::vector<cl::buffer>& into)
    {
        float scale = get_scale(simulation_width, dim);

        auto diff = [&](cl::buffer buf, int buffer_idx)
        {
            t3i diff_grid_size = dim - (t3i){2,2,2};
            cl_int diff_total = diff_grid_size.x() * diff_grid_size.y() * diff_grid_size.z();

            cl::args args;
            args.push_back(buf);
            args.push_back(into.at(buffer_idx * 3 + 0));
            args.push_back(into.at(buffer_idx * 3 + 1));
            args.push_back(into.at(buffer_idx * 3 + 2));
            args.push_back(dim);
            args.push_back(scale);
            args.push_back(diff_total);

            cqueue.exec("differentiate", args, {diff_total}, {128});
        };

        std::vector<cl::buffer> to_diff {
            pack.cY[0],
            pack.cY[1],
            pack.cY[2],
            pack.cY[3],
            pack.cY[4],
            pack.cY[5],
            pack.gA,
            pack.gB[0],
            pack.gB[1],
            pack.gB[2],
            pack.W,
        };

        for(int i=0; i < (int)to_diff.size(); i++)
        {
            diff(to_diff[i], i);
        }
    }

    void init(cl::context& ctx, cl::command_queue& cqueue, initial_params& params)
    {
        buffers[0].allocate(dim);

        buffers[0].for_each([&](cl::buffer b){
            cl_float nan = NAN;

            b.fill(cqueue, nan);
        });

        for(plugin* p : plugins)
        {
            plugin_buffers[0].push_back(p->get_buffer_factory(ctx));
            plugin_buffers[1].push_back(p->get_buffer_factory(ctx));
            plugin_buffers[2].push_back(p->get_buffer_factory(ctx));
        }

        {
            auto [found_u, pack] = params.build(ctx, cqueue, simulation_width, buffers[0]);

            std::vector<float> adm_masses = extract_adm_masses(ctx, cqueue, found_u, dim, get_scale(simulation_width, dim), params.params_bh);

            for(float mass : adm_masses)
            {
                printf("Found mass %f\n", mass);
            }

            {
                assert(plugins.size() == plugin_buffers[0].size());

                for(int i=0; i < (int)plugins.size(); i++)
                {
                    buffer_provider* pb = plugin_buffers[0][i];
                    plugin* p = plugins[i];
                    buffer_provider* util = p->get_utility_buffer_factory(ctx);

                    if(pb)
                        pb->allocate(ctx, cqueue, dim);

                    if(util)
                        util->allocate(ctx, cqueue, dim);

                    p->init(ctx, cqueue, buffers[0], pack, found_u, pb, util);

                    plugin_utility_buffers.push_back(util);
                }
            }
        }

        for(int i=1; i < 3; i++)
        {
            buffers[i].allocate(dim);

            buffers[i].for_each([&](cl::buffer b){
                cl_float nan = NAN;

                b.fill(cqueue, nan);
            });

            for(auto& kk : plugin_buffers[i])
            {
                assert(kk);
                kk->allocate(ctx, cqueue, dim);
            }
        }

        if(using_momentum_constraint)
        {
            for(int i=0; i < 3; i++)
            {
                cl::buffer buf(ctx);
                buf.alloc(sizeof(momentum_t::interior_type) * int64_t{dim.x()} * dim.y() * dim.z());
                buf.set_to_zero(cqueue);

                momentum_constraint.push_back(buf);
            }
        }

        for(int i=0; i < 11 * 3; i++)
        {
            cl::buffer buf(ctx);
            buf.alloc(sizeof(derivative_t::interior_type) * int64_t{dim.x()} * dim.y() * dim.z());

            static_assert(std::numeric_limits<derivative_t::interior_type>::has_quiet_NaN);

            auto qnan = std::numeric_limits<derivative_t::interior_type>::quiet_NaN();

            buf.fill(cqueue, qnan);

            derivatives.push_back(buf);
        }

        temporary_buffer.alloc(sizeof(cl_float) * uint64_t{dim.x()} * dim.y() * dim.z());
        temporary_single.alloc(sizeof(cl_ulong));

        std::vector<cl_int3> boundary;

        for(int z=0; z < dim.z(); z++)
        {
            for(int y=0; y < dim.y(); y++)
            {
                for(int x=0; x < dim.x(); x++)
                {
                    if(x == 1 || x == dim.x() - 2 || y == 1 || y == dim.y() - 2 || z == 1 || z == dim.z() - 2)
                    {
                        boundary.push_back({x, y, z});
                    }
                }
            }
        }

        std::cout << "BCOUNT " << boundary.size() << std::endl;

        std::sort(boundary.begin(), boundary.end(), [](auto p1, auto p2)
        {
            return std::tie(p1.s[2], p1.s[1], p1.s[0]) < std::tie(p2.s[2], p2.s[1], p2.s[0]);
        });

        sommerfeld_points.alloc(sizeof(cl_int3) * boundary.size());
        sommerfeld_points.write(cqueue, boundary);

        sommerfeld_length = boundary.size();

        std::vector<cl_short3> evolve;

        for(int z=0; z < dim.z(); z++)
        {
            for(int y=0; y < dim.y(); y++)
            {
                for(int x=0; x < dim.x(); x++)
                {
                    if(x > 1 && x < dim.x() - 2 && y > 1 && y < dim.y() - 2 && z > 1 && z < dim.z() - 2)
                    {
                        evolve.push_back({x, y, z});
                    }
                }
            }
        }

        std::sort(evolve.begin(), evolve.end(), [](auto p1, auto p2)
        {
            return std::tie(p1.s[2], p1.s[1], p1.s[0]) < std::tie(p2.s[2], p2.s[1], p2.s[0]);
        });

        evolve_length = evolve.size();

        temporary_buffer.set_to_zero(cqueue);
        temporary_single.set_to_zero(cqueue);
        calculate_derivatives_for(cqueue, buffers[0], derivatives);;
        valid_derivative_buffer = 0;
    }

    void add_plugin_args(cl::args& args, int pack_idx)
    {
        for(int i=0; i < (int)plugins.size(); i++)
        {
            auto bufs = plugin_buffers[pack_idx][i];
            auto bufs2 = plugin_utility_buffers[i];

            std::vector<cl::buffer> extra;

            if(bufs)
            {
                auto to_add = bufs->get_buffers();

                extra.insert(extra.end(), to_add.begin(), to_add.end());
            }

            if(bufs2)
            {
                auto to_add = bufs2->get_buffers();

                extra.insert(extra.end(), to_add.begin(), to_add.end());
            }

            for(auto& i : extra)
                args.push_back(i);
        }
    }

    void step(cl::context& ctx, cl::command_queue& cqueue, float timestep)
    {
        float scale = get_scale(simulation_width, dim);

        auto kreiss = [&](int in, int out)
        {
            auto kreiss_individual = [&](cl::buffer inb, cl::buffer outb, float eps, int order)
            {
                if(inb.alloc_size == 0)
                    return;

                if(eps == 0)
                {
                    cl::copy(cqueue, inb, outb);
                    return;
                }

                cl::args args;
                args.push_back(inb);
                args.push_back(outb);
                args.push_back(buffers[in].W);
                args.push_back(dim);
                args.push_back(scale);
                args.push_back(eps);

                cqueue.exec("kreiss_oliger" + std::to_string(order), args, {dim.x() * dim.y() * dim.z()}, {128});
            };

            std::vector<cl::buffer> linear_in;
            std::vector<cl::buffer> linear_out;

            buffers[in].for_each([&](cl::buffer b)
            {
                linear_in.push_back(b);
            });

            buffers[out].for_each([&](cl::buffer b)
            {
                linear_out.push_back(b);
            });

            for(int i=0; i < (int)linear_in.size(); i++)
            {
                kreiss_individual(linear_in[i], linear_out[i], 0.05f, 4);
            }

            for(int i=0; i < (int)plugin_buffers[in].size(); i++)
            {
                buffer_provider* buf_in = plugin_buffers[in][i];
                buffer_provider* buf_out = plugin_buffers[out][i];

                std::vector<buffer_descriptor> desc = buf_in->get_description();

                std::vector<cl::buffer> bufs_in = buf_in->get_buffers();
                std::vector<cl::buffer> bufs_out = buf_out->get_buffers();

                for(int kk=0; kk < (int)bufs_in.size(); kk++)
                {
                    kreiss_individual(bufs_in[kk], bufs_out[kk], desc[kk].dissipation_coeff, desc[kk].dissipation_order);
                }
            }

            //std::swap(plugin_buffers[in], plugin_buffers[out]);
        };

        auto enforce_constraints = [&](int idx)
        {
            int evolve_size = get_evolve_size_with_boundary(dim, 1);

            cl::args args;

            for(int i=0; i < 6; i++)
                args.push_back(buffers[idx].cY[i]);
            for(int i=0; i < 6; i++)
                args.push_back(buffers[idx].cA[i]);

            args.push_back(evolve_size);
            args.push_back(dim);

            cqueue.exec("enforce_algebraic_constraints", args, {evolve_size}, {128});
        };

        #if 1
        auto calculate_constraint_errors = [&](int pack_idx)
        {
            auto sum_over = [&](cl::buffer buf)
            {
                temporary_single.set_to_zero(cqueue);

                uint32_t len = dim.x() * dim.y() * dim.z();

                cl::args args;
                args.push_back(temporary_buffer);
                args.push_back(temporary_single);
                args.push_back(len);

                cqueue.exec("sum", args, {dim.x() * dim.y() * dim.z()}, {128});

                int64_t summed = temporary_single.read<int64_t>(cqueue).at(0);
                return (double)summed / pow(10., 8.);
            };

            {
                cl::args args;
                buffers[pack_idx].append_to(args);

                for(auto& i : derivatives)
                    args.push_back(i);

                add_plugin_args(args, pack_idx);

                args.push_back(temporary_buffer);
                args.push_back(dim);
                args.push_back(scale);

                cqueue.exec("calculate_hamiltonian", args, {dim.x() * dim.y() * dim.z()}, {128});
                hamiltonian_error.push_back(sum_over(temporary_buffer));

                #if 0
                double Mi = 0;

                cqueue.exec("calculate_Mi0", args, {dim.x() * dim.y() * dim.z()}, {128});
                Mi += fabs(sum_over(temporary_buffer));

                cqueue.exec("calculate_Mi1", args, {dim.x() * dim.y() * dim.z()}, {128});
                Mi += fabs(sum_over(temporary_buffer));

                cqueue.exec("calculate_Mi2", args, {dim.x() * dim.y() * dim.z()}, {128});
                Mi += fabs(sum_over(temporary_buffer));

                Mi_error.push_back(Mi);

                double cG = 0;

                cqueue.exec("calculate_cGi0", args, {dim.x() * dim.y() * dim.z()}, {128});
                cG += fabs(sum_over(temporary_buffer));

                cqueue.exec("calculate_cGi1", args, {dim.x() * dim.y() * dim.z()}, {128});
                cG += fabs(sum_over(temporary_buffer));

                cqueue.exec("calculate_cGi2", args, {dim.x() * dim.y() * dim.z()}, {128});
                cG += fabs(sum_over(temporary_buffer));

                cG_error.push_back(cG);
                #endif
            }
        };
        #endif

        auto plugin_step = [&](int base_idx, int in_idx, int out_idx)
        {
            for(int kk=0; kk < (int)plugins.size(); kk++)
            {
                plugin_step_data step_data(ctx);

                for(int i=0; i < 3; i++)
                    step_data.buffers[i] =  plugin_buffers[i].at(kk);

                step_data.utility_buffers = plugin_utility_buffers.at(kk);

                buffers[in_idx].for_each([&](cl::buffer in){
                    step_data.bssn_buffers.push_back(in);
                });

                step_data.evolve_length = evolve_length;

                step_data.dim = dim;
                step_data.scale = scale;
                step_data.timestep = timestep;
                step_data.total_elapsed = total_elapsed;

                step_data.in_idx = in_idx;
                step_data.out_idx = out_idx;
                step_data.base_idx = base_idx;

                plugins[kk]->step(ctx, cqueue, step_data);
            }
        };

        auto calculate_momentum_constraint_for = [&](int pack_idx)
        {
            cl::args args;
            buffers[pack_idx].append_to(args);

            add_plugin_args(args, pack_idx);

            for(auto& i : momentum_constraint)
                args.push_back(i);

            args.push_back(dim);
            args.push_back(scale);
            args.push_back(evolve_length);

            cqueue.exec("momentum_constraint", args, {evolve_length}, {128});
        };

        auto evolve_step = [&](int base_idx, int in_idx, int out_idx)
        {
            cl::args args;
            buffers[base_idx].append_to(args);
            buffers[in_idx].append_to(args);
            buffers[out_idx].append_to(args);

            for(auto& i : derivatives)
                args.push_back(i);

            //todo: i could use out_idx here, but only if we re-calculate the pressure at the end of the step,
            //or fully recalculate the pressure and w in "evolve"
            add_plugin_args(args, in_idx);

            if(using_momentum_constraint)
            {
                for(auto& i : momentum_constraint)
                    args.push_back(i);
            }
            else
            {
                args.push_back(nullptr, nullptr, nullptr);
            }

            args.push_back(timestep);
            args.push_back(scale);
            args.push_back(total_elapsed);
            args.push_back(dim);
            args.push_back(evolve_length);

            cqueue.exec("evolve", args, {evolve_length}, {128});
        };

        auto sommerfeld_buffer = [&](cl::buffer base, cl::buffer in, cl::buffer out, float asym, float wave_speed)
        {
            if(base.alloc_size == 0)
                return;

            cl::args args;
            args.push_back(base);
            args.push_back(in);
            args.push_back(out);
            args.push_back(timestep);
            args.push_back(dim);
            args.push_back(scale);
            args.push_back(wave_speed);
            args.push_back(asym);
            args.push_back(sommerfeld_points);
            args.push_back(sommerfeld_length);

            cqueue.exec("sommerfeld", args, {sommerfeld_length}, {128});
        };

        auto sommerfeld_all = [&](int base_idx, int in_idx, int out_idx)
        {
            bssn_buffer_pack& base = buffers[base_idx];
            bssn_buffer_pack& in = buffers[in_idx];
            bssn_buffer_pack& out = buffers[out_idx];

            #define SOMM(name, a, s) sommerfeld_buffer(base.name, in.name, out.name, a, s)

            SOMM(cY[0], 1.f, 1.f);
            SOMM(cY[1], 0.f, 1.f);
            SOMM(cY[2], 0.f, 1.f);
            SOMM(cY[3], 1.f, 1.f);
            SOMM(cY[4], 0.f, 1.f);
            SOMM(cY[5], 1.f, 1.f);

            SOMM(cA[0], 0.f, 1.f);
            SOMM(cA[1], 0.f, 1.f);
            SOMM(cA[2], 0.f, 1.f);
            SOMM(cA[3], 0.f, 1.f);
            SOMM(cA[4], 0.f, 1.f);
            SOMM(cA[5], 0.f, 1.f);

            SOMM(K, 0.f, 1.f);
            SOMM(W, 1.f, 1.f);
            SOMM(cG[0], 0.f, 1.f);
            SOMM(cG[1], 0.f, 1.f);
            SOMM(cG[2], 0.f, 1.f);

            SOMM(gA, 1.f, sqrt(2));
            SOMM(gB[0], 0.f, 1);
            SOMM(gB[1], 0.f, 1);
            SOMM(gB[2], 0.f, 1);

            for(int i=0; i < (int)plugins.size(); i++)
            {
                std::vector<cl::buffer> p_base = plugin_buffers[base_idx].at(i)->get_buffers();
                std::vector<cl::buffer> p_in = plugin_buffers[in_idx].at(i)->get_buffers();
                std::vector<cl::buffer> p_out = plugin_buffers[out_idx].at(i)->get_buffers();

                std::vector<buffer_descriptor> desc = plugin_buffers[in_idx].at(i)->get_description();

                assert(p_base.size() == p_in.size());
                assert(p_out.size() == p_base.size());
                assert(desc.size() == p_base.size());

                for(int kk=0; kk < (int)p_base.size(); kk++)
                {
                    sommerfeld_buffer(p_base[kk], p_in[kk], p_out[kk], desc[kk].asymptotic_value, desc[kk].wave_speed);
                }
            }
        };

        auto substep = [&](int iteration, int base_idx, int in_idx, int out_idx)
        {
            ///this assumes that in_idx == base_idx for iteration 0, so that they are both constraint enforced
            enforce_constraints(in_idx);

            sommerfeld_all(base_idx, in_idx, out_idx);

            calculate_derivatives_for(cqueue, buffers[in_idx], derivatives);

            //#define CALCULATE_CONSTRAINT_ERRORS
            #ifdef CALCULATE_CONSTRAINT_ERRORS
            if(iteration == 0)
                calculate_constraint_errors(in_idx);
            #endif

            plugin_step(base_idx, in_idx, out_idx);

            if(using_momentum_constraint)
                calculate_momentum_constraint_for(in_idx);

            evolve_step(base_idx, in_idx, out_idx);
        };

        int iterations = 3;

        for(int i=0; i < iterations; i++)
        {
            if(i == 0)
                substep(i, 0, 0, 1);
            else
                substep(i, 0, 2, 1);

            ///always swap buffer 1 to buffer 2, which means that buffer 2 becomes our next input
            std::swap(buffers[1], buffers[2]);
            std::swap(plugin_buffers[1], plugin_buffers[2]);
        }

        ///at the end of our iterations, our output is in buffer[2], and we want our result to end up in buffer[0]
        ///for this to work, kreiss must execute over every pixel unconditionally
        kreiss(2, 0);

        for(int kk=0; kk < (int)plugins.size(); kk++)
        {
            plugin* p = plugins[kk];

            std::vector<cl::buffer> bssn_buffers;

            buffers[0].for_each([&](cl::buffer in){
                bssn_buffers.push_back(in);
            });

            p->finalise(ctx, cqueue, bssn_buffers, plugin_buffers[0][kk], dim, evolve_length);
        }

        total_elapsed += timestep;
        valid_derivative_buffer = 2;
    }
};

float get_timestep(float simulation_width, t3i size)
{
    //float timestep_at_base_c = 0.035;

    float ratio_at_base = 30.f/255.f;
    float new_ratio = simulation_width / size.x();

    return 0.035f * (new_ratio / ratio_at_base);
}

#define MIP_LEVELS 10

struct raytrace_manager
{
    std::vector<plugin*> plugins;

    cl::buffer positions;
    cl::buffer velocities;
    cl::buffer results;
    int last_size = 0;
    bool is_snapshotting = false;
    float elapsed_dt = 0;

    float last_dt = 0.f;
    int captured_slices = 0;
    int slices = 120;
    float time_between_snapshots = 2;
    t3i reduced_dim = {101, 101, 101};
    bool capture_4slices = true;

    cl::buffer texture_coordinates;
    cl::buffer zshifts;
    cl::buffer matter_colours;

    cl::buffer gpu_position;

    std::array<cl::buffer, 4> tetrads;

    bool use_colour = false;
    bool use_matter = false;

    std::vector<cl::buffer> Guv_block;
    std::vector<cl::buffer> colour_block;
    cl::buffer density_block; ///absorption
    cl::buffer energy_block; ///emission
    std::vector<cl::buffer> velocity_block; //only spatial components

    raytrace_manager(cl::context& ctx, const std::vector<plugin*>& _plugins,
                     bool _use_colour, bool _use_matter, float _time_between_snapshots) : positions(ctx), velocities(ctx), results(ctx), texture_coordinates(ctx), zshifts(ctx), matter_colours(ctx), gpu_position(ctx), tetrads{ctx, ctx, ctx, ctx}, energy_block(ctx), density_block(ctx)
    {
        plugins = _plugins;
        use_colour = _use_colour;
        use_matter = _use_matter;
        time_between_snapshots = _time_between_snapshots;

        build_raytrace_kernels(ctx, plugins, use_matter, use_colour);
        build_raytrace_init_kernels(ctx);
        gpu_position.alloc(sizeof(cl_float4));

        for(int i=0; i < 4; i++)
            tetrads[i].alloc(sizeof(cl_float4));
    }

    void allocate(cl::context ctx, cl::command_queue cqueue)
    {
        uint64_t mem_size = sizeof(block_precision_t::interior_type) * int64_t{reduced_dim.x()} * reduced_dim.y() * reduced_dim.z() * slices;

        for(int i=0; i < 10; i++)
        {
            Guv_block.emplace_back(ctx);
        }

        for(auto& i : Guv_block)
        {
            i.alloc(mem_size);
            i.set_to_zero(cqueue);
        }

        if(!use_matter)
            return;

        density_block.alloc(mem_size);
        energy_block.alloc(mem_size);

        density_block.set_to_zero(cqueue);
        energy_block.set_to_zero(cqueue);

        for(int i=0; i < 3; i++)
            colour_block.emplace_back(ctx);

        if(use_colour)
        {
            for(auto& i : colour_block)
            {
                i.alloc(mem_size);
                i.set_to_zero(cqueue);
            }
        }

        for(int i=0; i < 4; i++)
            velocity_block.emplace_back(ctx);

        for(auto& i : velocity_block)
        {
            i.alloc(mem_size);
            i.set_to_zero(cqueue);
        }
    }

    void deallocate(cl::context& ctx)
    {
        Guv_block.clear();

        colour_block.clear();
        density_block = cl::buffer(ctx);
        energy_block = cl::buffer(ctx);
        velocity_block.clear();
    }

    ///shower thought: could use a circular buffer here
    void capture_snapshots(cl::context ctx, cl::command_queue cqueue, float dt, mesh& m)
    {
        if(!capture_4slices)
        {
            deallocate(ctx);
            return;
        }

        if(captured_slices == 0)
            allocate(ctx, cqueue);

        elapsed_dt += dt;

        if(elapsed_dt < time_between_snapshots && captured_slices != 0)
            return;

        elapsed_dt -= time_between_snapshots;
        elapsed_dt = std::max(elapsed_dt, 0.f);
        last_dt += time_between_snapshots;

        if(captured_slices >= slices)
            return;

        {
            cl::args args;
            args.push_back(m.dim, reduced_dim);

            m.buffers[0].append_to(args);

            for(auto& i : Guv_block)
                args.push_back(i);

            args.push_back(uint64_t{captured_slices});

            cqueue.exec("bssn_to_guv", args, {reduced_dim.x(), reduced_dim.y(), reduced_dim.z()}, {8,8,1});
        }

        if(use_matter)
        {
            float scale = get_scale(m.simulation_width, m.dim);

            cl::args args;
            args.push_back(m.dim, reduced_dim, scale);

            m.buffers[0].append_to(args);

            args.push_back(uint64_t{captured_slices});

            for(auto& i : velocity_block)
                args.push_back(i);

            args.push_back(density_block);
            args.push_back(energy_block);

            //might be null
            for(auto& i : colour_block)
                args.push_back(i);

            m.add_plugin_args(args, 0);

            cqueue.exec("capture_matter_fields", args, {reduced_dim.x(), reduced_dim.y(), reduced_dim.z()}, {8, 8, 1});
        }

        printf("Captured %i\n", captured_slices);

        captured_slices++;
    }

    void poll_render_resolution(int width, int height)
    {
        if(last_size == width * height)
            return;

        last_size = width * height;
        positions.alloc(width * height * sizeof(cl_float4));
        velocities.alloc(width * height * sizeof(cl_float4));
        results.alloc(width * height * sizeof(cl_int));
        texture_coordinates.alloc(width * height * sizeof(cl_float2));
        zshifts.alloc(width * height * sizeof(cl_float));
        matter_colours.alloc(width * height * sizeof(cl_float4));
    }

    void render3(cl::command_queue& cqueue, tensor<float, 4> camera_pos, quat camera_quat, cl::image& background, cl::gl_rendertexture& screen_tex, float simulation_width, float simulation_extra_width, mesh& m,
                 bool lock_camera_to_slider, bool progress_camera_time)
    {
        tensor<int, 2> screen_size = {screen_tex.size<2>().x(), screen_tex.size<2>().y()};

        float full_scale = get_scale(simulation_width, m.dim);

        texture_coordinates.set_to_zero(cqueue);
        zshifts.set_to_zero(cqueue);
        matter_colours.set_to_zero(cqueue);

        int buf = m.valid_derivative_buffer;

        {
            cl_float3 vel = {0,0,0};

            cl::args args;
            args.push_back(camera_pos, vel, gpu_position,
                           tetrads[0], tetrads[1], tetrads[2], tetrads[3]);

            m.buffers[buf].append_to(args);
            args.push_back(m.dim);
            args.push_back(full_scale);

            cqueue.exec("init_tetrads3", args, {screen_size.x(), screen_size.y()}, {8,8});
        }

        {
            cl_int is_adm = 1;
            int buf = 0;

            cl::args args;
            args.push_back(screen_size);
            args.push_back(positions, velocities);
            args.push_back(tetrads[0], tetrads[1], tetrads[2], tetrads[3]);
            args.push_back(gpu_position, camera_quat.q);
            args.push_back(m.dim, full_scale, is_adm);

            m.buffers[buf].append_to(args);

            cqueue.exec("init_rays_generic", args, {screen_size.x(), screen_size.y()}, {8,8});
        }

        {
            cl::args args;
            args.push_back(screen_size);
            args.push_back(camera_quat.q);
            args.push_back(positions, velocities, results, zshifts, matter_colours);
            args.push_back(m.dim);
            args.push_back(full_scale);
            args.push_back(simulation_extra_width * simulation_width/2.f);

            m.buffers[buf].append_to(args);

            for(auto& i : m.derivatives)
                args.push_back(i);

            m.add_plugin_args(args, buf);

            cqueue.exec("trace3", args, {screen_size.x(), screen_size.y()}, {8,8});
        }

        blit(cqueue, background, screen_tex, simulation_extra_width * simulation_width);
    }

    void render4(cl::command_queue& cqueue, tensor<float, 4> camera_pos, quat camera_quat, cl::image& background, cl::gl_rendertexture& screen_tex, float simulation_width, float simulation_extra_width, mesh& m,
                 bool lock_camera_to_slider, bool progress_camera_time)
    {
        if(Guv_block.size() == 0)
            return;

        float full_scale = get_scale(simulation_width, m.dim);
        float reduced_scale = get_scale(simulation_width, reduced_dim);

        tensor<int, 2> screen_size = {screen_tex.size<2>().x(), screen_tex.size<2>().y()};

        if(!(lock_camera_to_slider || progress_camera_time))
            camera_pos.x() -= time_between_snapshots*2;

        texture_coordinates.set_to_zero(cqueue);
        zshifts.set_to_zero(cqueue);
        matter_colours.set_to_zero(cqueue);

        {
            cl_float3 vel = {0,0,0};

            cl::args args;
            args.push_back(camera_pos, vel, gpu_position,
                           tetrads[0], tetrads[1], tetrads[2], tetrads[3]);

            for(auto& i : Guv_block)
                args.push_back(i);

            args.push_back(reduced_dim);
            args.push_back(reduced_scale);
            args.push_back(last_dt);
            args.push_back(captured_slices);

            cqueue.exec("init_tetrads4", args, {screen_size.x(), screen_size.y()}, {8,8});
        }

        {
            cl_int is_adm = 0;
            int buf = 0;

            cl::args args;
            args.push_back(screen_size);
            args.push_back(positions, velocities);
            args.push_back(tetrads[0], tetrads[1], tetrads[2], tetrads[3]);
            args.push_back(gpu_position, camera_quat.q);
            args.push_back(m.dim, full_scale, is_adm);

            m.buffers[buf].append_to(args);

            cqueue.exec("init_rays_generic", args, {screen_size.x(), screen_size.y()}, {8,8});
        }

        cl::args args;
        args.push_back(screen_size);
        args.push_back(positions, velocities, results, zshifts, matter_colours);
        args.push_back(reduced_dim);
        args.push_back(reduced_scale);
        args.push_back(simulation_extra_width * simulation_width/2.f);
        args.push_back(tetrads[0], tetrads[1], tetrads[2], tetrads[3]);

        for(auto& i : Guv_block)
            args.push_back(i);

        for(auto& i : velocity_block)
            args.push_back(i);

        args.push_back(density_block);
        args.push_back(energy_block);

        for(auto& i : colour_block)
            args.push_back(i);

        args.push_back(last_dt);
        args.push_back(captured_slices);
        args.push_back(time_between_snapshots);

        cqueue.exec("trace4x4", args, {screen_size.x(), screen_size.y()}, {8, 8});

        blit(cqueue, background, screen_tex, simulation_extra_width * simulation_width);
    }

    void blit(cl::command_queue& cqueue, cl::image background, cl::gl_rendertexture& screen_tex, float simulation_width)
    {
        tensor<int, 2> screen_size = {screen_tex.size<2>().x(), screen_tex.size<2>().y()};
        tensor<int, 2> background_size = {background.size<2>().x(), background.size<2>().y()};

        cl::args args;
        args.push_back(screen_size);
        args.push_back(positions, velocities);
        args.push_back(texture_coordinates);
        args.push_back(simulation_width/2.f);

        cqueue.exec("calculate_texture_coordinates", args, {screen_size.x(), screen_size.y()}, {8, 8});

        int mips = MIP_LEVELS;

        cl::args args2;
        args2.push_back(screen_size);
        args2.push_back(positions, velocities, results, zshifts, matter_colours);
        args2.push_back(texture_coordinates);
        args2.push_back(background, screen_tex);
        args2.push_back(background_size);
        args2.push_back(mips);

        cqueue.exec("render", args2, {screen_size.x(), screen_size.y()}, {8,8});
    }
};

cl::image load_mipped_image(sf::Image& img, cl::context& ctx, cl::command_queue& cqueue)
{
    const uint8_t* as_uint8 = reinterpret_cast<const uint8_t*>(img.getPixelsPtr());

    texture_settings bsett;
    bsett.width = img.getSize().x;
    bsett.height = img.getSize().y;
    bsett.is_srgb = false;

    texture opengl_tex;
    opengl_tex.load_from_memory(bsett, &as_uint8[0]);

    int max_mips = floor(log2(std::min(img.getSize().x, img.getSize().y))) + 1;

    max_mips = std::min(max_mips, MIP_LEVELS);

    cl::image image_mipped(ctx);
    image_mipped.alloc((vec3i){img.getSize().x, img.getSize().y, max_mips}, {CL_RGBA, CL_UNORM_INT8}, cl::image_flags::ARRAY);

    ///and all of THIS is to work around a bug in AMDs drivers, where you cant write to a specific array level!
    int swidth = img.getSize().x;
    int sheight = img.getSize().y;

    std::vector<vec<4, cl_uchar>> as_uniform;
    as_uniform.reserve(max_mips * sheight * swidth);

    for(int i=0; i < max_mips; i++)
    {
        std::vector<vec4f> mip = opengl_tex.read(i);

        int cwidth = swidth / pow(2, i);
        int cheight = sheight / pow(2, i);

        assert(cwidth * cheight == mip.size());

        for(int y = 0; y < sheight; y++)
        {
            for(int x=0; x < swidth; x++)
            {
                ///clamp to border
                int lx = std::min(x, cwidth - 1);
                int ly = std::min(y, cheight - 1);

                vec4f in = mip[ly * cwidth + lx];

                in = clamp(in, 0.f, 1.f);

                as_uniform.push_back({in.x() * 255, in.y() * 255, in.z() * 255, in.w() * 255});
            }
        }
    }

    vec<3, size_t> origin = {0, 0, 0};
    vec<3, size_t> region = {swidth, sheight, max_mips};

    image_mipped.write(cqueue, (char*)as_uniform.data(), origin, region);

    return image_mipped;
}

cl::image load_background(cl::context ctx, cl::command_queue cqueue, const std::string& name)
{
    sf::Image background;
    background.loadFromFile(name);

    return load_mipped_image(background, ctx, cqueue);
}

void solve()
{
    tov::parameters param;
    //param.K = 100;
    param.K = 123.641;
    param.Gamma = 2;

    /*return;
    auto results = tov::search_for_adm_mass(1.543, param);

    for(auto& i : results)
    {
        std::cout << "Density " << i << std::endl;
    }

    assert(false);*/

    //kg/m^3

    double paper_p0 = 6.235 * pow(10., 17.);

    //this is in c=g=msol, so you'd need to use make_integration_state()
    //double p0 = 1.28e-3;


    double rmin = 1e-6;

    //integration_state st = make_integration_state(p0, rmin, param);
    tov::integration_state st = tov::make_integration_state_si(paper_p0, rmin, param);

    tov::integration_solution sol = tov::solve_tov(st, param, rmin, 0);

    std::cout << "Solved for " << sol.R_geom() / 1000. << "km " << sol.M_msol << " msols " << std::endl;

    std::vector<double> tov_phi = initial::calculate_tov_phi(sol);
    //assert(false);
}

void mass_radius_curve()
{
    tov::parameters param;
    param.K = 123.641;
    param.Gamma = 2;

    double adm_mass = 1.5;

    double r_approx = adm_mass / 0.06;

    double start_E = adm_mass / ((4./3.) * M_PI * r_approx*r_approx*r_approx);
    double start_P = param.energy_density_to_pressure(start_E);
    double start_density = param.pressure_to_rest_mass_density(start_P);

    double rmin = 1e-6;

    double min_density = start_density / 100;
    double max_density = start_density * 2000;
    int to_check = 20000;

    std::string str = "Mass, Radius\n";

    for(int i=0; i < to_check; i++)
    {
        double frac = (double)i / to_check;

        double test_density = mix(min_density, max_density, frac);

        tov::integration_state next_st = tov::make_integration_state(test_density, rmin, param);
        tov::integration_solution next_sol = tov::solve_tov(next_st, param, rmin, 0.);

        str += std::to_string(next_sol.M_msol) + ", " + std::to_string(next_sol.R_geom()/1000.) + "\n";

        //std::cout << next_sol.M_msol << ", " << next_sol.R_geom() / 1000. << std::endl;
    }

    file::write("data.csv", str, file::mode::TEXT);
}

initial_params get_initial_params()
{
    //#define INSPIRAL
    #ifdef INSPIRAL
    black_hole_params p1;
    p1.bare_mass = 0.483f;
    p1.position = {3.257, 0, 0};
    p1.linear_momentum = {0, 0.133, 0};

    black_hole_params p2;
    p2.bare_mass = 0.483f;
    p2.position = {-3.257, 0, 0};
    p2.linear_momentum = {0, -0.133, 0};

    initial_conditions init(ctx, cqueue, dim);

    init.add(p1);
    init.add(p2);
    #endif // INSPIRAL

    #define NEUTRON_STAR_TEST
    #ifdef NEUTRON_STAR_TEST

    /*//double K = 123.641;
    //double p0_c_kg_m3 = 6.235 * pow(10., 17.);*/

    /*neutron_star::parameters p1;
    p1.position = {-15, 0, 0};
    p1.angular_momentum = {0, 0, 0};
    p1.linear_momentum = {0, -0.05, 0};
    p1.p0_c_kg_m3 = 6.235 * pow(10., 17.);

    neutron_star::parameters p2;
    p2.position = {15, 0, 0};
    p2.angular_momentum = {0, 0, 0};
    p2.linear_momentum = {0, 0.05, 0};
    p2.p0_c_kg_m3 = 6.235 * pow(10., 17.);

    initial_conditions init(ctx, cqueue, dim);

    init.add(p1);
    init.add(p2);*/

    #if 0
    neutron_star::parameters p1;
    p1.position = {0, 0, 0};
    p1.angular_momentum = {0, 0, 1.25};
    p1.linear_momentum = {0, 0, 0};
    p1.p0_c_kg_m3 = 6.235 * pow(10., 17.);
    #endif

    #if 0
    neutron_star::parameters p1;

    /*neutron_star::dimensionless_linear_momentum lin;
    lin.x = 0.1;
    lin.axis = {1, 0, 0};*/

    p1.position = {0, 0, 0};
    p1.angular_momentum.momentum = {0, 0, 1.25};
    //p1.linear_momentum.momentum = {0.25, 0, 0};
    //p1.linear_momentum.dimensionless = lin;
    p1.K.msols = 123.6;
    p1.mass.p0_kg_m3 = 5.91 * pow(10., 17.);

    initial_conditions init(ctx, cqueue, dim);

    init.add(p1);
    #endif // 0

    //#define HEADON_COLLAPSE
    #ifdef HEADON_COLLAPSE
    neutron_star::parameters p1;

    p1.position = {-15, 0, 0};
    p1.linear_momentum.momentum = {0, 0, 0};
    p1.K.msols = 123.6;
    p1.mass.p0_kg_m3 = 5.91 * pow(10., 17.);

    neutron_star::parameters p2;

    p2.position = {15, 0, 0};
    p2.linear_momentum.momentum = {0, 0, 0};
    p2.K.msols = 123.6;
    p2.mass.p0_kg_m3 = 5.91 * pow(10., 17.);

    initial_params init;
    init.dim = {155, 155, 155};
    init.simulation_width = 120;

    init.add(p1);
    init.add(p2);
    #endif

    //#define HEADON_COLLAPSE2
    #ifdef HEADON_COLLAPSE2
    neutron_star::parameters p1;

    p1.position = {-15, 0, 0};
    p1.linear_momentum.momentum = {0, -0.1, 0};
    p1.K.msols = 123.6;
    p1.mass.p0_kg_m3 = 5.91 * pow(10., 17.);

    neutron_star::parameters p2;

    p2.position = {15, 0, 0};
    p2.linear_momentum.momentum = {0, 0.1, 0};
    p2.K.msols = 123.6;
    p2.mass.p0_kg_m3 = 5.91 * pow(10., 17.);

    initial_params init;
    init.dim = {213, 213, 213};
    init.simulation_width = 120;

    init.add(p1);
    init.add(p2);

    init.time_between_snapshots = 3;
    #endif

    //#define INSPIRAL_2
    #ifdef INSPIRAL_2

    float rad_ms_to_s = 1000;

    float real_rad = 1.78 * rad_ms_to_s;

    float d_over_madm = 14.3;
    float m_adm = 2.681; //msols

    double m_to_kg = 1.3466 * std::pow(10., 27.);
    double msol_kg = 1.988416 * std::pow(10., 30.);
    double msol_meters = msol_kg / m_to_kg;

    float distance_between_stars = d_over_madm * m_adm

    ///1476.619635
    //printf("msol %f\n", msol_meters);

    //float separation_km =

    ///hi, you're trying to work out why the neutron stars lose too much energy
    ///its almost certainly kreiss-oliger
    neutron_star::parameters p1;

    //p1.colour = {1, 0, 0};
    p1.position = {-54.6/2, 0, 0};
    ///was 0.23
    p1.linear_momentum.momentum = {0, -0.23, 0};
    p1.K.msols = 123.6;
    p1.mass.p0_kg_m3 = 4.58 * pow(10., 17.);

    neutron_star::parameters p2;

    //p2.colour = {0, 0, 1};
    p2.position = {54.6/2, 0, 0};
    p2.linear_momentum.momentum = {0, 0.23, 0};
    p2.K.msols = 123.6;
    p2.mass.p0_kg_m3 = 4.58 * pow(10., 17.);

    initial_params init;

    init.dim = {155, 155, 155};
    init.simulation_width = 150;

    init.add(p1);
    init.add(p2);

    init.linear_viscosity_timescale = 500;
    init.time_between_snapshots = 20;
    init.lapse_damp_timescale = 20;


    #endif // INSPIRAL_2

    ///maybe the initial conditions are wrong
    #define INSPIRAL
    #ifdef INSPIRAL
    ///hi, you're trying to work out why the neutron stars lose too much energy
    ///its almost certainly kreiss-oliger
    neutron_star::parameters p1;

    float radial_pos = geometric_to_msol(1000 * 54.6/2, 1);

    printf("Radial pos %f\n", radial_pos);

    ///hang on
    ///i'm in units of c=g=msol=1
    ///so 1 unit of distance isn't 1km
    //p1.colour = {1, 0, 0};
    p1.position = {-radial_pos, 0, 0};
    ///was 0.23
    ///0.265 was reasonable
    p1.linear_momentum.momentum = {0, -0.3, 0};
    p1.K.msols = 123.641;
    p1.mass.p0_kg_m3 = 6.235 * pow(10., 17.);

    neutron_star::parameters p2;

    //p2.colour = {0, 0, 1};
    p2.position = {radial_pos, 0, 0};
    p2.linear_momentum.momentum = {0, 0.3, 0};
    p2.K.msols = 123.641;
    p2.mass.p0_kg_m3 = 6.235 * pow(10., 17.);

    initial_params init;

    init.dim = {213, 213, 213};
    init.simulation_width = radial_pos * 6 * 1.4;

    init.add(p1);
    init.add(p2);

    init.linear_viscosity_timescale = 200;
    init.time_between_snapshots = 15;
    init.lapse_damp_timescale = 20;

    #endif

    //#define FUN
    #ifdef FUN
    neutron_star::parameters p1;

    //p1.colour = {1, 1, 1};
    ///mass of 1.5
    p1.position = {-10, 0, 0};
    p1.linear_momentum.momentum = {-0.01, -0.07, 0};
    p1.K.msols = 123.641;
    p1.mass.p0_kg_m3 = 6.235 * pow(10., 17.);

    neutron_star::parameters p2;

    //p2.colour = {10 * 1, 10 * 128/255.f, 0};
    p2.position = {40, 0, 0};
    p2.linear_momentum.momentum = {0, 0.07, 0};
    p2.K.msols = 423.641;
    //p2.mass.p0_kg_m3 = 6.235 * pow(10., 17.);

    neutron_star::param_rest_mass rest;
    rest.mass = 0.5;
    p2.mass.rest_mass = rest;

    initial_params init;

    init.dim = {199, 199, 199};
    init.simulation_width = 180;

    init.add(p1);
    init.add(p2);

    init.linear_viscosity_timescale = 0;
    init.time_between_snapshots = 15;
    init.lapse_damp_timescale = 20;

    #endif

    //#define TURBO_DETONATE
    #ifdef TURBO_DETONATE
    neutron_star::parameters p1;

    //p1.colour = {1, 0, 0};
    p1.position = {-15, 0, 0};
    p1.linear_momentum.momentum = {0, -0.25f, 0};
    p1.K.msols = 123.6;
    p1.mass.p0_kg_m3 = 5.91 * pow(10., 17.);

    neutron_star::parameters p2;

    //p2.colour = {0, 0, 1};
    p2.position = {15, 0, 0};
    p2.linear_momentum.momentum = {0, 0.25f, 0};
    p2.K.msols = 123.6;
    p2.mass.p0_kg_m3 = 5.91 * pow(10., 17.);

    initial_params init;

    init.dim = {177, 177, 177};
    init.simulation_width = 100;
    init.time_between_snapshots = 20;

    init.add(p1);
    init.add(p2);
    #endif

    #if 0
    neutron_star::parameters p1;

    p1.colour = {1, 0, 0};
    p1.position = {-15, 0, 0};
    p1.linear_momentum.momentum = {0, -0.125f*0.5f, 0};
    p1.K.msols = 123.6;
    p1.mass.p0_kg_m3 = 1.91 * pow(10., 17.);

    neutron_star::parameters p2;

    p2.colour = {0, 0, 1};
    p2.position = {15, 0, 0};
    p2.linear_momentum.momentum = {0, 0.125f*0.5f, 0};
    p2.K.msols = 123.6;
    p2.mass.p0_kg_m3 = 1.91 * pow(10., 17.);

    initial_params init;

    ///minimum resolution is currently 210 width / 255 grid res

    init.dim = {155, 155, 155};
    init.simulation_width = 90;

    init.add(p1);
    init.add(p2);
    #endif
    #endif

    #ifdef SINGLE
    black_hole_params p1;
    p1.bare_mass = 0.483f;
    p1.position = {0, 0, 0};
    p1.linear_momentum = {0, 0, 0};

    initial_conditions init(ctx, cqueue, dim);

    init.add(p1);
    #endif

    return init;
}

int main()
{
    solve();
    //mass_radius_curve();

    render_settings sett;
    sett.width = 1280;
    sett.height = 720;
    sett.opencl = true;
    sett.no_double_buffer = false;
    sett.is_srgb = false;
    sett.no_decoration = false;
    sett.viewports = false;

    render_window win(sett, "Geodesics");

    assert(win.clctx);

    cl::context& ctx = win.clctx->ctx;
    std::cout << cl::get_extensions(ctx) << std::endl;

    initial_params params = get_initial_params();
    boot_initial_kernels(ctx);

    t3i dim = params.dim;

    plugin* hydro = new hydrodynamic_plugin(ctx, params.linear_viscosity_timescale, params.hydrodynamics_wants_colour());

    std::vector<plugin*> plugins;
    plugins.push_back(hydro);

    make_derivatives(ctx);
    make_bssn(ctx, plugins, params);
    init_debugging(ctx, plugins);
    make_momentum_constraint(ctx, plugins);
    enforce_algebraic_constraints(ctx);
    make_sommerfeld(ctx);
    make_initial_conditions(ctx);
    init_christoffel(ctx);
    make_kreiss_oliger(ctx);
    make_hamiltonian_error(ctx, plugins);
    make_global_sum(ctx);
    /*make_momentum_error(ctx, 0);
    make_momentum_error(ctx, 1);
    make_momentum_error(ctx, 2);*/
    make_cG_error(ctx, 0);
    make_cG_error(ctx, 1);
    make_cG_error(ctx, 2);

    cl::command_queue cqueue(ctx);

    neutron_star::boot_solver(ctx);

    ImFontAtlas* atlas = ImGui::GetIO().Fonts;
    atlas->FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImFontConfig font_cfg;
    font_cfg.GlyphExtraSpacing = ImVec2(0, 0);
    font_cfg.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImGuiIO& io = ImGui::GetIO();

    io.Fonts->Clear();
    io.Fonts->AddFontFromFileTTF("VeraMono.ttf", 14, &font_cfg);

    float simulation_width = params.simulation_width;

    mesh m(ctx, dim, simulation_width);
    m.plugins = plugins;
    m.init(ctx, cqueue, params);

    printf("Post init\n");

    texture_settings tsett2;
    tsett2.width = sett.width;
    tsett2.height = sett.height;
    tsett2.is_srgb = false;
    tsett2.generate_mipmaps = false;

    texture tex2;
    tex2.load_from_memory(tsett2, nullptr);

    cl::gl_rendertexture screen_tex{ctx};
    screen_tex.create_from_texture(tex2.handle);

    cqueue.block();

    raytrace_manager rt_bssn(ctx, plugins, params.hydrodynamics_wants_colour(), params.hydrodynamics_enabled(), params.time_between_snapshots);

    cl::image background = load_background(ctx, cqueue, "../common/esa.png");

    //build_thread.join();

    printf("Start\n");

    float elapsed_t = 0;
    //float timestep = 0.001f;
    float timestep = get_timestep(simulation_width, dim);
    bool step = false;
    bool running = false;
    bool pause = false;
    float pause_time = 220;
    bool render = false;
    int render_skipping = 4;
    bool render2 = false;
    bool debug_render = true;
    bool lock_camera_to_slider = false;
    bool progress_camera_time = false;
    float render_size_scale = 1;
    float advance_time_mult = 1;

    vec3f camera_pos = {0, 0, -25};;
    quat camera_quat;
    steady_timer frame_time;

    /*vec3f camera_pos = {0, 25, 0};;
    {
        vec3f right = rot_quat({1, 0, 0}, camera_quat);

        quat q;
        q.load_from_axis_angle({right.x(), right.y(), right.z(), M_PI/2});

        camera_quat = q * camera_quat;
    }*/

    float cam_time = 0;
    uint32_t render_frame_idx = 0;

    while(!win.should_close())
    {
        win.poll();

        float ftime_s = frame_time.restart();

        if(progress_camera_time)
            cam_time += ftime_s * advance_time_mult;

        if(!ImGui::GetIO().WantCaptureKeyboard)
        {
            float speed = 0.001;

            if(ImGui::IsKeyDown(GLFW_KEY_LEFT_SHIFT))
                speed = 0.1;

            if(ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL))
                speed = 0.00001;

            if(ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT))
                speed /= 1000;

            if(ImGui::IsKeyDown(GLFW_KEY_Z))
                speed *= 100;

            if(ImGui::IsKeyDown(GLFW_KEY_X))
                speed *= 100;

            if(ImGui::IsKeyPressed(GLFW_KEY_B))
            {
                camera_pos = {0, 0, -100};
            }

            if(ImGui::IsKeyPressed(GLFW_KEY_C))
            {
                camera_pos = {0, 0, 0};
            }

            if(ImGui::IsKeyDown(GLFW_KEY_RIGHT))
            {
                mat3f m = mat3f().ZRot(-M_PI/128);

                quat q;
                q.load_from_matrix(m);

                camera_quat = q * camera_quat;
            }

            if(ImGui::IsKeyDown(GLFW_KEY_LEFT))
            {
                mat3f m = mat3f().ZRot(M_PI/128);

                quat q;
                q.load_from_matrix(m);

                camera_quat = q * camera_quat;
            }

            vec3f up = {0, 0, -1};
            vec3f right = rot_quat({1, 0, 0}, camera_quat);
            vec3f forward_axis = rot_quat({0, 0, 1}, camera_quat);

            if(ImGui::IsKeyDown(GLFW_KEY_DOWN))
            {
                quat q;
                q.load_from_axis_angle({right.x(), right.y(), right.z(), M_PI/128});

                camera_quat = q * camera_quat;
            }

            if(ImGui::IsKeyDown(GLFW_KEY_UP))
            {
                quat q;
                q.load_from_axis_angle({right.x(), right.y(), right.z(), -M_PI/128});

                camera_quat = q * camera_quat;
            }

            vec3f offset = {0,0,0};

            offset += forward_axis * ((ImGui::IsKeyDown(GLFW_KEY_W) - ImGui::IsKeyDown(GLFW_KEY_S)) * speed);
            offset += right * (ImGui::IsKeyDown(GLFW_KEY_D) - ImGui::IsKeyDown(GLFW_KEY_A)) * speed;
            offset += up * (ImGui::IsKeyDown(GLFW_KEY_E) - ImGui::IsKeyDown(GLFW_KEY_Q)) * speed;

            /*camera.y() += offset.x();
            camera.z() += offset.y();
            camera.w() += offset.z();*/

            camera_pos += offset;
        }

        step = false;

        ImGui::Begin("Hi");

        if(ImGui::Button("Step"))
            step = true;

        ImGui::Checkbox("Run", &running);
        ImGui::Checkbox("Pause", &pause);
        ImGui::Checkbox("Render", &render);
        ImGui::Checkbox("Render2", &render2);
        ImGui::Checkbox("Debug Render", &debug_render);
        ImGui::DragFloat("Pause At", &pause_time, 1, 0, 99999999);

        ImGui::Checkbox("Override Camera Time", &lock_camera_to_slider);
        ImGui::Checkbox("Advance Override Camera Time", &progress_camera_time);
        ImGui::SliderFloat("Advance Time Mult", &advance_time_mult, 0.1f, 100.f);

        ///lock to camera, progress camera time
        ImGui::DragFloat("Override Time", &cam_time, 1.f, 0.f, 4000.f);
        ImGui::Checkbox("Capture Render Slices", &rt_bssn.capture_4slices);
        ImGui::SliderInt("Render Skipping", &render_skipping, 1, 32);
        ImGui::SliderFloat("Render Size Scale", &render_size_scale, 0.1f, 10.f);

        step = step || running;

        ImGui::Text("Elapsed %f", elapsed_t);

        #if 1
        std::vector<float> lines;

        for(auto& i : m.hamiltonian_error)
            lines.push_back(i);

        if(lines.size() > 0)
            ImGui::PlotLines("H", lines.data(), lines.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(400, 100));

        #if 0
        std::vector<float> Mis;

        for(auto& i : m.Mi_error)
            Mis.push_back(i);

        if(Mis.size() > 0)
            ImGui::PlotLines("Mi", Mis.data(), Mis.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(400, 100));

        std::vector<float> cgs;

        for(auto& i : m.cG_error)
            cgs.push_back(i);

        if(cgs.size() > 0)
            ImGui::PlotLines("cG", cgs.data(), cgs.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(400, 100));
        #endif
        #endif

        ImGui::End();

        steady_timer t;

        screen_tex.acquire(cqueue);

        if(pause && elapsed_t > pause_time)
            step = false;

        if(step)
        {
            m.step(ctx, cqueue, timestep);
            rt_bssn.capture_snapshots(ctx, cqueue, timestep, m);
        }

        {
            rt_bssn.poll_render_resolution(screen_tex.size<2>().x(), screen_tex.size<2>().y());

            tensor<float, 4> camera4 = {elapsed_t, camera_pos.x(), camera_pos.y(), camera_pos.z()};

            if(lock_camera_to_slider || progress_camera_time)
                camera4.x() = cam_time;

            if(render && (render_frame_idx % render_skipping) == 0)
                rt_bssn.render3(cqueue, camera4, camera_quat, background, screen_tex, simulation_width, render_size_scale, m, lock_camera_to_slider, progress_camera_time);

            if(render2 && (render_frame_idx % render_skipping) == 0)
                rt_bssn.render4(cqueue, camera4, camera_quat, background, screen_tex, simulation_width, render_size_scale, m, lock_camera_to_slider, progress_camera_time);
        }

        if(debug_render)
        {
            float scale = get_scale(simulation_width, dim);

            cl::args args;
            m.buffers[m.valid_derivative_buffer].append_to(args);

            for(auto& i : m.derivatives)
                args.push_back(i);

            m.add_plugin_args(args, m.valid_derivative_buffer);

            args.push_back(dim);
            args.push_back(scale);
            args.push_back(screen_tex);

            cqueue.exec("debug", args, {dim.x() * dim.y() * dim.z()}, {128});
        }

        screen_tex.unacquire(cqueue);

        ImGui::GetBackgroundDrawList()->AddImage((void*)tex2.handle, ImVec2(0,0), ImVec2(tex2.get_size().x(), tex2.get_size().y()));
        //ImGui::GetBackgroundDrawList()->AddImage((void*)tex.handle, ImVec2(0,0), ImVec2(dim.x() * 3, dim.y() * 3));

        win.display();

        std::cout << "T " << t.get_elapsed_time_s() * 1000. << std::endl;

        if(step)
            elapsed_t += timestep;

        render_frame_idx++;
    }
}

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

using t3i = tensor<int, 3>;

float get_scale(float simulation_width, t3i dim)
{
    return simulation_width / (dim.x() - 1);
}

struct mesh
{
    std::array<bssn_buffer_pack, 3> buffers;
    t3i dim;

    std::vector<cl::buffer> derivatives;

    bool using_momentum_constraint = false;
    std::vector<cl::buffer> momentum_constraint;
    //cl::buffer temporary_buffer;
    //cl::buffer temporary_single;
    //std::vector<double> hamiltonian_error;
    //std::vector<double> Mi_error;
    //std::vector<double> cG_error;

    cl::buffer sommerfeld_points;
    cl_int sommerfeld_length = 0;
    float total_elapsed = 0;
    float simulation_width = 0;

    cl::buffer evolve_points;
    cl_int evolve_length;
    int valid_derivative_buffer = 0;

    ///strictly only for rendering
    mesh(cl::context& ctx, t3i _dim, float _simulation_width) : buffers{ctx, ctx, ctx}, sommerfeld_points(ctx), evolve_points(ctx)
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
            cl::args args;
            args.push_back(buf);
            args.push_back(into.at(buffer_idx * 3 + 0));
            args.push_back(into.at(buffer_idx * 3 + 1));
            args.push_back(into.at(buffer_idx * 3 + 2));
            args.push_back(dim);
            args.push_back(scale);

            cqueue.exec("differentiate", args, {dim.x()*dim.y()*dim.z()}, {128});
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

    void init(cl::context& ctx, cl::command_queue& cqueue)
    {
        buffers[0].allocate(dim);

        buffers[0].for_each([&](cl::buffer b){
            cl_float nan = NAN;

            b.fill(cqueue, nan);
        });

        {
            #define INSPIRAL
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
            #else
            black_hole_params p1;
            p1.bare_mass = 0.483f;
            p1.position = {0, 0, 0};
            p1.linear_momentum = {0, 0, 0};

            initial_conditions init(ctx, cqueue, dim);

            init.add(p1);
            #endif

            cl::buffer found_u = init.build(ctx, cqueue, simulation_width, buffers[0]);

            std::vector<float> adm_masses = init.extract_adm_masses(ctx, cqueue, found_u, dim, get_scale(simulation_width, dim));

            for(float mass : adm_masses)
            {
                printf("Found mass %f\n", mass);
            }
        }

        for(int i=1; i < 3; i++)
        {
            buffers[i].allocate(dim);

            buffers[i].for_each([&](cl::buffer b){
                cl_float nan = NAN;

                b.fill(cqueue, nan);
            });
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

            uint16_t hnan = 0b0111110000000001;

            buf.fill(cqueue, hnan);

            derivatives.push_back(buf);
        }

        //temporary_buffer.alloc(sizeof(cl_float) * uint64_t{dim.x()} * dim.y() * dim.z());
        //temporary_single.alloc(sizeof(cl_ulong));

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

        evolve_points.alloc(sizeof(cl_short3) * evolve.size());
        evolve_points.write(cqueue, evolve);

        evolve_length = evolve.size();

        //temporary_buffer.set_to_zero(cqueue);
        //temporary_single.set_to_zero(cqueue);
        calculate_derivatives_for(cqueue, buffers[0], derivatives);;
        valid_derivative_buffer = 0;
    }

    void step(cl::context& ctx, cl::command_queue& cqueue, float timestep)
    {
        float scale = get_scale(simulation_width, dim);

        auto kreiss = [&](int in, int inout)
        {
            std::vector<cl::buffer> linear_base;
            std::vector<cl::buffer> linear_inout;

            buffers[in].for_each([&](cl::buffer b)
            {
                linear_base.push_back(b);
            });

            buffers[inout].for_each([&](cl::buffer b)
            {
                linear_inout.push_back(b);
            });

            for(int i=0; i < (int)linear_base.size(); i++)
            {
                float eps = 0.05f;

                cl::args args;
                args.push_back(linear_base.at(i));
                args.push_back(linear_inout.at(i));
                args.push_back(buffers[in].W);
                args.push_back(dim);
                args.push_back(scale);
                args.push_back(eps);

                cqueue.exec("kreiss_oliger", args, {dim.x() * dim.y() * dim.z()}, {128});
            }
        };

        auto enforce_constraints = [&](int idx)
        {
            cl::args args;

            for(int i=0; i < 6; i++)
                args.push_back(buffers[idx].cY[i]);
            for(int i=0; i < 6; i++)
                args.push_back(buffers[idx].cA[i]);

            args.push_back(dim);

            cqueue.exec("enforce_algebraic_constraints", args, {dim.x() * dim.y() * dim.z()}, {128});
        };

        #if 0
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

                args.push_back(temporary_buffer);
                args.push_back(dim);
                args.push_back(scale);

                cqueue.exec("calculate_hamiltonian", args, {dim.x() * dim.y() * dim.z()}, {128});
                hamiltonian_error.push_back(sum_over(temporary_buffer));

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
            }
        };
        #endif

        auto calculate_momentum_constraint_for = [&](int pack_idx)
        {
            cl::args args;
            buffers[pack_idx].append_to(args);

            for(auto& i : momentum_constraint)
                args.push_back(i);

            args.push_back(dim);
            args.push_back(scale);
            //args.push_back(evolve_points);
            //args.push_back(evolve_length);

            cqueue.exec("momentum_constraint", args, {dim.x() * dim.y() * dim.z()}, {128});
        };

        auto evolve_step = [&](int base_idx, int in_idx, int out_idx)
        {
            cl::args args;
            buffers[base_idx].append_to(args);
            buffers[in_idx].append_to(args);
            buffers[out_idx].append_to(args);

            for(auto& i : derivatives)
                args.push_back(i);

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
            args.push_back(dim);
            args.push_back(scale);
            args.push_back(total_elapsed);
            args.push_back(dim);
            args.push_back(evolve_points);
            args.push_back(evolve_length);

            cqueue.exec("evolve", args, {evolve_length}, {128});
        };

        auto sommerfeld_buffer = [&](cl::buffer base, cl::buffer in, cl::buffer out, float asym, float wave_speed)
        {
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

            if(using_momentum_constraint)
                calculate_momentum_constraint_for(in_idx);

            evolve_step(base_idx, in_idx, out_idx);
        };

        int iterations = 2;

        for(int i=0; i < iterations; i++)
        {
            if(i == 0)
                substep(i, 0, 0, 1);
            else
                substep(i, 0, 2, 1);

            ///always swap buffer 1 to buffer 2, which means that buffer 2 becomes our next input
            std::swap(buffers[1], buffers[2]);
        }

        ///at the end of our iterations, our output is in buffer[2], and we want our result to end up in buffer[0]
        ///for this to work, kreiss must execute over every pixel unconditionally
        kreiss(2, 0);

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
    cl::buffer positions;
    cl::buffer velocities;
    cl::buffer results;
    int last_size = 0;
    bool is_snapshotting = false;
    float elapsed_dt = 0;

    float last_dt = 0.f;
    int captured_slices = 0;
    float time_between_snapshots = 2;
    t3i reduced_dim = {101, 101, 101};
    bool capture_4slices = true;

    cl::buffer texture_coordinates;
    cl::buffer zshifts;

    cl::buffer gpu_position;

    std::array<cl::buffer, 4> tetrads;

    std::vector<cl::buffer> Guv_block;

    raytrace_manager(cl::context& ctx) : positions(ctx), velocities(ctx), results(ctx), texture_coordinates(ctx), zshifts(ctx), gpu_position(ctx), tetrads{ctx, ctx, ctx, ctx}
    {
        build_raytrace_kernels(ctx);
        build_raytrace_init_kernels(ctx);
        gpu_position.alloc(sizeof(cl_float4));

        for(int i=0; i < 4; i++)
            tetrads[i].alloc(sizeof(cl_float4));
    }

    ///shower thought: could use a circular buffer here
    void capture_snapshots(cl::context ctx, cl::command_queue cqueue, float dt, mesh& m)
    {
        if(!capture_4slices)
        {
            for(int i=0; i < 10; i++)
                Guv_block[i] = cl::buffer(ctx);

            return;
        }

        int slices = 120;

        if(captured_slices == 0)
        {
            for(int i=0; i < 10; i++)
                Guv_block.emplace_back(ctx);

            for(auto& i : Guv_block)
            {
                uint64_t array_size = sizeof(block_precision_t::interior_type) * int64_t{reduced_dim.x()} * reduced_dim.y() * reduced_dim.z();

                i.alloc(array_size * slices);
                i.set_to_zero(cqueue);
            }
        }

        elapsed_dt += dt;

        if(elapsed_dt < time_between_snapshots && captured_slices != 0)
            return;

        elapsed_dt -= time_between_snapshots;
        elapsed_dt = std::max(elapsed_dt, 0.f);
        last_dt += time_between_snapshots;

        if(captured_slices >= slices)
            return;

        cl::args args;
        args.push_back(m.dim, reduced_dim);

        m.buffers[0].append_to(args);

        for(auto& i : Guv_block)
            args.push_back(i);

        args.push_back(uint64_t{captured_slices});

        cqueue.exec("bssn_to_guv", args, {reduced_dim.x(), reduced_dim.y(), reduced_dim.z()}, {8,8,1});

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
    }

    void render3(cl::command_queue& cqueue, tensor<float, 4> camera_pos, quat camera_quat, cl::image& background, cl::gl_rendertexture& screen_tex, float simulation_width, mesh& m,
                 bool lock_camera_to_slider, bool progress_camera_time)
    {
        tensor<int, 2> screen_size = {screen_tex.size<2>().x(), screen_tex.size<2>().y()};

        float full_scale = get_scale(simulation_width, m.dim);

        texture_coordinates.set_to_zero(cqueue);
        zshifts.set_to_zero(cqueue);

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
            args.push_back(positions, velocities, results, zshifts);
            args.push_back(m.dim);
            args.push_back(full_scale);

            m.buffers[buf].append_to(args);

            for(auto& i : m.derivatives)
                args.push_back(i);

            cqueue.exec("trace3", args, {screen_size.x(), screen_size.y()}, {8,8});
        }

        blit(cqueue, background, screen_tex);
    }

    void render4(cl::command_queue& cqueue, tensor<float, 4> camera_pos, quat camera_quat, cl::image& background, cl::gl_rendertexture& screen_tex, float simulation_width, mesh& m,
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
        args.push_back(positions, velocities, results, zshifts);
        args.push_back(reduced_dim);
        args.push_back(reduced_scale);
        args.push_back(tetrads[0], tetrads[1], tetrads[2], tetrads[3]);

        for(auto& i : Guv_block)
            args.push_back(i);

        args.push_back(last_dt);
        args.push_back(captured_slices);
        args.push_back(time_between_snapshots);

        cqueue.exec("trace4x4", args, {screen_size.x(), screen_size.y()}, {8, 8});

        blit(cqueue, background, screen_tex);
    }

    void blit(cl::command_queue& cqueue, cl::image background, cl::gl_rendertexture& screen_tex)
    {
        tensor<int, 2> screen_size = {screen_tex.size<2>().x(), screen_tex.size<2>().y()};
        tensor<int, 2> background_size = {background.size<2>().x(), background.size<2>().y()};

        cl::args args;
        args.push_back(screen_size);
        args.push_back(positions, velocities);
        args.push_back(texture_coordinates);

        cqueue.exec("calculate_texture_coordinates", args, {screen_size.x(), screen_size.y()}, {8, 8});

        int mips = MIP_LEVELS;

        cl::args args2;
        args2.push_back(screen_size);
        args2.push_back(positions, velocities, results, zshifts);
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

int main()
{
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

    t3i dim = {213, 213, 213};

    make_derivatives(ctx);
    make_bssn(ctx, dim);
    init_debugging(ctx);
    make_momentum_constraint(ctx);
    enforce_algebraic_constraints(ctx);
    make_sommerfeld(ctx);
    make_initial_conditions(ctx);
    init_christoffel(ctx);
    make_kreiss_oliger(ctx);
    make_hamiltonian_error(ctx);
    make_global_sum(ctx);
    make_momentum_error(ctx, 0);
    make_momentum_error(ctx, 1);
    make_momentum_error(ctx, 2);
    make_cG_error(ctx, 0);
    make_cG_error(ctx, 1);
    make_cG_error(ctx, 2);

    cl::command_queue cqueue(ctx);

    ImFontAtlas* atlas = ImGui::GetIO().Fonts;
    atlas->FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImFontConfig font_cfg;
    font_cfg.GlyphExtraSpacing = ImVec2(0, 0);
    font_cfg.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImGuiIO& io = ImGui::GetIO();

    io.Fonts->Clear();
    io.Fonts->AddFontFromFileTTF("VeraMono.ttf", 14, &font_cfg);

    float simulation_width = 30;

    mesh m(ctx, dim, simulation_width);
    m.init(ctx, cqueue);

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

    raytrace_manager rt_bssn(ctx);

    cl::image background = load_background(ctx, cqueue, "../common/esa.png");

    //build_thread.join();

    printf("Start\n");

    float elapsed_t = 0;
    //float timestep = 0.001f;
    float timestep = get_timestep(simulation_width, dim);
    bool step = false;
    bool running = false;
    bool pause = false;
    float pause_time = 100;
    bool render = true;
    int render_skipping = 4;
    bool render2 = false;
    bool debug_render = false;
    bool lock_camera_to_slider = false;
    bool progress_camera_time = false;

    vec3f camera_pos = {0, 0, -25};;
    quat camera_quat;
    steady_timer frame_time;

    float cam_time = 0;
    uint32_t render_frame_idx = 0;

    while(!win.should_close())
    {
        win.poll();

        float ftime_s = frame_time.restart();

        if(progress_camera_time)
            cam_time += ftime_s;

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

        ImGui::Checkbox("Override Camera Time", &lock_camera_to_slider);
        ImGui::Checkbox("Advance Override Camera Time", &progress_camera_time);

        ///lock to camera, progress camera time
        ImGui::DragFloat("Override Time", &cam_time, 1.f, 0.f, 400.f);
        ImGui::Checkbox("Capture Render Slices", &rt_bssn.capture_4slices);
        ImGui::SliderInt("Render Skipping", &render_skipping, 1, 32);

        step = step || running;

        ImGui::Text("Elapsed %f", elapsed_t);

        #if 0
        std::vector<float> lines;

        for(auto& i : m.hamiltonian_error)
            lines.push_back(i);

        ImGui::PlotLines("H", lines.data(), lines.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(400, 100));

        std::vector<float> Mis;

        for(auto& i : m.Mi_error)
            Mis.push_back(i);

        ImGui::PlotLines("Mi", Mis.data(), Mis.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(400, 100));

        std::vector<float> cgs;

        for(auto& i : m.cG_error)
            cgs.push_back(i);

        ImGui::PlotLines("cG", cgs.data(), cgs.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(400, 100));
        #endif

        ImGui::End();

        steady_timer t;

        screen_tex.acquire(cqueue);

        if(pause && elapsed_t > 100)
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
                rt_bssn.render3(cqueue, camera4, camera_quat, background, screen_tex, simulation_width, m, lock_camera_to_slider, progress_camera_time);

            if(render2)
                rt_bssn.render4(cqueue, camera4, camera_quat, background, screen_tex, simulation_width, m, lock_camera_to_slider, progress_camera_time);
        }

        if(!render && !render2 && debug_render)
        {
            float scale = get_scale(simulation_width, dim);

            cl::args args;
            m.buffers[0].append_to(args);
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

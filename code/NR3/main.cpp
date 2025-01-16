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

struct raytrace_bssn
{
    ///SO
    ///i think that perhaps we should take multi kernel approach
    ///where we raytrace between two slices, step a few times, then raytrace into the next slice
    ///we can still use a dynamic timestep, we just iterate until you've exceeded a certain threshold, and then
    ///recollect
    cl::buffer position;
    cl::buffer velocity;
    cl::buffer results;
    int last_size = 0;
    bool is_snapshotting = false;
    float elapsed_dt = 0;

    std::vector<bssn_buffer_pack> snapshots;

    raytrace_bssn(cl::context& ctx) : position(ctx), velocity(ctx), results(ctx)
    {
        auto get_metric = [](v4f position, bssn_args_mem<buffer<valuef>> in, literal<v3i> dim, literal<valuef> scale)
        {
            v3f grid = world_to_grid(position.yzw(), dim.get(), scale.get());

            grid = clamp(grid, (v3f){2,2,2}, (v3f)dim.get() - (v3f){3,3,3});

            adm_variables adm = admf_at(grid, dim.get(), in);

            return calculate_real_metric(adm.Yij, adm.gA, adm.gB);
        };

        std::string str = value_impl::make_function(build_initial_tetrads<get_metric, bssn_args_mem<buffer<valuef>>, literal<v3i>, literal<valuef>>, "init_tetrads");

        std::cout << "Got 1 " << str << std::endl;

        cl::program p1 = cl::build_program_with_cache(ctx, {str}, false);

        ctx.register_program(p1);

        std::string str2 = value_impl::make_function(trace3, "trace3");

        std::cout << "Got 2 " << str2 << std::endl;

        cl::program p2 = cl::build_program_with_cache(ctx, {str2}, false);

        ctx.register_program(p2);

        std::string str3 = value_impl::make_function(init_rays3, "init_rays3");

        cl::program p3 = cl::build_program_with_cache(ctx, {str3}, false);

        ctx.register_program(p3);

        std::string str4 = value_impl::make_function(trace4, "trace4");

        cl::program p4 = cl::build_program_with_cache(ctx, {str4}, false);

        ctx.register_program(p4);
    }

    void capture_snapshots(cl::context ctx, cl::command_queue cqueue, float dt, mesh& m)
    {
        float time_between_snapshots = 4;

        elapsed_dt += dt;

        if(elapsed_dt < time_between_snapshots && snapshots.size() > 0)
            return;

        elapsed_dt -= time_between_snapshots;
        elapsed_dt = std::max(elapsed_dt, 0.f);

        bssn_buffer_pack next(ctx);
        next.allocate(m.dim);

        for(auto& i : next.cG)
            i = cl::buffer(ctx);

        printf("Captured\n");
        snapshots.push_back(next);
    }

    void poll_render_resolution(int width, int height)
    {
        if(last_size == width * height)
            return;

        last_size = width * height;
        position.alloc(width * height * sizeof(cl_float4));
        velocity.alloc(width * height * sizeof(cl_float4));
        results.alloc(width * height * sizeof(cl_int));
    }
};

cl::image load_background(cl::context ctx, cl::command_queue cqueue, const std::string& name)
{
    sf::Image background;
    background.loadFromFile(name);

    int background_width = background.getSize().x;
    int background_height = background.getSize().y;

    std::vector<float> background_floats;

    for(int y=0; y < background.getSize().y; y++)
    {
        for(int x=0; x < background.getSize().x; x++)
        {
            sf::Color col = background.getPixel(x, y);

            tensor<float, 3> tcol = {col.r, col.g, col.b};

            tcol = tcol / 255.f;
            tcol = tcol.for_each([](auto&& in){return srgb_to_lin(in);});

            background_floats.push_back(tcol.x());
            background_floats.push_back(tcol.y());
            background_floats.push_back(tcol.z());
            background_floats.push_back(1.f);
        }
    }

    cl_image_format fmt;
    fmt.image_channel_order = CL_RGBA;
    fmt.image_channel_data_type = CL_FLOAT;

    int w = background.getSize().x;
    int h = background.getSize().y;

    cl::image img(ctx);
    img.alloc({w, h}, fmt);
    img.write<2>(cqueue, (const char*)background_floats.data(), {0,0}, {w, h});

    return img;
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

    std::jthread build_thread([&]()
    {
        steady_timer btime;

        std::vector<std::jthread> threads;

        auto make_and_register = [&](const std::string& str)
        {
            cl::program p1 = cl::build_program_with_cache(ctx, {str}, false);

            ctx.register_program(p1);
        };

        #define THREAD(x) threads.emplace_back([&](){make_and_register(x); printf("Buildy " #x "\n");});

        THREAD(make_derivatives());
        THREAD(make_bssn(dim));
        THREAD(make_initial_conditions());
        THREAD(init_christoffel());
        THREAD(init_debugging());
        THREAD(make_momentum_constraint());
        THREAD(make_kreiss_oliger());
        THREAD(make_hamiltonian_error());
        THREAD(make_global_sum());
        THREAD(make_momentum_error(0));
        THREAD(make_momentum_error(1));
        THREAD(make_momentum_error(2));
        THREAD(make_cG_error(0));
        THREAD(make_cG_error(1));
        THREAD(make_cG_error(2));
        THREAD(enforce_algebraic_constraints());
        THREAD(make_sommerfeld());

        for(auto& i : threads)
            i.join();

        std::cout << "Btime " << btime.get_elapsed_time_s() << std::endl;

        printf("Built kernels\n");
    });

    build_thread.join();

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

    //m.load_from(cqueue);

    texture_settings tsett;
    tsett.width = dim.x();
    tsett.height = dim.y();
    tsett.is_srgb = false;
    tsett.generate_mipmaps = false;

    texture tex;
    tex.load_from_memory(tsett, nullptr);

    cl::gl_rendertexture rtex{ctx};
    rtex.create_from_texture(tex.handle);

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

    raytrace_bssn rt_bssn(ctx);

    cl::image background = load_background(ctx, cqueue, "../common/nasa.png");

    std::array<cl::buffer, 4> tetrads{ctx, ctx, ctx, ctx};
    cl::buffer gpu_position(ctx);

    for(int i=0; i < 4; i++)
        tetrads[i].alloc(sizeof(cl_float4));

    gpu_position.alloc(sizeof(cl_float4));

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
    bool render2 = false;

    vec3f camera_pos = {0, 0, -25};;
    quat camera_quat;

    while(!win.should_close())
    {
        win.poll();

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

        //rtex.acquire(cqueue);

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

            cl_float4 camera = {elapsed_t,camera_pos.x(),camera_pos.y(),camera_pos.z()};
            quat q = camera_quat;

            cl_float4 cq = {q.q[0], q.q[1], q.q[2], q.q[3]};

            cl_int bwidth = background.size<2>().x();
            cl_int bheight = background.size<2>().y();

            cl_int screen_width = screen_tex.size<2>().x();
            cl_int screen_height = screen_tex.size<2>().y();
            float scale = get_scale(simulation_width, dim);

            int buf = m.valid_derivative_buffer;

            ///so. The derivatigves are last valid for [2], which is super *super* hacky if I'm doing it like this
            {
                cl_float3 vel = {0,0,0};

                cl::args args;
                args.push_back(camera, vel, gpu_position,
                               tetrads[0], tetrads[1], tetrads[2], tetrads[3]);

                m.buffers[buf].append_to(args);
                args.push_back(dim);
                args.push_back(scale);

                cqueue.exec("init_tetrads", args, {screen_width, screen_height}, {8,8});
            }

            {
                cl::args args;
                args.push_back(screen_width, screen_height);
                args.push_back(rt_bssn.position, rt_bssn.velocity);
                args.push_back(tetrads[0], tetrads[1], tetrads[2], tetrads[3]);
                args.push_back(gpu_position, cq);
                args.push_back(dim, scale);

                m.buffers[buf].append_to(args);

                cqueue.exec("init_rays3", args, {screen_width, screen_height}, {8,8});
            }

            #if 1
            if(render)
            {
                cl::args args;
                args.push_back(screen_width, screen_height);
                args.push_back(background);
                args.push_back(screen_tex);
                args.push_back(bwidth, bheight);
                args.push_back(cq);
                args.push_back(rt_bssn.position, rt_bssn.velocity);
                args.push_back(dim);
                args.push_back(scale);

                m.buffers[buf].append_to(args);

                for(auto& i : m.derivatives)
                    args.push_back(i);

                cqueue.exec("trace3", args, {screen_width, screen_height}, {8,8});
            }
            #endif // 0

            if(render2)
            {
                if(rt_bssn.snapshots.size() >= 2)
                {
                    std::vector<cl::buffer> derivs_next;
                    std::vector<cl::buffer> derivs_cur;

                    for(int i=0; i < 11 * 3; i++)
                    {
                        derivs_next.emplace_back(ctx);
                        derivs_next.back().alloc(sizeof(derivative_t::interior_type) * int64_t{m.dim.x()} * m.dim.y() * m.dim.z());

                        derivs_cur.emplace_back(ctx);
                        derivs_cur.back().alloc(sizeof(derivative_t::interior_type) * int64_t{m.dim.x()} * m.dim.y() * m.dim.z());
                    }

                    rt_bssn.results.set_to_zero(cqueue);

                    m.calculate_derivatives_for(cqueue, rt_bssn.snapshots.back(), derivs_next);

                    float time_start = elapsed_t;

                    for(int i=(int)rt_bssn.snapshots.size() - 1; i >= 1; i--)
                    {
                        auto& next = rt_bssn.snapshots[i];
                        auto& current = rt_bssn.snapshots[i - 1];

                        float next_time = time_start;
                        float current_time = time_start - timestep;
                        time_start -= timestep;

                        m.calculate_derivatives_for(cqueue, current, derivs_next);

                        cl::args args;
                        args.push_back(screen_width, screen_height);
                        args.push_back(background, screen_tex);
                        args.push_back(bwidth, bheight, cq, rt_bssn.position, rt_bssn.velocity, dim, scale);
                        args.push_back(next_time, current_time);

                        current.append_to(args);
                        next.append_to(args);

                        for(auto& i : derivs_cur)
                            args.push_back(i);

                        for(auto& i : derivs_next)
                            args.push_back(i);

                        args.push_back(rt_bssn.results);

                        cqueue.exec("trace4", args, {screen_width, screen_height}, {8, 8});

                        std::swap(derivs_next, derivs_cur);
                    }
                }
            }
        }

        /*{
            float scale = get_scale(simulation_width, dim);

            cl::args args;
            m.buffers[0].append_to(args);
            args.push_back(dim);
            args.push_back(scale);
            args.push_back(rtex);

            cqueue.exec("debug", args, {dim.x() * dim.y() * dim.z()}, {128});
        }*/

        //rtex.unacquire(cqueue);
        screen_tex.unacquire(cqueue);

        ImGui::GetBackgroundDrawList()->AddImage((void*)tex2.handle, ImVec2(0,0), ImVec2(tex2.get_size().x(), tex2.get_size().y()));
        //ImGui::GetBackgroundDrawList()->AddImage((void*)tex.handle, ImVec2(0,0), ImVec2(dim.x() * 3, dim.y() * 3));

        win.display();

        std::cout << "T " << t.get_elapsed_time_s() * 1000. << std::endl;

        if(step)
            elapsed_t += timestep;
    }
}

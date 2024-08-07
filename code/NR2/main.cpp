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
    std::vector<cl::buffer> momentum_constraint;
    cl::buffer temporary_buffer;
    cl::buffer temporary_single;
    std::vector<double> hamiltonian_error;
    std::vector<double> Mi_error;

    mesh(cl::context& ctx, t3i _dim) : buffers{ctx, ctx, ctx}, temporary_buffer(ctx), temporary_single(ctx)
    {
        dim = _dim;
    }

    void allocate(cl::context& ctx)
    {
        for(int i=0; i < 3; i++)
        {
            buffers[i].allocate(dim);
        }

        for(int i=0; i < 3; i++)
        {
            cl::buffer buf(ctx);
            buf.alloc(sizeof(cl_float) * int64_t{dim.x()} * dim.y() * dim.z());

            momentum_constraint.push_back(buf);
        }

        for(int i=0; i < 11 * 3; i++)
        {
            cl::buffer buf(ctx);
            buf.alloc(sizeof(derivative_t::interior_type) * int64_t{dim.x()} * dim.y() * dim.z());

            derivatives.push_back(buf);
        }

        temporary_buffer.alloc(sizeof(cl_float) * uint64_t{dim.x()} * dim.y() * dim.z());
        temporary_single.alloc(sizeof(cl_ulong));
    }

    void init(float simulation_width, cl::context& ctx, cl::command_queue& cqueue)
    {
        cl_int4 cldim = {dim.x(), dim.y(), dim.z(), 0};
        float scale = get_scale(simulation_width, dim);

        {
            black_hole_params p1;
            p1.bare_mass = 0.5f;
            p1.position = {4, 0, 0};
            p1.linear_momentum = {0, 0.133, 0};

            black_hole_data d1 = init_black_hole(ctx, cqueue, p1, dim, scale);

            black_hole_params p2;
            p2.bare_mass = 0.5f;
            p2.position = {-4, 0, 0};
            p2.linear_momentum = {0, -0.133, 0};

            black_hole_data d2 = init_black_hole(ctx, cqueue, p2, dim, scale);

            initial_conditions init(ctx, cqueue, dim);

            init.add(cqueue, d1);
            init.add(cqueue, d2);

            init.build(ctx, cqueue, scale, buffers[0]);
        }

        /*{
            cl::args args;
            buffers[0].append_to(args);
            args.push_back(cldim);
            args.push_back(scale);

            //cqueue.exec("init", args, {dim.x() * dim.y() * dim.z()}, {128});
            cqueue.exec("init_christoffel", args, {dim.x() * dim.y() * dim.z()}, {128});
        }*/

        temporary_buffer.set_to_zero(cqueue);
        temporary_single.set_to_zero(cqueue);
    }

    void step(cl::context& ctx, cl::command_queue& cqueue, float timestep, float simulation_width)
    {
        cl_int4 cldim = {dim.x(), dim.y(), dim.z(), 0};
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
                float eps = 0.25f;

                cl::args args;
                args.push_back(linear_base.at(i));
                args.push_back(linear_inout.at(i));
                args.push_back(timestep);
                args.push_back(cldim);
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

            args.push_back(cldim);

            cqueue.exec("enforce_algebraic_constraints", args, {dim.x() * dim.y() * dim.z()}, {128});
        };

        auto diff = [&](cl::buffer buf, int buffer_idx)
        {
            cl::args args;
            args.push_back(buf);
            args.push_back(derivatives.at(buffer_idx * 3 + 0));
            args.push_back(derivatives.at(buffer_idx * 3 + 1));
            args.push_back(derivatives.at(buffer_idx * 3 + 2));
            args.push_back(cldim);
            args.push_back(scale);

            cqueue.exec("differentiate", args, {dim.x()*dim.y()*dim.z()}, {128});
        };

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
                args.push_back(cldim);
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
            }
        };

        auto calculate_momentum_constraint_for = [&](int pack_idx)
        {
            cl::args args;
            buffers[pack_idx].append_to(args);

            for(auto& i : momentum_constraint)
                args.push_back(i);

            args.push_back(cldim);
            args.push_back(scale);

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

            for(auto& i : momentum_constraint)
                args.push_back(i);

            args.push_back(timestep);
            args.push_back(cldim);
            args.push_back(scale);

            cqueue.exec("evolve", args, {dim.x()*dim.y()*dim.z()}, {128});
        };

        auto substep = [&](int iteration, int base_idx, int in_idx, int out_idx)
        {
            ///this assumes that in_idx == base_idx for iteration 0, so that they are both constraint enforced
            enforce_constraints(in_idx);

            {
                std::vector<cl::buffer> to_diff {
                    buffers[in_idx].cY[0],
                    buffers[in_idx].cY[1],
                    buffers[in_idx].cY[2],
                    buffers[in_idx].cY[3],
                    buffers[in_idx].cY[4],
                    buffers[in_idx].cY[5],
                    buffers[in_idx].gA,
                    buffers[in_idx].gB[0],
                    buffers[in_idx].gB[1],
                    buffers[in_idx].gB[2],
                    buffers[in_idx].W,
                };

                for(int i=0; i < (int)to_diff.size(); i++)
                {
                    diff(to_diff[i], i);
                }
            }

            //#define CALCULATE_CONSTRAINT_ERRORS
            #ifdef CALCULATE_CONSTRAINT_ERRORS
            if(iteration == 0)
                calculate_constraint_errors(in_idx);
            #endif

            //#define CALCULATE_MOMENTUM_CONSTRAINT
            #ifdef CALCULATE_MOMENTUM_CONSTRAINT
            calculate_momentum_constraint_for(in_idx);
            #endif

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
    }
};

int main()
{
    render_settings sett;
    sett.width = 1280;
    sett.height = 720;
    sett.opencl = true;
    sett.no_double_buffer = false;
    sett.is_srgb = true;
    sett.no_decoration = false;
    sett.viewports = false;

    render_window win(sett, "Geodesics");

    assert(win.clctx);

    cl::context& ctx = win.clctx->ctx;
    std::cout << cl::get_extensions(ctx) << std::endl;

    t3i dim = {213, 213, 213};

    {
        auto make_and_register = [&](const std::string& str)
        {
            cl::program p1(ctx, str, false);
            p1.build(ctx, "");

            ctx.register_program(p1);
        };

        make_and_register(make_derivatives());
        make_and_register(make_bssn(dim));
        make_and_register(make_initial_conditions());
        make_and_register(init_christoffel());
        make_and_register(init_debugging());
        make_and_register(make_momentum_constraint());
        make_and_register(make_kreiss_oliger());
        make_and_register(make_hamiltonian_error());
        make_and_register(make_global_sum());
        make_and_register(make_momentum_error(0));
        make_and_register(make_momentum_error(1));
        make_and_register(make_momentum_error(2));
        make_and_register(enforce_algebraic_constraints());
    }

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

    mesh m(ctx, dim);
    m.allocate(ctx);
    m.init(simulation_width, ctx, cqueue);
    //m.load_from(cqueue);

    texture_settings tsett;
    tsett.width = 600;
    tsett.height = 300;
    tsett.is_srgb = false;
    tsett.generate_mipmaps = false;

    texture tex;
    tex.load_from_memory(tsett, nullptr);

    cl::gl_rendertexture rtex{ctx};
    rtex.create_from_texture(tex.handle);

    cqueue.block();

    printf("Start\n");

    float elapsed_t = 0;
    //float timestep = 0.001f;
    float timestep = 0.03;

    while(!win.should_close())
    {
        win.poll();

        ImGui::Begin("Hi");

        ImGui::Text("Elapsed %f", elapsed_t);

        std::vector<float> lines;

        for(auto& i : m.hamiltonian_error)
            lines.push_back(i);

        ImGui::PlotLines("H", lines.data(), lines.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(400, 100));

        std::vector<float> Mis;

        for(auto& i : m.Mi_error)
            Mis.push_back(i);

        ImGui::PlotLines("Mi", Mis.data(), Mis.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(400, 100));

        ImGui::End();

        steady_timer t;

        rtex.acquire(cqueue);
        m.step(ctx, cqueue, timestep, simulation_width);

        {
            cl_int4 cldim = {dim.x(), dim.y(), dim.z(), 0};
            float scale = get_scale(simulation_width, dim);

            cl::args args;
            m.buffers[0].append_to(args);
            args.push_back(cldim);
            args.push_back(scale);
            args.push_back(rtex);

            cqueue.exec("debug", args, {dim.x() * dim.y() * dim.z()}, {128});
        }

        rtex.unacquire(cqueue);

        ImGui::GetBackgroundDrawList()->AddImage((void*)tex.handle, ImVec2(0,0), ImVec2(1280,720));

        win.display();

        std::cout << "T " << t.get_elapsed_time_s() * 1000. << std::endl;

        elapsed_t += timestep;
    }
}

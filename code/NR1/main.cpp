#include <iostream>
#include <toolkit/render_window.hpp>
#include <imgui/misc/freetype/imgui_freetype.h>
#include "bssn.hpp"
#include <vec/tensor.hpp>
#include <toolkit/texture.hpp>
#include <toolkit/fs_helpers.hpp>
#include "errors.hpp"
#include "init.hpp"

using t3i = tensor<int, 3>;

struct bssn_buffer_pack
{
    std::array<cl::buffer, 6> cY;
    std::array<cl::buffer, 6> cA;
    cl::buffer K;
    cl::buffer W;
    std::array<cl::buffer, 3> cG;

    cl::buffer gA;
    std::array<cl::buffer, 3> gB;

    //lovely
    bssn_buffer_pack(cl::context& ctx) :
        cY{ctx, ctx, ctx, ctx, ctx, ctx},
        cA{ctx, ctx, ctx, ctx, ctx, ctx},
        K{ctx},
        W{ctx},
        cG{ctx, ctx, ctx},
        gA{ctx},
        gB{ctx, ctx, ctx}
    {

    }

    void allocate(t3i size)
    {
        int64_t linear_size = int64_t{size.x()} * size.y() * size.z();

        for(auto& i : cY)
            i.alloc(sizeof(cl_float) * linear_size);
        for(auto& i : cA)
            i.alloc(sizeof(cl_float) * linear_size);
        for(auto& i : cG)
            i.alloc(sizeof(cl_float) * linear_size);
        for(auto& i : gB)
            i.alloc(sizeof(cl_float) * linear_size);

        K.alloc(sizeof(cl_float) * linear_size);
        W.alloc(sizeof(cl_float) * linear_size);
        gA.alloc(sizeof(cl_float) * linear_size);
    }

    template<typename T>
    void for_each(T&& func)
    {
        for(auto& i : cY)
            func(i);

        for(auto& i : cA)
            func(i);

        func(K);
        func(W);

        for(auto& i : cG)
            func(i);

        func(gA);

        for(auto& i : gB)
            func(i);
    }

    void append_to(cl::args& args)
    {
        for(auto& i : cY)
            args.push_back(i);

        for(auto& i : cA)
            args.push_back(i);

        args.push_back(K);
        args.push_back(W);

        for(auto& i : cG)
            args.push_back(i);

        args.push_back(gA);

        for(auto& i : gB)
            args.push_back(i);
    }
};

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

    void init(cl::command_queue& cqueue)
    {
        #define MINK
        #ifdef MINK
        cl_float zero = 0;
        cl_float one = 1;

        bssn_buffer_pack& pck = buffers[0];

        pck.cY[0].fill(cqueue, one);
        pck.cY[1].fill(cqueue, zero);
        pck.cY[2].fill(cqueue, zero);
        pck.cY[3].fill(cqueue, one);
        pck.cY[4].fill(cqueue, zero);
        pck.cY[5].fill(cqueue, one);

        for(auto& i : pck.cA)
            i.set_to_zero(cqueue);

        for(auto& i : pck.cG)
            i.set_to_zero(cqueue);

        for(auto& i : pck.gB)
            i.set_to_zero(cqueue);

        pck.K.set_to_zero(cqueue);
        pck.W.fill(cqueue, one);
        pck.gA.fill(cqueue, one);
        #endif

        cl_int4 cldim = {dim.x(), dim.y(), dim.z(), 0};
        float c_at_max = 1;
        float scale = c_at_max / dim.x();

        {
            cl::args args;
            buffers[0].append_to(args);
            args.push_back(cldim);
            args.push_back(scale);

            cqueue.exec("init", args, {dim.x() * dim.y() * dim.z()}, {128});
            cqueue.exec("init_christoffel", args, {dim.x() * dim.y() * dim.z()}, {128});
        }

        temporary_buffer.set_to_zero(cqueue);
        temporary_single.set_to_zero(cqueue);
    }

    void load_from(cl::command_queue cqueue)
    {
        auto load = [&](cl::buffer& buf, const std::string& name)
        {
            std::string data = file::read(name, file::mode::BINARY);

            buf.write(cqueue, data.data(), data.size());
        };

        bssn_buffer_pack& pck = buffers[0];

        load(pck.cY[0], "./init/buf_cY0.bin");
        load(pck.cY[1], "./init/buf_cY1.bin");
        load(pck.cY[2], "./init/buf_cY2.bin");
        load(pck.cY[3], "./init/buf_cY3.bin");
        load(pck.cY[4], "./init/buf_cY4.bin");
        load(pck.cY[5], "./init/buf_cY5.bin");

        load(pck.cA[0], "./init/buf_cA0.bin");
        load(pck.cA[1], "./init/buf_cA1.bin");
        load(pck.cA[2], "./init/buf_cA2.bin");
        load(pck.cA[3], "./init/buf_cA3.bin");
        load(pck.cA[4], "./init/buf_cA4.bin");
        load(pck.cA[5], "./init/buf_cA5.bin");

        load(pck.gA, "./init/buf_gA.bin");
        load(pck.gB[0], "./init/buf_gB0.bin");
        load(pck.gB[1], "./init/buf_gB1.bin");
        load(pck.gB[2], "./init/buf_gB2.bin");
        load(pck.K, "./init/buf_K.bin");
        load(pck.W, "./init/buf_X.bin");

        load(pck.cG[0], "./init/buf_cGi0.bin");
        load(pck.cG[1], "./init/buf_cGi1.bin");
        load(pck.cG[2], "./init/buf_cGi2.bin");
    }

    void step(cl::context& ctx, cl::command_queue& cqueue, float timestep, float c_at_max)
    {
        cl_int4 cldim = {dim.x(), dim.y(), dim.z(), 0};
        float scale = c_at_max / dim.x();

        auto kreiss = [&](int in, int inout)
        {
            //std::swap(buffers[in], buffers[inout]);
            //return;

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

        auto substep = [&](int iteration, int base_idx, int in_idx, int out_idx)
        {
            {
                std::vector<cl::buffer> d_in {
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

                int which_deriv = 0;

                for(cl::buffer& to_diff : d_in)
                {
                    cl::args args;
                    args.push_back(to_diff);
                    args.push_back(derivatives.at(which_deriv * 3 + 0));
                    args.push_back(derivatives.at(which_deriv * 3 + 1));
                    args.push_back(derivatives.at(which_deriv * 3 + 2));
                    args.push_back(cldim);
                    args.push_back(scale);

                    cqueue.exec("differentiate", args, {dim.x()*dim.y()*dim.z()}, {128});

                    which_deriv++;
                }
            }

            #define CALCULATE_CONSTRAINT_ERRORS
            #ifdef CALCULATE_CONSTRAINT_ERRORS
            if(iteration == 0)
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
                    buffers[in_idx].append_to(args);

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
            }
            #endif

            //#define CALCULATE_MOMENTUM_CONSTRAINT
            #ifdef CALCULATE_MOMENTUM_CONSTRAINT
            {
                cl::args args;
                buffers[in_idx].append_to(args);

                for(auto& i : momentum_constraint)
                    args.push_back(i);

                args.push_back(cldim);
                args.push_back(scale);

                cqueue.exec("momentum_constraint", args, {dim.x() * dim.y() * dim.z()}, {128});
            }
            #endif

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

        int iterations = 2;

        for(int i=0; i < iterations; i++)
        {
            if(i == 0)
                substep(i, 0, 0, 1);
            else
                substep(i, 0, 2, 1);

            ///we always output into buffer 1, which means that buffer 2 becomes our next input
            if(i != iterations - 1)
                std::swap(buffers[1], buffers[2]);
        }

        ///now that we've finished, our result is in buffer[1]
        std::swap(buffers[1], buffers[0]);

        kreiss(0, 1);
        std::swap(buffers[0], buffers[1]);

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

    t3i dim = {300, 300, 300};

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
    }

    cl::command_queue cqueue(ctx, (1<<9));

    ImFontAtlas* atlas = ImGui::GetIO().Fonts;
    atlas->FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImFontConfig font_cfg;
    font_cfg.GlyphExtraSpacing = ImVec2(0, 0);
    font_cfg.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImGuiIO& io = ImGui::GetIO();

    io.Fonts->Clear();
    io.Fonts->AddFontFromFileTTF("VeraMono.ttf", 14, &font_cfg);

    mesh m(ctx, dim);
    m.allocate(ctx);
    m.init(cqueue);
    //m.load_from(cqueue);

    texture_settings tsett;
    tsett.width = 1280;
    tsett.height = 720;
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
    float timestep = 0.001;

    float c_at_max = 1;

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
        m.step(ctx, cqueue, timestep, c_at_max);

        {
            cl_int4 cldim = {dim.x(), dim.y(), dim.z(), 0};
            float scale = c_at_max / dim.x();

            cl::args args;
            m.buffers[0].append_to(args);
            args.push_back(cldim);
            args.push_back(scale);
            args.push_back(rtex);

            cqueue.exec("debug", args, {dim.x() * dim.y() * dim.z()}, {128});
        }

        rtex.unacquire(cqueue);

        cqueue.block();

        ImGui::GetBackgroundDrawList()->AddImage((void*)tex.handle, ImVec2(0,0), ImVec2(1280,720));

        win.display();

        std::cout << "T " << t.get_elapsed_time_s() * 1000. << std::endl;

        elapsed_t += timestep;
    }
}

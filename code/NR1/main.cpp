#include <iostream>
#include <toolkit/render_window.hpp>
#include <imgui/misc/freetype/imgui_freetype.h>
#include "bssn.hpp"
#include <vec/tensor.hpp>
#include <toolkit/texture.hpp>

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

    mesh(cl::context& ctx, t3i _dim) : buffers{ctx, ctx, ctx}
    {
        dim = _dim;
    }

    void allocate(cl::context& ctx)
    {
        for(int i=0; i < 3; i++)
        {
            buffers[i].allocate(dim);
        }

        for(int i=0; i < 11 * 3; i++)
        {
            cl::buffer buf(ctx);
            buf.alloc(sizeof(cl_float) * int64_t{dim.x()} * dim.y() * dim.z());

            derivatives.push_back(buf);
        }
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
    }

    void step(cl::context& ctx, cl::command_queue& cqueue, float timestep)
    {
        cl_int4 cldim = {dim.x(), dim.y(), dim.z(), 0};
        float c_at_max = 1;
        float scale = c_at_max / dim.x();

        auto substep = [&](int base_idx, int in_idx, int out_idx)
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

            cl::args args;
            buffers[base_idx].append_to(args);
            buffers[in_idx].append_to(args);
            buffers[out_idx].append_to(args);

            for(auto& i : derivatives)
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
                substep(0, 0, 1);
            else
                substep(0, 2, 1);

            ///we always output into buffer 1, which means that buffer 2 becomes our next input
            if(i != iterations - 1)
                std::swap(buffers[1], buffers[2]);
        }

        ///now that we've finished, our result is in buffer[2]

        std::swap(buffers[1], buffers[0]);
    }
};

int main()
{
    //std::cout << make_derivatives() << std::endl;
    //std::cout << make_bssn() << std::endl;

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

    {
        auto make_and_register = [&](const std::string& str)
        {
            cl::program p1(ctx, str, false);
            p1.build(ctx, "");

            ctx.register_program(p1);
        };

        make_and_register(make_derivatives());
        make_and_register(make_bssn());
        make_and_register(make_initial_conditions());
        make_and_register(init_christoffel());
        make_and_register(init_debugging());
    }

    cl::command_queue& cqueue = win.clctx->cqueue;

    ImFontAtlas* atlas = ImGui::GetIO().Fonts;
    atlas->FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImFontConfig font_cfg;
    font_cfg.GlyphExtraSpacing = ImVec2(0, 0);
    font_cfg.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImGuiIO& io = ImGui::GetIO();

    io.Fonts->Clear();
    io.Fonts->AddFontFromFileTTF("VeraMono.ttf", 14, &font_cfg);

    t3i dim = {256, 256, 256};

    mesh m(ctx, dim);
    m.allocate(ctx);
    m.init(cqueue);

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

    while(!win.should_close())
    //for(int i=0; i < 10; i++)
    {
        win.poll();

        ImGui::Begin("Hi");

        ImGui::End();

        rtex.acquire(cqueue);
        m.step(ctx, cqueue, 0.001f);

        {
            cl_int4 cldim = {dim.x(), dim.y(), dim.z(), 0};
            float c_at_max = 1;
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

        steady_timer t;


        std::cout << "T " << t.get_elapsed_time_s() * 1000. << std::endl;
    }
}

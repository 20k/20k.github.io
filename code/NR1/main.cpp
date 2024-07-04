#include <iostream>
#include <toolkit/render_window.hpp>
#include <imgui/misc/freetype/imgui_freetype.h>
#include "bssn.hpp"
#include <vec/tensor.hpp>

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
};

struct mesh
{
    std::array<bssn_buffer_pack, 3> buffers;
    t3i dim;

    mesh(cl::context& ctx, t3i _dim) : buffers{ctx, ctx, ctx}
    {
        dim = _dim;
    }

    void allocate()
    {
        for(int i=0; i < 3; i++)
        {
            buffers[i].allocate(dim);
        }
    }

    void step(float timestep)
    {

    }
};

int main()
{
    std::cout << make_derivatives() << std::endl;
    std::cout << make_bssn() << std::endl;

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

    ImFontAtlas* atlas = ImGui::GetIO().Fonts;
    atlas->FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImFontConfig font_cfg;
    font_cfg.GlyphExtraSpacing = ImVec2(0, 0);
    font_cfg.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImGuiIO& io = ImGui::GetIO();

    io.Fonts->Clear();
    io.Fonts->AddFontFromFileTTF("VeraMono.ttf", 14, &font_cfg);

    while(!win.should_close())
    {
        win.poll();

        ImGui::Begin("Hi");

        ImGui::End();

        win.display();
    }
}

add_rules("mode.debug", "mode.release")
add_requires("opengl", "glfw", "glew", "sfml", "opencl")

add_files("../common/imgui/backends/imgui_impl_glfw.cpp")
add_files("../common/imgui/backends/imgui_impl_opengl3.cpp")
add_files("../common/imgui/misc/freetype/imgui_freetype.cpp")
add_files("../common/imgui/misc/cpp/imgui_stdlib.cpp")
add_files("../common/imgui/imgui.cpp")
add_files("../common/imgui/imgui_draw.cpp")
add_files("../common/imgui/imgui_tables.cpp")
add_files("../common/imgui/imgui_widgets.cpp")

add_files("../common/toolkit/clock.cpp")
add_files("../common/toolkit/fs_helpers.cpp")
add_files("../common/toolkit/opencl.cpp")
add_files("../common/toolkit/render_window.cpp")
add_files("../common/toolkit/render_window_glfw.cpp")
add_files("../common/toolkit/texture.cpp")

add_files("../common/libtov/tov.cpp")

add_includedirs("../common")
add_includedirs("../common/imgui")

set_languages("c99", "cxx23")

add_defines("IMGUI_IMPL_OPENGL_LOADER_GLEW",
"SUBPIXEL_FONT_RENDERING",
"CL_TARGET_OPENCL_VERSION=220",
"IMGUI_ENABLE_FREETYPE",
"NO_SERIALISE_RATELIMIT",
"NO_OPENCL_SCREEN")

add_packages("opengl", "glfw", "glew", "freetype", "sfml", "opencl")

set_optimize("fastest")

add_links("imm32")

if is_plat("mingw") then
    add_cxflags("-mwindows")
    add_ldflags("-mwindows")
end

target("nr4")
    set_kind("binary")
    add_files("main.cpp")
    add_files("*.cpp")
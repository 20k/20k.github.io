To build, add ../common to the include search paths. You'll also need to link against:

../common/
    clock.cpp
    fs_helpers.cpp
    opencl.cpp
    render_window.cpp
    render_window_glfw.cpp
    texture.cpp
    
../common/imgui
    imgui.cpp
    imgui_draw.cpp
    imgui_tables.cpp
    imgui_widgets.cpp
    
../common/imgui/backends
    imgui_impl_glfw.cpp
    imgui_impl_opengl3.cpp

As well as the main files for this project:
    bssn.cpp
    derivatives.cpp
    errors.cpp
    init.cpp
    kreiss_oliger.cpp
    main.cpp

Link libraries are:

-lsfml-window
-lsfml-system
-lsfml-graphics
-lglew32
-lopengl32
-lopencl


The following defines are used:

    IMGUI_IMPL_OPENGL_LOADER_GLEW
    SUBPIXEL_FONT_RENDERING
    CL_TARGET_OPENCL_VERSION=220
    IMGUI_ENABLE_FREETYPE
    NO_SERIALISE_RATELIMIT
    NO_OPENCL_SCREEN
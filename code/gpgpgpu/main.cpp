#include "value.hpp"
#include "../common/vec/tensor.hpp"
#include "../common/vec/dual.hpp"
#include "single_source.hpp"
#include "opencl.hpp"
#include "schwarzschild_single_source.hpp"
#include <SFML/Graphics.hpp>
#include <fstream>

template<typename T>
using dual = dual_types::dual_v<T>;

template<typename T>
T my_func(const T& x)
{
    T v1 = x;
    T v2 = 2;

    return v1 * v1 * v2 + v1;
}

void my_function(execution_context& ectx, buffer_mut<float> b1, buffer<float> b2, buffer<float> b3, literal<int> size) {
    value<int> idx = value_impl::get_global_id(0);

    ectx.pin(idx);

    if_e(ectx, idx >= size.get(), [&]()
    {
        return_e(ectx);
    });

    assign_e(ectx, b1[idx], b2[idx] + b3[idx]);
}

void image_test(execution_context& ectx, write_only_image<2> out)
{
    tensor<value<float>, 4> data = {1,2,3,4};
    tensor<value<int>, 2> pos = {1, 2};

    out.write(ectx, pos, data);
}

void debug_build_status(cl_program prog, cl_device_id selected_device)
{
    cl_build_status bstatus = CL_BUILD_ERROR;
    cl_int build_info_result = clGetProgramBuildInfo(prog, selected_device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &bstatus, nullptr);

    if(build_info_result != CL_SUCCESS)
    {
        std::cout << "Error in clGetProgramBuildInfo " << build_info_result << std::endl;
        return;
    }

    if(bstatus == CL_SUCCESS)
        return;

    std::cout << "Build Status: " << bstatus << std::endl;

    assert(bstatus == CL_BUILD_ERROR);

    std::string log;
    size_t log_size = 0;

    clGetProgramBuildInfo(prog, selected_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

    log.resize(log_size + 1);

    clGetProgramBuildInfo(prog, selected_device, CL_PROGRAM_BUILD_LOG, log.size(), &log[0], nullptr);

    std::cout << log << std::endl;

    throw std::runtime_error("Failed to build");
}

cl_kernel build_program_and_get_kernel(opencl& cl, const std::string& src, const std::string& kernel_name, const std::string& options)
{
    const char* ptr = src.data();

    cl_program prog = clCreateProgramWithSource(cl.ctx, 1, &ptr, nullptr, nullptr);
    cl_int build_err = clBuildProgram(prog, 1, &cl.selected_device, options.c_str(), nullptr, nullptr);

    if(build_err != CL_SUCCESS && build_err != CL_BUILD_PROGRAM_FAILURE)
    {
        if(build_err == -66)
            std::cout << "Failed to compile due to build options " << options << std::endl;

        throw std::runtime_error("Build Error " + std::to_string(build_err));
    }

    debug_build_status(prog, cl.selected_device);

    return clCreateKernel(prog, kernel_name.c_str(), nullptr);
}


int main() {
    std::cout << value_impl::make_function(my_function, "my_kernel") << std::endl;
    std::cout << value_impl::make_function(image_test, "image_test") << std::endl;
    std::cout << value_impl::make_function(opencl_raytrace, "raytrace") << std::endl;

    std::string source = value_impl::make_function(opencl_raytrace, "raytrace");
    std::string source2;

    {
        std::ifstream t("./hand_written.cl");
        std::stringstream buffer;
        buffer << t.rdbuf();

        source2 = buffer.str();
    }

    int screen_width = 1000;
    int screen_height = 800;

    sf::VideoMode mode(screen_width, screen_height);
    sf::RenderWindow win(mode, "I am a black hole");

    opencl cl = boot_opencl();

    cl_command_queue cqueue = clCreateCommandQueue(cl.ctx, cl.selected_device, 0, nullptr);

    sf::Image background;
    background.loadFromFile("../common/nasa.png");

    int background_width = background.getSize().x;
    int background_height = background.getSize().y;

    std::vector<float> background_floats;

    for(int y=0; y < background.getSize().y; y++)
    {
        for(int x=0; x < background.getSize().x; x++)
        {
            sf::Color col = background.getPixel(x, y);

            background_floats.push_back(col.r/255.f);
            background_floats.push_back(col.g/255.f);
            background_floats.push_back(col.b/255.f);
            background_floats.push_back(1.f);
        }
    }

    cl_mem background_mem = make_image(cl, background.getSize().x, background.getSize().y);

    {
        size_t origin[3] = {0,0,0};
        size_t region[3] = {background.getSize().x, background.getSize().y, 1};

        clEnqueueWriteImage(cqueue, background_mem, CL_TRUE, origin, region, 0, 0, background_floats.data(), 0, nullptr, nullptr);
    }

    sf::Texture gl_interop;
    gl_interop.create(screen_width, screen_height);

    unsigned int handle = gl_interop.getNativeHandle();

    sf::Texture::bind(&gl_interop);

    cl_mem shared_texture = make_shared_image(cl, handle);

    #define GENERATED
    #ifdef GENERATED
    cl_kernel kern = build_program_and_get_kernel(cl, source, "raytrace", "");
    #endif

    //#define HAND_WRITTEN
    #ifdef HAND_WRITTEN
    cl_kernel kern = build_program_and_get_kernel(cl, source2, "hand_raytracer", "-cl-fast-relaxed-math");
    #endif // HAND_WRITTEN

    while(win.isOpen())
    {
        sf::Event evt;

        while(win.pollEvent(evt))
        {
            if(evt.type == sf::Event::Closed)
                win.close();
        }

        sf::Clock clk;

        clEnqueueAcquireGLObjects(cqueue,1, &shared_texture, 0, nullptr, nullptr);
        clSetKernelArg(kern, 0, sizeof(cl_int), &screen_width);
        clSetKernelArg(kern, 1, sizeof(cl_int), &screen_height);
        clSetKernelArg(kern, 2, sizeof(cl_mem), &background_mem);
        clSetKernelArg(kern, 3, sizeof(cl_mem), &shared_texture);
        clSetKernelArg(kern, 4, sizeof(cl_int), &background_width);
        clSetKernelArg(kern, 5, sizeof(cl_int), &background_height);

        size_t work_offset[3] = {0,0,0};
        size_t g_ws[3] = {screen_width, screen_height, 1};
        size_t l_ws[3] = {8,8,1};

        int dim = 2;

        for(int i=0; i < 2; i++)
        {
            if((g_ws[i] % l_ws[i]) != 0)
            {
                size_t rem = g_ws[i] % l_ws[i];

                g_ws[i] -= rem;
                g_ws[i] += l_ws[i];
            }
        }

        cl_int err = clEnqueueNDRangeKernel(cqueue, kern, 2, work_offset, g_ws, l_ws, 0, nullptr, nullptr);

        if(err != CL_SUCCESS)
        {
            throw std::runtime_error(std::to_string(err));
        }

        clEnqueueReleaseGLObjects(cqueue, 1, &shared_texture, 0, nullptr, nullptr);

        clFinish(cqueue);

        std::cout << "Time " << clk.getElapsedTime().asMicroseconds() / 1000. << std::endl;

        sf::Sprite sprite(gl_interop);
        win.draw(sprite);

        win.display();

        sf::sleep(sf::milliseconds(8));
    }

    return 0;

    #if 0
    value<float> x = 2;
    x.name = "x";

    value<float> result = my_func(x);

    dual<float> as_dual = result.replay([]<typename T>(const T& in)
    {
        return dual<T>(in);
    },
    [](const value_base& base)
    {
        if(base.name == "x")
            return dual<float>(std::get<float>(base.concrete), 1.f);
        else
            return dual<float>(std::get<float>(base.concrete), 0.f);
    });

    //our function is 2x^2 + 2
    //our real part is therefore 10
    //our derivative is D(2x^2 + x), which gives 4x + 1
    //which gives 9
    std::cout << as_dual.real << " " << as_dual.dual << std::endl;
    #endif
}

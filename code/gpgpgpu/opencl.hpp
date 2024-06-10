#ifndef OPENCL_HPP_INCLUDED
#define OPENCL_HPP_INCLUDED

#include <cl/cl.h>
#include <CL/cl_gl.h>
#include <GL/glew.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <GL/glx.h>
#endif

struct opencl
{
    cl_context ctx;
    cl_device_id selected_device;
};

std::string get_platform_name(cl_platform_id id)
{
    std::string val;

    size_t length = 0;

    clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &length);

    val.resize(length + 1);

    clGetPlatformInfo(id, CL_PLATFORM_NAME, length, (void*)val.data(), nullptr);
    return val;
}

cl_platform_id get_platform_ids()
{
    std::optional<cl_platform_id> ret;

    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);

    if(err != CL_SUCCESS)
        throw std::runtime_error("Bad clGetPlatformIDs call " + std::to_string(err));

    if(num_platforms == 0)
        throw std::runtime_error("No available platforms");

    std::vector<cl_platform_id> clPlatformIDs;
    clPlatformIDs.resize(num_platforms);

    clGetPlatformIDs(num_platforms, &clPlatformIDs[0], nullptr);

    for(int i = 0; i < (int)num_platforms; i++)
    {
        std::string name = get_platform_name(clPlatformIDs[i]);

        if(name.contains("NVIDIA") || name.contains("AMD"))
            ret = clPlatformIDs[i];
    }

    if(!ret.has_value())
        ret = clPlatformIDs[num_platforms-1];

    if(!ret.has_value())
        throw std::runtime_error("Did not find platform");

    return ret.value();
}

opencl boot_opencl()
{
    opencl ret;

    cl_platform_id pid = get_platform_ids();

    cl_uint num_devices = 0;
    cl_device_id devices[100] = {};

    clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 1, devices, &num_devices);

    ret.selected_device = devices[0];

    #ifdef _WIN32
    cl_context_properties props[] =
    {
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)pid,
        0
    };
    #elif defined(__APPLE__)
    CGLContextObj cgl_context = CGLGetCurrentContext();
    CGLShareGroupObj cgl_share_group = CGLGetShareGroup(cgl_current_context);

    cl_context_properties properties[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
        (cl_context_properties) cgl_share_group,
        0
    };
    #else
    cl_context_properties props[] =
    {
        CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)pid,
        0
    };
    #endif

    cl_int error = 0;

    cl_context ctx = clCreateContext(props, 1, &ret.selected_device, nullptr, nullptr, &error);

    if(error != CL_SUCCESS)
        throw std::runtime_error("Failed to create context " + std::to_string(error));

    ret.ctx = ctx;

    return ret;
}

cl_mem make_buffer(opencl& cl, size_t size)
{
    return clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE, size, nullptr, nullptr);
}

cl_mem make_image(opencl& cl, int width, int height)
{
    cl_image_format fmt;
    fmt.image_channel_order = CL_RGBA;
    fmt.image_channel_data_type = CL_FLOAT;

    cl_image_desc desc = {};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = width;
    desc.image_height = height;
    desc.image_depth = 1;
    desc.image_array_size = 1;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    desc.mem_object = nullptr;

    return clCreateImage(cl.ctx, CL_MEM_READ_WRITE, &fmt, &desc, nullptr, nullptr);
}

cl_mem make_shared_image(opencl& cl, int texture_id)
{
    return clCreateFromGLTexture(cl.ctx, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, texture_id, nullptr);
}

#endif // OPENCL_HPP_INCLUDED

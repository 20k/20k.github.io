#include <iostream>
#include "../common/single_source.hpp"
#include "raytracer.hpp"
#include "../common/toolkit/opencl.hpp"
#include <SFML/Graphics.hpp>
#include <vec/vec.hpp>

struct camera
{
    quat rot;

    camera()
    {
        rot.load_from_axis_angle({1, 0, 0, -M_PI/2});
    }

    void rotate(vec2f mouse_delta)
    {
        quat local_camera_quat = rot;

        if(mouse_delta.x() != 0)
        {
            quat q;
            q.load_from_axis_angle((vec4f){0, 0, -1, mouse_delta.x()});

            local_camera_quat = q * local_camera_quat;
        }

        {
            vec3f right = rot_quat((vec3f){1, 0, 0}, local_camera_quat);

            if(mouse_delta.y() != 0)
            {
                quat q;
                q.load_from_axis_angle({right.x(), right.y(), right.z(), mouse_delta.y()});

                local_camera_quat = q * local_camera_quat;
            }
        }

        rot = local_camera_quat;
    }
};

//https://www2.mpia-hd.mpg.de/homes/tmueller/pdfs/catalogue_2014-05-21.pdf 2.2.1
metric<valuef, 4, 4> schwarzschild_metric(const tensor<valuef, 4>& position) {
    valuef rs = 1;
    valuef r = position[1];
    valuef theta = position[2];

    metric<valuef, 4, 4> m;
    /*m[0, 0] = -(1-rs/r);
    m[1, 1] = 1/(1-rs/r);*/

    m[0, 0] = -(1-rs/r);
    m[1, 0] = 1;
    m[0, 1] = 1;

    m[2, 2] = r*r;
    m[3, 3] = r*r * sin(theta)*sin(theta);

    return m;
}

cl::image load_background(cl::context ctx, cl::command_queue cqueue)
{
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

cl::kernel make_kernel(cl::context& ctx, const std::string& str, const std::string& name, const std::string& options)
{
    cl::program prog(ctx, str, false);
    prog.build(ctx, options);
    return cl::kernel(prog, name);
}

int main()
{
    int screen_width = 1920;
    int screen_height = 1080;

    sf::VideoMode mode(screen_width, screen_height);
    sf::RenderWindow win(mode, "I am a black hole");

    cl::context ctx;

    cl::command_queue cqueue(ctx);

    std::string kernel = value_impl::make_function(opencl_raytrace<schwarzschild_metric>, "raytrace");

    std::cout << kernel << std::endl;

    std::string tetrad_calc = value_impl::make_function(build_initial_tetrads<schwarzschild_metric>, "tetrad");

    std::cout << tetrad_calc << std::endl;

    std::string geodesic_src = value_impl::make_function(trace_geodesic<schwarzschild_metric>, "trace_geodesic");

    std::cout << geodesic_src << std::endl;

    std::string transport_src = value_impl::make_function(parallel_transport_tetrads<schwarzschild_metric>, "transport");

    std::cout << transport_src << std::endl;

    std::string interpolate_src = value_impl::make_function(interpolate, "interpolate");

    std::cout << interpolate_src << std::endl;

    cl::kernel tetrad_kern = make_kernel(ctx, tetrad_calc, "tetrad", "");
    cl::kernel trace_kern = make_kernel(ctx, kernel, "raytrace", "-cl-fast-relaxed-math");
    cl::kernel geodesic_kern = make_kernel(ctx, geodesic_src, "trace_geodesic", "");
    cl::kernel transport_kern = make_kernel(ctx, transport_src, "transport", "");
    cl::kernel interpolate_kern = make_kernel(ctx, interpolate_src, "interpolate", "");

    float pi = std::numbers::pi_v<float>;

    cl_float4 camera_pos = {0, 7, pi/2, -pi/2};
    cl::buffer gpu_camera_pos(ctx);
    gpu_camera_pos.alloc(sizeof(cl_float4));

    std::array<cl::buffer, 4> tetrads{ctx, ctx, ctx, ctx};

    for(int i=0; i < tetrads.size(); i++)
        tetrads[i].alloc(sizeof(cl_float4));

    cl::image background = load_background(ctx, cqueue);

    int background_width = background.size<2>().x();
    int background_height = background.size<2>().y();

    sf::Texture tex;
    tex.create(screen_width, screen_height);

    unsigned int handle = tex.getNativeHandle();

    cl::gl_rendertexture screen(ctx);
    screen.create_from_texture(handle);

    cl::buffer positions(ctx);
    cl::buffer velocities(ctx);
    cl::buffer steps(ctx);
    int max_writes = 1024 * 1024;

    positions.alloc(sizeof(cl_float4) * max_writes);
    velocities.alloc(sizeof(cl_float4) * max_writes);
    steps.alloc(sizeof(cl_int));

    std::array<cl::buffer, 3> transported_tetrads{ctx, ctx, ctx};

    for(auto& e : transported_tetrads)
        e.alloc(sizeof(cl_float4) * max_writes);

    std::array<cl::buffer, 4> final_tetrads{ctx, ctx, ctx, ctx};

    for(auto& e : final_tetrads)
        e.alloc(sizeof(cl_float4));

    cl::buffer final_camera_position(ctx);
    final_camera_position.alloc(sizeof(cl_float4));

    float desired_proper_time = 0.f;

    camera cam;

    sf::Keyboard key;

    while(win.isOpen())
    {
        sf::Event evt;

        while(win.pollEvent(evt))
        {
            if(evt.type == sf::Event::Closed)
                win.close();
        }

        desired_proper_time += 0.1f;

        if(key.isKeyPressed(sf::Keyboard::D))
            cam.rotate({0.1f, 0.f});
        if(key.isKeyPressed(sf::Keyboard::A))
            cam.rotate({-0.1f, 0.f});

        if(key.isKeyPressed(sf::Keyboard::S))
            cam.rotate({0.f, 0.1f});
        if(key.isKeyPressed(sf::Keyboard::W))
            cam.rotate({0.f, -0.1f});

        sf::Clock clk;

        {
            cl::args args;
            args.push_back(camera_pos);
            args.push_back(gpu_camera_pos);
            args.push_back(tetrads[0]);
            args.push_back(tetrads[1]);
            args.push_back(tetrads[2]);
            args.push_back(tetrads[3]);

            tetrad_kern.set_args(args);

            cqueue.exec(tetrad_kern, {1}, {1}, {});
        }

        {
            cl::args args;
            args.push_back(gpu_camera_pos);
            args.push_back(tetrads[0]);
            args.push_back(positions);
            args.push_back(velocities);
            args.push_back(steps);
            args.push_back(max_writes);

            geodesic_kern.set_args(args);

            cqueue.exec(geodesic_kern, {1}, {1}, {});
        }

        {
            cl::args args;
            args.push_back(tetrads[1], tetrads[2], tetrads[3]);
            args.push_back(positions, velocities);
            args.push_back(steps);
            args.push_back(transported_tetrads[0], transported_tetrads[1], transported_tetrads[2]);

            transport_kern.set_args(args);

            cqueue.exec(transport_kern, {1}, {1}, {});
        }

        {
            //float desired_proper_time = 10.f;

            cl::args args;
            args.push_back(positions);
            args.push_back(velocities, transported_tetrads[0], transported_tetrads[1], transported_tetrads[2]);
            args.push_back(steps);
            args.push_back(desired_proper_time);
            args.push_back(final_camera_position);
            args.push_back(final_tetrads[0], final_tetrads[1], final_tetrads[2], final_tetrads[3]);

            interpolate_kern.set_args(args);

            cqueue.exec(interpolate_kern, {1}, {1}, {});
        }

        printf("Pos %f\n", final_camera_position.read<float>(cqueue)[1]);

        {
            cl_float4 q = {cam.rot.q.v[0], cam.rot.q.v[1], cam.rot.q.v[2], cam.rot.q.v[3]};

            screen.acquire(cqueue);

            cl::args args;
            args.push_back(screen_width, screen_height);
            args.push_back(background);
            args.push_back(screen);
            args.push_back(background_width);
            args.push_back(background_height);
            args.push_back(final_tetrads[0]);
            args.push_back(final_tetrads[1]);
            args.push_back(final_tetrads[2]);
            args.push_back(final_tetrads[3]);
            args.push_back(final_camera_position);
            args.push_back(q);

            trace_kern.set_args(args);

            cqueue.exec(trace_kern, {screen_width, screen_height}, {8,8}, {});

            screen.unacquire(cqueue);
        }

        cqueue.block();

        std::cout << "Time " << clk.getElapsedTime().asMicroseconds() / 1000. << std::endl;

        sf::Sprite spr(tex);

        win.draw(spr);

        win.display();

        sf::sleep(sf::milliseconds(8));
    }

    return 0;
}

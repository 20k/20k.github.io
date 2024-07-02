#include <iostream>
#include "../common/single_source.hpp"
#include "raytracer.hpp"
#include "../common/toolkit/opencl.hpp"
#include <SFML/Graphics.hpp>
#include <vec/vec.hpp>
#include "accretion_disk.hpp"
#include "blackbody.hpp"
#include "metrics.hpp"

struct camera
{
    tensor<float, 4> pos;
    quat rot;

    camera()
    {
        float pi = std::numbers::pi_v<float>;

        pos = {0, 5, pi/2, -pi/2};
        rot.load_from_axis_angle({1, 0, 0, -M_PI/2});
    }

    tensor<float, 4> get_position()
    {
        return pos;
    }

    void move(tensor<float, 3> direction)
    {
        tensor<float, 3> cart = spherical_to_cartesian(pos.yzw());
        cart += direction;

        auto spatial = cartesian_to_spherical(cart);

        pos.y() = spatial.x();
        pos.z() = spatial.y();
        pos.w() = spatial.z();

        pos.y() = clamp(pos.y(), -UNIVERSE_SIZE, UNIVERSE_SIZE);
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

cl::kernel make_kernel(cl::context& ctx, const std::string& str, const std::string& name, const std::string& options)
{
    cl::program prog(ctx, str, false);
    prog.build(ctx, options);
    return cl::kernel(prog, name);
}

int main()
{
    accretion_disk accrete = make_accretion_disk_kerr(BH_MASS, BH_SPIN);

    int screen_width = 1920/2;
    int screen_height = 1080/2;

    sf::VideoMode mode(screen_width, screen_height);
    sf::RenderWindow win(mode, "I am a black hole");

    cl::context ctx;

    cl::command_queue cqueue(ctx);

    std::string kernel = value_impl::make_function(opencl_raytrace<get_metric>, "raytrace");

    std::cout << kernel << std::endl;

    std::string tetrad_calc = value_impl::make_function(build_initial_tetrads<get_metric, metric_to_spherical, spherical_to_metric>, "tetrad");

    std::cout << tetrad_calc << std::endl;

    std::string geodesic_src = value_impl::make_function(trace_geodesic<get_metric>, "trace_geodesic");

    std::cout << geodesic_src << std::endl;

    std::string transport_src = value_impl::make_function(parallel_transport_tetrads<get_metric>, "transport");

    std::cout << transport_src << std::endl;

    std::string interpolate_src = value_impl::make_function(interpolate, "interpolate");

    std::cout << interpolate_src << std::endl;

    cl::kernel tetrad_kern = make_kernel(ctx, tetrad_calc, "tetrad", "");
    cl::kernel trace_kern = make_kernel(ctx, kernel, "raytrace", "-cl-fast-relaxed-math");
    cl::kernel geodesic_kern = make_kernel(ctx, geodesic_src, "trace_geodesic", "");
    cl::kernel transport_kern = make_kernel(ctx, transport_src, "transport", "");
    cl::kernel interpolate_kern = make_kernel(ctx, interpolate_src, "interpolate", "");

    cl::buffer gpu_camera_pos(ctx);
    gpu_camera_pos.alloc(sizeof(cl_float4));

    std::array<cl::buffer, 4> tetrads{ctx, ctx, ctx, ctx};

    for(int i=0; i < tetrads.size(); i++)
        tetrads[i].alloc(sizeof(cl_float4));

    cl::image background = load_background(ctx, cqueue, "../common/nasa.png");
    cl::image background2 = load_background(ctx, cqueue, "../common/wes.png");

    int background_width = background.size<2>().x();
    int background_height = background.size<2>().y();

    sf::Texture tex;
    tex.create(screen_width, screen_height);

    unsigned int handle = tex.getNativeHandle();

    cl::gl_rendertexture screen(ctx);
    screen.create_from_texture(handle);

    cl::buffer accretion_buf(ctx);
    accretion_buf.alloc(accrete.brightness.size() * sizeof(cl_float4));

    {
        std::vector<cl_float4> as_float4;

        for(auto& i : accrete.brightness)
            as_float4.push_back({i.x(), i.y(), i.z(), 0});

        accretion_buf.write(cqueue, as_float4);
    }


    cl::buffer positions(ctx);
    cl::buffer velocities(ctx);
    cl::buffer steps(ctx);
    int max_writes = 1024 * 1024;

    positions.alloc(sizeof(cl_float4) * max_writes);
    velocities.alloc(sizeof(cl_float4) * max_writes);
    steps.alloc(sizeof(cl_int));

    std::array<cl::buffer, 4> transported_tetrads{ctx, ctx, ctx, ctx};

    for(auto& e : transported_tetrads)
        e.alloc(sizeof(cl_float4) * max_writes);

    std::array<cl::buffer, 4> final_tetrads{ctx, ctx, ctx, ctx};

    for(auto& e : final_tetrads)
        e.alloc(sizeof(cl_float4));

    cl::buffer final_camera_position(ctx);
    final_camera_position.alloc(sizeof(cl_float4));

    float desired_proper_time = 0.f;

    auto bbody_table = blackbody_table();

    cl::buffer blackbody(ctx);
    blackbody.alloc(sizeof(cl_float4) * bbody_table.size());

    cl::buffer temperature_data(ctx);
    temperature_data.alloc(accrete.temperature.size() * sizeof(cl_float));
    temperature_data.write(cqueue, accrete.temperature);

    {
        std::vector<cl_float4> data;

        for(auto& i : bbody_table)
        {
            data.push_back({i.x(), i.y(), i.z(), 0});
        }

        blackbody.write(cqueue, data);
    }

    camera cam;
    cam.move({0, 0, 8});
    cam.move({0, -14, 0});

    std::cout << "CAM " << cam.get_position().x() << " " << cam.get_position().y() << " " << cam.get_position().z() << " " << cam.get_position().w() << std::endl;

    cam.pos.x() = 22.676;
    cam.pos.y() = 19.745;
    cam.pos.z() = 1.172;
    cam.pos.w() = -1.583;

    //std::cout << "C " << cart.x() << " " << cart.y() << " " << cart.z() << std::endl;

    sf::Keyboard key;

    sf::Clock elapsed;

    while(win.isOpen())
    {
        sf::Event evt;

        while(win.pollEvent(evt))
        {
            if(evt.type == sf::Event::Closed)
                win.close();
        }

        //desired_proper_time += 2.25f * elapsed.restart().asMicroseconds() / 1000. / 1000.;

        float ptime_diff = 0;

        if(key.isKeyPressed(sf::Keyboard::Num1))
            ptime_diff += 0.1f;

        if(key.isKeyPressed(sf::Keyboard::Num2))
            ptime_diff -= 0.1f;

        if(key.isKeyPressed(sf::Keyboard::Num3))
            ptime_diff += 1.f;

        if(key.isKeyPressed(sf::Keyboard::Num4))
            ptime_diff -= 1.f;

        if(key.isKeyPressed(sf::Keyboard::Num5))
            ptime_diff += 10.f;

        if(key.isKeyPressed(sf::Keyboard::Num6))
            ptime_diff -= 10.f;

        if(key.isKeyPressed(sf::Keyboard::LControl))
            ptime_diff *= 0.1f;

        if(key.isKeyPressed(sf::Keyboard::LAlt))
            ptime_diff *= 0.1f;

        //if(key.isKeyPressed(sf::Keyboard::Num0))
        //    desired_proper_time = 96.824f;

        if(key.isKeyPressed(sf::Keyboard::Num0))
            desired_proper_time = 96.9f;

        desired_proper_time += ptime_diff;

        printf("DPT %f\n", desired_proper_time);

        if(key.isKeyPressed(sf::Keyboard::J))
            cam.rotate({0.1f, 0.f});
        if(key.isKeyPressed(sf::Keyboard::L))
            cam.rotate({-0.1f, 0.f});

        if(key.isKeyPressed(sf::Keyboard::I))
            cam.rotate({0.f, 0.1f});
        if(key.isKeyPressed(sf::Keyboard::K))
            cam.rotate({0.f, -0.1f});

        float kspeed = 0.1f;

        if(key.isKeyPressed(sf::Keyboard::A))
            cam.move({-kspeed, 0, 0});

        if(key.isKeyPressed(sf::Keyboard::D))
            cam.move({kspeed, 0, 0});

        if(key.isKeyPressed(sf::Keyboard::W))
            cam.move({0, kspeed, 0});

        if(key.isKeyPressed(sf::Keyboard::S))
            cam.move({0, -kspeed, 0});

        if(key.isKeyPressed(sf::Keyboard::Q))
            cam.move({0, 0, kspeed});

        if(key.isKeyPressed(sf::Keyboard::E))
            cam.move({0, 0, -kspeed});

        sf::Clock clk;


        {
            //cl_float3 local_velocity = {0.001, 0.002, -0.8f};

            cl_float3 local_velocity = {0,0,0};

            cl::args args;
            args.push_back(cam.get_position());
            args.push_back(local_velocity);
            args.push_back(gpu_camera_pos);
            args.push_back(tetrads[0]);
            args.push_back(tetrads[1]);
            args.push_back(tetrads[2]);
            args.push_back(tetrads[3]);

            tetrad_kern.set_args(args);

            cqueue.exec(tetrad_kern, {1}, {1}, {});
        }

        #define PARALLEL_TRANSPORT
        #ifdef PARALLEL_TRANSPORT
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
            args.push_back(tetrads[0], tetrads[1], tetrads[2], tetrads[3]);
            args.push_back(positions, velocities);
            args.push_back(steps);
            args.push_back(transported_tetrads[0], transported_tetrads[1], transported_tetrads[2], transported_tetrads[3]);

            transport_kern.set_args(args);

            cqueue.exec(transport_kern, {1}, {1}, {});
        }

        {
            //float desired_proper_time = 10.f;

            cl::args args;
            args.push_back(positions);
            args.push_back(velocities);
            args.push_back(transported_tetrads[0], transported_tetrads[1], transported_tetrads[2], transported_tetrads[3]);
            args.push_back(steps);
            args.push_back(desired_proper_time);
            args.push_back(final_camera_position);
            args.push_back(final_tetrads[0], final_tetrads[1], final_tetrads[2], final_tetrads[3]);

            interpolate_kern.set_args(args);

            cqueue.exec(interpolate_kern, {1}, {1}, {});
        }

        //printf("Pos %f\n", final_camera_position.read<float>(cqueue)[1]);
        #else
        cl::copy(cqueue, tetrads[0], final_tetrads[0]);
        cl::copy(cqueue, tetrads[1], final_tetrads[1]);
        cl::copy(cqueue, tetrads[2], final_tetrads[2]);
        cl::copy(cqueue, tetrads[3], final_tetrads[3]);
        cl::copy(cqueue, gpu_camera_pos, final_camera_position);
        #endif

        {
            cl_float4 q = {cam.rot.q.v[0], cam.rot.q.v[1], cam.rot.q.v[2], cam.rot.q.v[3]};

            screen.acquire(cqueue);

            cl::args args;
            args.push_back(screen_width, screen_height);
            args.push_back(background);
            args.push_back(background2);
            args.push_back(screen);
            args.push_back(accretion_buf);
            args.push_back(background_width);
            args.push_back(background_height);
            args.push_back(final_tetrads[0]);
            args.push_back(final_tetrads[1]);
            args.push_back(final_tetrads[2]);
            args.push_back(final_tetrads[3]);
            args.push_back(final_camera_position);
            args.push_back(q);
            args.push_back(blackbody);
            args.push_back(temperature_data);

            trace_kern.set_args(args);

            cqueue.exec(trace_kern, {screen_width, screen_height}, {8,8}, {});

            screen.unacquire(cqueue);
        }

        cqueue.block();

        std::cout << "Time " << clk.getElapsedTime().asMicroseconds() / 1000. << std::endl;

        sf::Sprite spr(tex);

        win.draw(spr);

        win.display();

        //sf::sleep(sf::milliseconds(8));
    }

    return 0;
}

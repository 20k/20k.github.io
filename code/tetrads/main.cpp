#include <iostream>
#include "../common/single_source.hpp"
#include "../common/raytracer.hpp"
#include "../common/toolkit/opencl.hpp"
#include <SFML/Graphics.hpp>

//https://www2.mpia-hd.mpg.de/homes/tmueller/pdfs/catalogue_2014-05-21.pdf 2.2.1
metric<valuef, 4, 4> schwarzschild_metric(const tensor<valuef, 4>& position) {
    valuef rs = 1;
    valuef r = position[1];
    valuef theta = position[2];

    metric<valuef, 4, 4> m;
    m[0, 0] = -(1-rs/r);
    m[1, 1] = 1/(1-rs/r);
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

int main()
{
    int screen_width = 1000;
    int screen_height = 800;

    sf::VideoMode mode(screen_width, screen_height);
    sf::RenderWindow win(mode, "I am a black hole");

    cl::context ctx;

    cl::command_queue cqueue(ctx);

    std::string kernel = value_impl::make_function(opencl_raytrace<schwarzschild_metric>, "raytrace");

    std::cout << kernel << std::endl;

    std::string tetrad_calc = value_impl::make_function(build_initial_tetrads<schwarzschild_metric>, "tetrad");

    std::cout << tetrad_calc << std::endl;

    cl::program tetrad_p(ctx, tetrad_calc, false);
    tetrad_p.build(ctx, "");
    cl::kernel tetrad_kern(tetrad_p, "tetrad");

    cl::program trace_p(ctx, kernel, false);
    trace_p.build(ctx, "");
    cl::kernel trace_kern(trace_p, "raytrace");

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

    while(win.isOpen())
    {
        sf::Event evt;

        while(win.pollEvent(evt))
        {
            if(evt.type == sf::Event::Closed)
                win.close();
        }

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
            screen.acquire(cqueue);

            cl::args args;
            args.push_back(screen_width, screen_height);
            args.push_back(background);
            args.push_back(screen);
            args.push_back(background_width);
            args.push_back(background_height);
            args.push_back(tetrads[0]);
            args.push_back(tetrads[1]);
            args.push_back(tetrads[2]);
            args.push_back(tetrads[3]);
            args.push_back(gpu_camera_pos);

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

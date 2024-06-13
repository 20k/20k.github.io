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

int main()
{
    int screen_width = 1000;
    int screen_height = 800;

    sf::VideoMode mode(screen_width, screen_height);
    sf::RenderWindow win(mode, "I am a black hole");

    cl::context ctx;

    cl::command_queue cqueue(ctx);

    std::string kernel = value_impl::make_function(opencl_raytrace<schwarzschild_metric>, "opencl_raytrace");

    std::cout << kernel << std::endl;

    while(win.isOpen())
    {
        sf::Event evt;

        while(win.pollEvent(evt))
        {
            if(evt.type == sf::Event::Closed)
                win.close();
        }

        sf::Clock clk;

        cqueue.block();

        win.display();

        sf::sleep(sf::milliseconds(8));
    }

    return 0;
}

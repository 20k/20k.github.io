#include <iostream>
#include "../common/single_source.hpp"
#include "../common/raytracer.hpp"
#include "../common/toolkit/opencl.hpp"
#include <SFML/Graphics.hpp>


int main()
{
    int screen_width = 1000;
    int screen_height = 800;

    sf::VideoMode mode(screen_width, screen_height);
    sf::RenderWindow win(mode, "I am a black hole");

    cl::context ctx;

    cl::command_queue cqueue(ctx);

    while(win.isOpen())
    {
        sf::Event evt;

        while(win.pollEvent(evt))
        {
            if(evt.type == sf::Event::Closed)
                win.close();
        }

        win.display();

        sf::sleep(sf::milliseconds(8));
    }

    return 0;
}

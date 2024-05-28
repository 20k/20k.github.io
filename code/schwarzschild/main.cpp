#include <iostream>
#include <vec/tensor.hpp>
#include <numbers>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <thread>
#include <vector>

metric<float, 4, 4> schwarzschild_metric(const tensor<float, 4>& position) {
    float rs = 1;

    float r = position[1];
    float theta = position[2];

    metric<float, 4, 4> m;
    m[0, 0] = -(1-rs/r);
    m[1, 1] = 1/(1-rs/r);
    m[2, 2] = r*r;
    m[3, 3] = r*r * std::sin(theta)*std::sin(theta);

    return m;
}

struct tetrad
{
    std::array<tensor<float, 4>, 4> v;
};

tetrad calculate_schwarzschild_tetrad(const tensor<float, 4>& position) {
    float rs = 1;
    float r = position[1];
    float theta = position[2];

    tensor<float, 4> et = {1/std::sqrt(1 - rs/r), 0, 0, 0};
    tensor<float, 4> er = {0, std::sqrt(1 - rs/r), 0, 0};
    tensor<float, 4> etheta = {0, 0, 1/r, 0};
    tensor<float, 4> ephi = {0, 0, 0, 1/(r * std::sin(theta))};

    return {et, er, etheta, ephi};
}

tensor<float, 3> get_ray_through_pixel(int sx, int sy, int screen_width, int screen_height, float fov_degrees) {
    float fov_rad = (fov_degrees / 360.f) * 2 * std::numbers::pi_v<float>;
    float f_stop = (screen_width/2) / tan(fov_rad/2);

    tensor<float, 3> pixel_direction = {(float)(sx - screen_width/2), (float)(sy - screen_height/2), f_stop};
    //pixel_direction = rot_quat(pixel_direction, camera_quat); //if you have quaternions, or some rotation library, rotate your pixel direction here by your cameras rotation

    return pixel_direction.norm();
}

struct geodesic
{
    tensor<float, 4> position;
    tensor<float, 4> velocity;
};

geodesic make_lightlike_geodesic(const tensor<float, 4>& position, const tensor<float, 3>& direction, const tetrad& tetrads) {
    geodesic g;
    g.position = position;
    g.velocity = tetrads.v[0] * -1 //Flipped time component, we're tracing backwards in time
               + tetrads.v[1] * direction[0]
               + tetrads.v[2] * direction[1]
               + tetrads.v[3] * direction[2];

    return g;
}

auto diff(auto&& func, const tensor<float, 4>& position, int direction) {
    auto p_up = position;
    auto p_lo = position;

    float h = 0.00001f;

    p_up[direction] += h;
    p_lo[direction] -= h;

    auto up = func(p_up);
    auto lo = func(p_lo);

    return (func(p_up) - func(p_lo)) / (2 * h);
}

tensor<float, 4, 4, 4> calculate_christoff2(const tensor<float, 4>& position, auto&& get_metric) {
    metric<float, 4, 4> metric = get_metric(position);
    inverse_metric<float, 4, 4> metric_inverse = metric.invert();
    tensor<float, 4, 4, 4> metric_diff; ///uses the index signature, diGjk

    for(int i=0; i < 4; i++) {
        auto differentiated = diff(get_metric, position, i);

        for(int j=0; j < 4; j++) {
            for(int k=0; k < 4; k++) {
                metric_diff[i, j, k] = differentiated[j, k];
            }
        }
    }

    tensor<float, 4, 4, 4> Gamma;

    for(int mu = 0; mu < 4; mu++)
    {
        for(int al = 0; al < 4; al++)
        {
            for(int be = 0; be < 4; be++)
            {
                float sum = 0;

                for(int sigma = 0; sigma < 4; sigma++)
                {
                    sum += 0.5f * metric_inverse[mu, sigma] * (metric_diff[be, sigma, al] + metric_diff[al, sigma, be] - metric_diff[sigma, al, be]);
                }

                Gamma[mu, al, be] = sum;
            }
        }
    }

    //note that for simplicities sake, we fully calculate all the christoffel symbol components
    //but the lower two indices are symmetric, and can be mirrored to save significant calculations
    return Gamma;
}

tensor<float, 4> calculate_acceleration_of(const tensor<float, 4>& X, const tensor<float, 4>& v, auto&& get_metric) {
    tensor<float, 4, 4, 4> christoff2 = calculate_christoff2(X, get_metric);

    tensor<float, 4> acceleration;

    for(int mu = 0; mu < 4; mu++) {
        float sum = 0;

        for(int al = 0; al < 4; al++) {
            for(int be = 0; be < 4; be++) {
                sum += -christoff2[mu, al, be] * v[al] * v[be];
            }
        }

        acceleration[mu] = sum;
    }

    return acceleration;
}

tensor<float, 4> calculate_schwarzschild_acceleration(const tensor<float, 4>& X, const tensor<float, 4>& v) {
    return calculate_acceleration_of(X, v, schwarzschild_metric);
}

struct integration_result {
    enum hit_type {
        ESCAPED,
        EVENT_HORIZON,
        UNFINISHED
    };

    hit_type type = UNFINISHED;
    geodesic g;
};

integration_result integrate(geodesic& g, bool debug) {
    integration_result result;

    float dt = 0.005f;
    float rs = 1;
    float start_time = g.position[0];

    for(int i=0; i < 100000; i++) {
        tensor<float, 4> acceleration = calculate_schwarzschild_acceleration(g.position, g.velocity);

        g.velocity += acceleration * dt;
        g.position += g.velocity * dt;

        float radius = g.position[1];

        if(radius > 10) {
            //ray escaped
            result.g = g;
            result.type = integration_result::ESCAPED;

            return result;
        }

        if(radius <= rs + 0.0001f || g.position[0] > start_time + 1000) {
            //ray has very likely hit the event horizon
            result.g = g;
            result.type = integration_result::EVENT_HORIZON;

            return result;
        }
    }

    result.g = g;
    return result;
}

tensor<float, 2> angle_to_tex(const tensor<float, 2>& angle)
{
    float pi = std::numbers::pi_v<float>;

    float thetaf = std::fmod(angle[0], 2 * pi);
    float phif = angle[1];

    if(thetaf >= pi)
    {
        phif += pi;
        thetaf -= pi;
    }

    phif = std::fmod(phif, 2 * pi);

    float sxf = phif / (2 * pi);
    float syf = thetaf / pi;

    sxf += 0.5f;

    return {sxf, syf};
}

tensor<float, 3> render_pixel(int x, int y, int screen_width, int screen_height, const sf::Image& background)
{
    tensor<float, 3> ray_direction = get_ray_through_pixel(x, y, screen_width, screen_height, 90);

    float pi = std::numbers::pi_v<float>;

    tensor<float, 4> camera_position = {0, 5, pi/2, -pi/2};

    tetrad tetrads = calculate_schwarzschild_tetrad(camera_position);

    //so, the tetrad vectors give us a basis, that points in the direction t, r, theta, and phi, because schwarzschild is diagonal
    //we'd like the ray to point towards the black hole: this means we make +z point towards -r, +y point towards +theta, and +x point towards +phi
    tensor<float, 3> modified_ray = {-ray_direction[2], ray_direction[1], ray_direction[0]};

    geodesic my_geodesic = make_lightlike_geodesic(camera_position, modified_ray, tetrads);

    integration_result result = integrate(my_geodesic, x == 300 && y == 150);

    if(result.type == integration_result::EVENT_HORIZON || result.type == integration_result::UNFINISHED)
        return {0,0,0};
    else
    {
        float theta = my_geodesic.position[2];
        float phi = my_geodesic.position[3];

        tensor<float, 2> texture_coordinate = angle_to_tex({theta, phi});

        int tx = (int)(texture_coordinate[0] * background.getSize().x) % background.getSize().x;
        int ty = (int)(texture_coordinate[1] * background.getSize().y) % background.getSize().y;

        auto icol = background.getPixel(tx, ty);

        return {icol.r/255.f, icol.g/255.f, icol.b/255.f};;
    }
}

int main()
{
    int screen_width = 400;
    int screen_height = 300;

    sf::VideoMode mode(screen_width, screen_height);
    sf::RenderWindow win(mode, "I am a black hole");

    sf::Image background;
    background.loadFromFile("nasa.png");

    std::vector<tensor<float, 3>> result;
    result.resize(screen_width*screen_height);

    std::vector<tensor<int, 2>> work;
    std::atomic_int assigned_work{0};
    int max_work = screen_width*screen_height;

    for(int i=0; i < screen_width; i++)
    {
        for(int j=0; j < screen_height; j++)
        {
            work.push_back({i, j});
        }
    }

    int thread_count = std::thread::hardware_concurrency();
    std::vector<std::jthread> threads;

    for(int t=0; t < thread_count; t++)
    {
        threads.emplace_back([&]()
        {
            while(1)
            {
                int work_size = 2048;
                int start_pixel = assigned_work.fetch_add(work_size);

                if(start_pixel >= max_work)
                    break;

                for(int idx = start_pixel; idx < start_pixel + work_size && idx < max_work; idx++)
                {
                    tensor<int, 2> pixel = work[idx];

                    result[pixel[1] * screen_width + pixel[0]] = render_pixel(pixel[0], pixel[1], screen_width, screen_height, background);
                }
            }
        });
    }

    for(auto& i : threads)
    {
        i.join();
    }

    sf::Image img;
    img.create(screen_width, screen_height);

    for(int j=0; j < screen_height; j++)
    {
        for(int i=0; i < screen_width; i++)
        {
            tensor<float, 3> colour = result[j * screen_width + i];

            colour[0] = std::clamp(colour[0], 0.f, 1.f);
            colour[1] = std::clamp(colour[1], 0.f, 1.f);
            colour[2] = std::clamp(colour[2], 0.f, 1.f);

            sf::Color sf_colour(colour[0] * 255.f, colour[1] * 255.f, colour[2] * 255.f, 255);

            img.setPixel(i, j, sf_colour);
        }
    }

    sf::Texture tex;
    tex.loadFromImage(img);

    while(win.isOpen())
    {
        sf::Event evt;

        while(win.pollEvent(evt))
        {
            if(evt.type == sf::Event::Closed)
                win.close();
        }

        sf::Sprite sprite(tex);
        win.draw(sprite);

        win.display();
    }

    return 0;
}

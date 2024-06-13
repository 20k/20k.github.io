#ifndef SCHWARZSCHILD_SINGLE_SOURCE_HPP_INCLUDED
#define SCHWARZSCHILD_SINGLE_SOURCE_HPP_INCLUDED

#include "../common/vec/tensor.hpp"
#include "single_source.hpp"

using valuef = value<float>;
using valuei = value<int>;

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

struct tetrad
{
    std::array<tensor<valuef, 4>, 4> v;
};

//https://www2.mpia-hd.mpg.de/homes/tmueller/pdfs/catalogue_2014-05-21.pdf 2.2.6
tetrad calculate_schwarzschild_tetrad(const tensor<valuef, 4>& position) {
    valuef rs = 1;
    valuef r = position[1];
    valuef theta = position[2];

    tensor<valuef, 4> et = {1/sqrt(1 - rs/r), 0, 0, 0};
    tensor<valuef, 4> er = {0, sqrt(1 - rs/r), 0, 0};
    tensor<valuef, 4> etheta = {0, 0, 1/r, 0};
    tensor<valuef, 4> ephi = {0, 0, 0, 1/(r * sin(theta))};

    return {et, er, etheta, ephi};
}

tensor<valuef, 3> get_ray_through_pixel(valuei sx, valuei sy, valuei screen_width, valuei screen_height, float fov_degrees) {
    float fov_rad = (fov_degrees / 360.f) * 2 * std::numbers::pi_v<float>;
    valuef f_stop = (screen_width/2).to<float>() / tan(fov_rad/2);

    tensor<valuef, 3> pixel_direction = {(sx - screen_width/2).to<float>(), (sy - screen_height/2).to<float>(), f_stop};
    //pixel_direction = rot_quat(pixel_direction, camera_quat); //if you have quaternions, or some rotation library, rotate your pixel direction here by your cameras rotation

    return pixel_direction.norm();
}

struct geodesic
{
    tensor<valuef, 4> position;
    tensor<valuef, 4> velocity;
};

geodesic make_lightlike_geodesic(const tensor<valuef, 4>& position, const tensor<valuef, 3>& direction, const tetrad& tetrads) {
    geodesic g;
    g.position = position;
    g.velocity = tetrads.v[0] * -1 //Flipped time component, we're tracing backwards in time
               + tetrads.v[1] * direction[0]
               + tetrads.v[2] * direction[1]
               + tetrads.v[3] * direction[2];

    return g;
}

auto diff(auto&& func, const tensor<valuef, 4>& position, int direction) {
    auto p_up = position;
    auto p_lo = position;

    float h = 0.00001f;

    p_up[direction] += h;
    p_lo[direction] -= h;

    auto up = func(p_up);
    auto lo = func(p_lo);

    return (func(p_up) - func(p_lo)) * (1/(2 * h));
}

//get the christoffel symbols that we need for the geodesic equation
////https://www2.mpia-hd.mpg.de/homes/tmueller/pdfs/catalogue_2014-05-21.pdf you can check that this function returns the correct results, against 2.2.2a
tensor<valuef, 4, 4, 4> calculate_christoff2(const tensor<valuef, 4>& position, auto&& get_metric) {
    metric<valuef, 4, 4> metric = get_metric(position);
    inverse_metric<valuef, 4, 4> metric_inverse = metric.invert();

    tensor<valuef, 4, 4, 4> metric_diff; ///uses the index signature, diGjk

    for(int i=0; i < 4; i++) {
        auto differentiated = diff(get_metric, position, i);

        for(int j=0; j < 4; j++) {
            for(int k=0; k < 4; k++) {
                metric_diff[i, j, k] = differentiated[j, k];
            }
        }
    }

    tensor<valuef, 4, 4, 4> Gamma;

    for(int mu = 0; mu < 4; mu++)
    {
        for(int al = 0; al < 4; al++)
        {
            for(int be = 0; be < 4; be++)
            {
                valuef sum = 0;

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

//use the geodesic equation to get our acceleration
tensor<valuef, 4> calculate_acceleration_of(const tensor<valuef, 4>& X, const tensor<valuef, 4>& v, auto&& get_metric) {
    tensor<valuef, 4, 4, 4> christoff2 = calculate_christoff2(X, get_metric);

    tensor<valuef, 4> acceleration;

    for(int mu = 0; mu < 4; mu++) {
        valuef sum = 0;

        for(int al = 0; al < 4; al++) {
            for(int be = 0; be < 4; be++) {
                sum += -christoff2[mu, al, be] * v[al] * v[be];
            }
        }

        acceleration[mu] = sum;
    }

    return acceleration;
}

tensor<valuef, 4> calculate_schwarzschild_acceleration(const tensor<valuef, 4>& X, const tensor<valuef, 4>& v) {
    return calculate_acceleration_of(X, v, schwarzschild_metric);
}

tensor<valuef, 2> angle_to_tex(const tensor<valuef, 2>& angle)
{
    using namespace single_source;

    float pi = std::numbers::pi_v<float>;

    mut<valuef> thetaf = declare_mut_e("theta", fmod(angle[0], valuef(2 * pi)));
    mut<valuef> phif = declare_mut_e("phi", angle[1]);

    if_e(thetaf >= pi, [&]
    {
        as_ref(phif) = phif + pi;
        as_ref(thetaf) = thetaf - pi;
    });

    as_ref(phif) = fmod(phif, valuef(2 * pi));

    valuef sxf = phif / (2 * pi);
    valuef syf = thetaf / pi;

    sxf += 0.5f;

    return {sxf, syf};
}

//this integrates a geodesic, until it either escapes our small universe or hits the event horizon
std::pair<valuei, tensor<valuef, 4>> integrate(geodesic& g) {
    using namespace single_source;

    mut<valuei> result = declare_mut_e(valuei(2));

    tensor<mut<valuef>, 4> position = declare_mut_e(g.position);
    tensor<mut<valuef>, 4> velocity = declare_mut_e(g.velocity);

    float dt = 0.005f;
    float rs = 1;
    valuef start_time = g.position[0];

    pin(start_time);

    mut<valuei> idx = declare_mut_e("i", valuei(0));

    for_e(idx < 1024 * 1024, assign_b(idx, idx + 1), [&]
    {
        tensor<valuef, 4> cposition = declare_e(position);
        tensor<valuef, 4> cvelocity = declare_e(velocity);

        tensor<valuef, 4> acceleration = calculate_schwarzschild_acceleration(cposition, cvelocity);

        as_ref(velocity) = cvelocity + acceleration * dt;
        as_ref(position) = cposition + velocity.as<valuef>() * dt;

        valuef radius = position[1];

        if_e(radius > 10, [&] {
            //ray escaped
            as_ref(result) = valuei(0);
            break_e();
        });

        if_e(radius <= rs + 0.0001f || position[0] > start_time + 1000, [&] {
            //ray has very likely hit the event horizon
            as_ref(result) = valuei(1);
            break_e();
        });

        //we could do better than this by upgrading the tensor library
        if_e(!isfinite(position[0]) || !isfinite(position[1]) || !isfinite(position[2]) || !isfinite(position[3]), [&]
        {
            as_ref(result) = valuei(1);
            break_e();
        });
    });

    return {result, position.as<valuef>()};
}

tensor<valuef, 3> render_pixel(valuei x, valuei y, valuei screen_width, valuei screen_height, const read_only_image<2>& background, valuei background_width, valuei background_height)
{
    using namespace single_source;

    tensor<valuef, 3> ray_direction = get_ray_through_pixel(x, y, screen_width, screen_height, 90);

    float pi = std::numbers::pi_v<float>;

    tensor<valuef, 4> camera_position = {0, 5, pi/2, -pi/2};

    tetrad tetrads = calculate_schwarzschild_tetrad(camera_position);

    //so, the tetrad vectors give us a basis, that points in the direction t, r, theta, and phi, because schwarzschild is diagonal
    //we'd like the ray to point towards the black hole: this means we make +z point towards -r, +y point towards +theta, and +x point towards +phi
    tensor<valuef, 3> modified_ray = {-ray_direction[2], ray_direction[1], ray_direction[0]};

    geodesic my_geodesic = make_lightlike_geodesic(camera_position, modified_ray, tetrads);

    auto [result, position] = integrate(my_geodesic);

    valuef theta = position[2];
    valuef phi = position[3];

    tensor<valuef, 2> texture_coordinate = angle_to_tex({theta, phi});

    valuei tx = (texture_coordinate[0] * background_width.to<float>() + background_width.to<float>()).to<int>() % background_width;
    valuei ty = (texture_coordinate[1] * background_height.to<float>() + background_height.to<float>()).to<int>() % background_height;

    tensor<valuef, 4> col = background.read<float, 4>({tx, ty});

    tensor<mut<valuef>, 3> colour = declare_mut_e(col.xyz());

    if_e(result == 2 || result == 1, [&] {
        as_ref(colour) = (tensor<valuef, 3>){0,0,0};
    });

    return colour.as<valuef>();
}


void opencl_raytrace(execution_context& ectx, literal<int> screen_width, literal<int> screen_height, read_only_image<2> background, write_only_image<2> screen, literal<int> background_width, literal<int> background_height)
{
    using namespace single_source;

    valuei x = value_impl::get_global_id(0);
    valuei y = value_impl::get_global_id(1);

    //get_global_id() is not a const function, so assign it to an unnamed variable to avoid compilers repeatedly evaluating it
    pin(x);
    pin(y);

    if_e(y >= screen_height.get(), [&] {
        return_e();
    });

    if_e(x >= screen_width.get(), [&] {
        return_e();
    });

    tensor<valuef, 3> colour = render_pixel(x, y, screen_width.get(), screen_height.get(), background, background_width.get(), background_height.get());

    //the tensor library does actually support .x() etc, but I'm trying to keep the requirements for whatever you use yourself minimal
    tensor<valuef, 4> crgba = {colour[0], colour[1], colour[2], 1.f};

    screen.write(ectx, {x,y}, crgba);
}

#endif // SCHWARZSCHILD_SINGLE_SOURCE_HPP_INCLUDED

#ifndef SCHWARZSCHILD_SINGLE_SOURCE_HPP_INCLUDED
#define SCHWARZSCHILD_SINGLE_SOURCE_HPP_INCLUDED

#include "../common/vec/tensor.hpp"
#include "../common/vec/dual.hpp"
#include "single_source.hpp"

using valuef = value<float>;
using valuei = value<int>;

using v2f = tensor<valuef, 2>;
using v3f = tensor<valuef, 3>;
using v4f = tensor<valuef, 4>;
using v2i = tensor<valuei, 2>;
using v3i = tensor<valuei, 3>;
using m44f = metric<valuef, 4, 4>;

using mut_v4f = tensor<mut<valuef>, 4>;
using mut_v3f = tensor<mut<valuef>, 3>;

template<typename T>
using dual = dual_types::dual_v<T>;

struct tetrad
{
    std::array<v4f, 4> v;
};

//https://www2.mpia-hd.mpg.de/homes/tmueller/pdfs/catalogue_2014-05-21.pdf 2.2.6
tetrad calculate_schwarzschild_tetrad(const v4f& position) {
    valuef rs = 1;
    valuef r = position[1];
    valuef theta = position[2];

    v4f et = {1/sqrt(1 - rs/r), 0, 0, 0};
    v4f er = {0, sqrt(1 - rs/r), 0, 0};
    v4f etheta = {0, 0, 1/r, 0};
    v4f ephi = {0, 0, 0, 1/(r * sin(theta))};

    return {et, er, etheta, ephi};
}

v3f get_ray_through_pixel(v2i screen_position, v2i screen_size, float fov_degrees) {
    float fov_rad = (fov_degrees / 360.f) * 2 * std::numbers::pi_v<float>;
    valuef f_stop = (screen_size.x()/2).to<float>() / tan(fov_rad/2);

    v3f pixel_direction = {(screen_position.x() - screen_size.x()/2).to<float>(), (screen_position.y() - screen_size.y()/2).to<float>(), f_stop};
    //pixel_direction = rot_quat(pixel_direction, camera_quat); //if you have quaternions, or some rotation library, rotate your pixel direction here by your cameras rotation

    return pixel_direction.norm();
}

struct geodesic
{
    v4f position;
    v4f velocity;
};

geodesic make_lightlike_geodesic(const v4f& position, const v3f& direction, const tetrad& tetrads) {
    geodesic g;
    g.position = position;
    g.velocity = tetrads.v[0] * -1 //Flipped time component, we're tracing backwards in time
               + tetrads.v[1] * direction[0]
               + tetrads.v[2] * direction[1]
               + tetrads.v[3] * direction[2];

    return g;
}

auto diff(auto&& func, const v4f& position, int direction) {
    m44f metric = func(position);

    tensor<valuef, 4, 4> differentiated;

    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            dual<value_base> as_dual = replay_value_base<dual<value_base>>(metric[i, j], [&](const value_base& in)
            {
                value_base one;
                one.type = value_impl::op::VALUE;

                value_base zero;
                zero.type = value_impl::op::VALUE;

                std::visit([&]<typename T>(const T& conc)
                {
                    one.concrete = (T)1;
                    zero.concrete = (T)0;
                }, in.concrete);

                if(equivalent(in, position[direction]))
                    return dual<value_base>(in, one);
                else
                    return dual<value_base>(in, zero);
            });

            valuef out;
            out.set_from_base(as_dual.dual);

            differentiated[i, j] = out;
        }
    }

    return differentiated;
}

//get the christoffel symbols that we need for the geodesic equation
////https://www2.mpia-hd.mpg.de/homes/tmueller/pdfs/catalogue_2014-05-21.pdf you can check that this function returns the correct results, against 2.2.2a
tensor<valuef, 4, 4, 4> calculate_christoff2(const v4f& position, auto&& get_metric) {
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
v4f calculate_acceleration_of(const tensor<valuef, 4>& X, const tensor<valuef, 4>& v, auto&& get_metric) {
    tensor<valuef, 4, 4, 4> christoff2 = calculate_christoff2(X, get_metric);

    v4f acceleration;

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

v2f angle_to_tex(const v2f& angle)
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
std::pair<valuei, tensor<valuef, 4>> integrate(geodesic& g, auto&& get_metric) {
    using namespace single_source;

    mut<valuei> result = declare_mut_e(valuei(0));

    mut_v4f position = declare_mut_e(g.position);
    mut_v4f velocity = declare_mut_e(g.velocity);

    float dt = 0.005f;
    float rs = 1;
    valuef start_time = g.position[0];

    pin(start_time);

    mut<valuei> idx = declare_mut_e("i", valuei(0));

    for_e(idx < 1024 * 1024, assign_b(idx, idx + 1), [&]
    {
        v4f cposition = declare_e(position);
        v4f cvelocity = declare_e(velocity);

        v4f acceleration = calculate_acceleration_of(cposition, cvelocity, get_metric);

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
            //as_ref(result) = valuei(1);
            break_e();
        });
    });

    return {result, position.as<valuef>()};
}

v3f render_pixel(v2i screen_position, v2i screen_size, const read_only_image<2>& background, v2i background_size, const tetrad& tetrads, v4f start_position, auto&& get_metric)
{
    using namespace single_source;

    v3f ray_direction = get_ray_through_pixel(screen_position, screen_size, 90);

    //so, the tetrad vectors give us a basis, that points in the direction t, r, theta, and phi, because schwarzschild is diagonal
    //we'd like the ray to point towards the black hole: this means we make +z point towards -r, +y point towards +theta, and +x point towards +phi
    v3f modified_ray = {-ray_direction[2], ray_direction[1], ray_direction[0]};

    geodesic my_geodesic = make_lightlike_geodesic(start_position, modified_ray, tetrads);

    /*value_base se;
    se.type = value_impl::op::SIDE_EFFECT;
    se.abstract_value = "printf(\"%f\\n\"," + value_to_string(my_geodesic.velocity.z()) + ")";

    value_impl::get_context().add(se);*/

    auto [result, position] = integrate(my_geodesic, get_metric);

    valuef theta = position[2];
    valuef phi = position[3];

    v2f texture_coordinate = angle_to_tex({theta, phi});

    valuei tx = (texture_coordinate[0] * background_size.x().to<float>() + background_size.x().to<float>()).to<int>() % background_size.x();
    valuei ty = (texture_coordinate[1] * background_size.y().to<float>() + background_size.y().to<float>()).to<int>() % background_size.y();

    v4f col = background.read<float, 4>({tx, ty});

    mut_v3f colour = declare_mut_e(col.xyz());

    if_e(result == 2 || result == 1, [&] {
        as_ref(colour) = (tensor<valuef, 3>){0,1,0};
    });

    return colour.as<valuef>();
}

template<auto GetMetric>
void opencl_raytrace(execution_context& ectx, literal<int> screen_width, literal<int> screen_height,
                     read_only_image<2> background, write_only_image<2> screen,
                     literal<int> background_width, literal<int> background_height,
                     buffer<tensor<float, 4>> e0, buffer<tensor<float, 4>> e1, buffer<tensor<float, 4>> e2, buffer<tensor<float, 4>> e3,
                     buffer<tensor<float, 4>> position
                     )
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

    v2i screen_pos = {x, y};
    v2i screen_size = {screen_width.get(), screen_height.get()};
    v2i background_size = {background_width.get(), background_height.get()};

    tetrad tetrads = {e0[0], e1[0], e2[0], e3[0]};

    v3f colour = render_pixel(screen_pos, screen_size, background, background_size, tetrads, position[0], GetMetric);

    //the tensor library does actually support .x() etc, but I'm trying to keep the requirements for whatever you use yourself minimal
    v4f crgba = {colour[0], colour[1], colour[2], 1.f};

    screen.write(ectx, {x,y}, crgba);
}

valuef dot(v4f u, v4f v, m44f m) {
    v4f lowered = m.lower(u);

    return dot(lowered, v);
}

v4f gram_project(v4f u, v4f v, m44f m) {
    valuef top = dot_metric(u, v, m);
    valuef bottom = dot_metric(u, u, m);

    return (top / bottom) * u;
}

v4f normalise(v4f in, m44f m)
{
    valuef d = dot_metric(in, in, m);

    return in / sqrt(fabs(d));
}

tetrad gram_schmidt(v4f v0, v4f v1, v4f v2, v4f v3, m44f m)
{
    v4f u0 = v0;

    v4f u1 = v1;
    u1 = u1 - gram_project(u0, u1, m);

    v4f u2 = v2;
    u2 = u2 - gram_project(u0, u2, m);
    u2 = u2 - gram_project(u1, u2, m);

    v4f u3 = v3;
    u3 = u3 - gram_project(u0, u3, m);
    u3 = u3 - gram_project(u1, u3, m);
    u3 = u3 - gram_project(u2, u3, m);

    u0 = normalise(u0, m);
    u1 = normalise(u1, m);
    u2 = normalise(u2, m);
    u3 = normalise(u3, m);

    return {u0, u1, u2, u3};
}

template<auto GetMetric>
void build_initial_tetrads(execution_context& ectx, literal<tensor<float, 4>> camera_position,
                           buffer_mut<tensor<float, 4>> position_out,
                           buffer_mut<tensor<float, 4>> e0_out, buffer_mut<tensor<float, 4>> e1_out, buffer_mut<tensor<float, 4>> e2_out, buffer_mut<tensor<float, 4>> e3_out)
{
    using namespace single_source;

    as_ref(position_out[0]) = camera_position.get();

    v4f v0 = {1, 0, 0, 0};
    v4f v1 = {0, 1, 0, 0};
    v4f v2 = {0, 0, 1, 0};
    v4f v3 = {0, 0, 0, 1};

    m44f metric = GetMetric(camera_position.get());

    tetrad tetrads = gram_schmidt(v0, v1, v2, v3, metric);

    as_ref(e0_out[0]) = tetrads.v[0];
    as_ref(e1_out[0]) = tetrads.v[1];
    as_ref(e2_out[0]) = tetrads.v[2];
    as_ref(e3_out[0]) = tetrads.v[3];
}

#endif // SCHWARZSCHILD_SINGLE_SOURCE_HPP_INCLUDED

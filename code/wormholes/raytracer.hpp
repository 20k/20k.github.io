#ifndef SCHWARZSCHILD_SINGLE_SOURCE_HPP_INCLUDED
#define SCHWARZSCHILD_SINGLE_SOURCE_HPP_INCLUDED

#include "../common/vec/tensor.hpp"
#include "../common/vec/dual.hpp"
#include "single_source.hpp"
#include "redshift.hpp"
#include "metrics.hpp"

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

#define UNIVERSE_SIZE 100

struct inverse_tetrad
{
    std::array<v4f, 4> v_lo;

    v4f into_frame_of_reference(v4f in)
    {
        v4f out;

        for(int i=0; i < 4; i++)
            out[i] = dot(in, v_lo[i]);

        return out;
    }
};

struct tetrad
{
    std::array<v4f, 4> v;

    v4f into_coordinate_space(v4f in)
    {
        return v[0] * in.x() + v[1] * in.y() + v[2] * in.z() + v[3] * in.w();
    }

    inverse_tetrad invert()
    {
        inverse_tetrad invert;

        tensor<valuef, 4, 4> as_matrix;

        for(int i=0; i < 4; i++)
        {
            for(int j=0; j < 4; j++)
            {
                as_matrix[i, j] = v[i][j];
            }
        }

        tensor<valuef, 4, 4> inv = as_matrix.asymmetric_invert();

        invert.v_lo[0] = {inv[0, 0], inv[0, 1], inv[0, 2], inv[0, 3]};
        invert.v_lo[1] = {inv[1, 0], inv[1, 1], inv[1, 2], inv[1, 3]};
        invert.v_lo[2] = {inv[2, 0], inv[2, 1], inv[2, 2], inv[2, 3]};
        invert.v_lo[3] = {inv[3, 0], inv[3, 1], inv[3, 2], inv[3, 3]};

        return invert;
    }
};

inline
auto cartesian_to_spherical = []<typename T>(const tensor<T, 3>& cartesian)
{
    T r = cartesian.length();
    T theta = acos(cartesian[2] / r);
    T phi = atan2(cartesian[1], cartesian[0]);

    return tensor<T, 3>{r, theta, phi};
};

template<typename T, int N, typename Func>
inline
tensor<T, N> convert_velocity(Func&& f, const tensor<T, N>& pos, const tensor<T, N>& deriv)
{
    tensor<dual<T>, N> val;

    for(int i=0; i < N; i++)
        val[i] = dual(pos[i], deriv[i]);

    auto d = f(val);

    tensor<T, N> ret;

    for(int i=0; i < N; i++)
        ret[i] = d[i].dual;

    return ret;
}

inline
auto spherical_to_cartesian = []<typename T>(const tensor<T, 3>& spherical)
{
    T r = spherical[0];
    T theta = spherical[1];
    T phi = spherical[2];

    T x = r * sin(theta) * cos(phi);
    T y = r * sin(theta) * sin(phi);
    T z = r * cos(theta);

    return tensor<T, 3>{x, y, z};
};

v3f get_ray_through_pixel(v2i screen_position, v2i screen_size, float fov_degrees, v4f camera_quat) {
    float fov_rad = (fov_degrees / 360.f) * 2 * std::numbers::pi_v<float>;
    valuef f_stop = (screen_size.x()/2).to<float>() / tan(fov_rad/2);

    v3f pixel_direction = {(screen_position.x() - screen_size.x()/2).to<float>(), (screen_position.y() - screen_size.y()/2).to<float>(), f_stop};
    pixel_direction = rot_quat(pixel_direction, camera_quat); //if you have quaternions, or some rotation library, rotate your pixel direction here by your cameras rotation

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
    #define ANALYTIC_DERIVATIVES
    #ifdef ANALYTIC_DERIVATIVES
    m44f metric = func(position);

    tensor<valuef, 4, 4> differentiated;

    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            dual<value_base> as_dual = replay_value_base<dual<value_base>>(metric[i, j], [&](const value_base& in)
            {
                if(equivalent(in, position[direction]))
                    return dual<value_base>(in, in.make_constant_of_type(1.f));
                else
                    return dual<value_base>(in, in.make_constant_of_type(0.f));
            });

            differentiated[i, j].set_from_base(as_dual.dual);
        }
    }

    return differentiated;
    #else
    auto p_up = position;
    auto p_lo = position;

    float h = 0.00001f;

    p_up[direction] += h;
    p_lo[direction] -= h;

    auto up = func(p_up);
    auto lo = func(p_lo);

    return (func(p_up) - func(p_lo)) * (1/(2 * h));
    #endif // ANALYTIC
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

value<bool> should_terminate(v4f start, v4f position, v4f velocity)
{
    value<bool> is_broken = !isfinite(position[0]) || !isfinite(position[1]) || !isfinite(position[2]) || !isfinite(position[3]) ||
                            !isfinite(velocity[0]) || !isfinite(velocity[1]) || !isfinite(velocity[2]) || !isfinite(velocity[3]) ;

    return fabs(position[1]) > UNIVERSE_SIZE || position[0] > start[0] + 100 || fabs(velocity[0]) >= 10 || fabs(velocity[1]) >= 10 || is_broken;
}

valuef get_timelike_timestep(v4f position, v4f velocity)
{
    v4f avelocity = fabs(velocity);
    valuef divisor = max(max(avelocity.x(), avelocity.y()), max(avelocity.z(), avelocity.w()));

    valuef low_precision = 0.05f/divisor;
    valuef normal_precision = 0.012f/divisor;
    valuef high_precision = 0.005f/divisor;

    return ternary(fabs(position[1]) < 10, ternary(fabs(position[1]) < 3.f, high_precision, normal_precision), low_precision);
}

valuef get_lightlike_timestep(v4f position, v4f velocity, v4f acceleration)
{
    return get_timelike_timestep(position, velocity);
}

struct integration_result
{
    valuei type;
    v4f position;
    v4f velocity;

    v3f sample_colour;
    valuef sample_opacity;
};

std::tuple<v4f, v4f, v4f> verlet(v4f position, v4f velocity, v4f acceleration, valuef dt, auto&& get_metric)
{
    using namespace single_source;

    v4f next_position = position + velocity * dt + 0.5f * acceleration * dt * dt;

    v4f intermediate_next_velocity = velocity + acceleration * dt;

    pin(next_position);
    pin(intermediate_next_velocity);

    v4f next_acceleration = calculate_acceleration_of(next_position, intermediate_next_velocity, get_metric);

    pin(next_acceleration);

    v4f next_velocity = velocity + 0.5f * (acceleration + next_acceleration) * dt;

    pin(next_velocity);

    return {next_position, next_velocity, next_acceleration};
}

template<typename T>
T lookup(buffer<T> in, valuef coordinate, valuef minimum, valuef maximum)
{
    valuef rootf = clamp(floor(coordinate), minimum, maximum);
    valuef nextf = clamp(rootf + 1, minimum, maximum);

    valuei root = rootf.to<int>();
    valuei next = nextf.to<int>();

    T at_root = in[root];
    T at_next = in[next];

    return mix(at_root, at_next, clamp(coordinate, minimum, maximum) - rootf);
}

//this integrates a geodesic, until it either escapes our small universe or hits the event horizon
integration_result integrate(geodesic& g, v4f initial_observer, buffer<v3f> accretion_disk, buffer<v3f> bbody_table, buffer<valuef> temperature, auto&& get_metric) {
    using namespace single_source;

    integration_result found;

    mut<valuei> result = declare_mut_e(valuei(1));

    v4f start_cvel = declare_e(g.velocity);

    mut_v4f position = declare_mut_e(g.position);
    mut_v4f velocity = declare_mut_e(start_cvel);

    v4f start = g.position;

    mut<valuei> idx = declare_mut_e("i", valuei(0));

    mut_v3f colour_out = declare_mut_e((v3f){0,0,0});

    mut<valuef> opacity = declare_mut_e(valuef(0));

    float pi = std::numbers::pi_v<float>;

    //v4f start_cpos = declare_e(g.position);
    //mut_v4f acceleration = declare_mut_e(calculate_acceleration_of(start_cpos, start_cvel, get_metric));

    for_e(idx < 1024 * 1000, assign_b(idx, idx + 1), [&]
    {
        /*v4f cposition = declare_e(position);
        v4f cvelocity = declare_e(velocity);
        v4f cacceleration = declare_e(acceleration);

        valuef dt = get_timestep2(cposition, cvelocity, cacceleration);

        auto [next_position, next_velocity, next_acceleration] = verlet(cposition, cvelocity, cacceleration, dt, get_metric);

        as_ref(position) = next_position;
        as_ref(velocity) = next_velocity;
        as_ref(acceleration) = next_acceleration;*/

        v4f cposition = declare_e(position);
        v4f cvelocity = declare_e(velocity);

        v4f acceleration = calculate_acceleration_of(cposition, cvelocity, get_metric);

        valuef dt = get_lightlike_timestep(cposition, cvelocity, acceleration);

        pin(acceleration);

        as_ref(position) = cposition + cvelocity * dt;
        as_ref(velocity) = cvelocity + acceleration * dt;

        valuef radius = position[1];

        if_e(fabs(radius) > UNIVERSE_SIZE, [&] {
            //ray escaped
            as_ref(result) = valuei(0);
            break_e();
        });

        #ifdef HAS_ACCRETION_DISK
        valuef period_start = floor(position.z() / pi) * pi;

        valuef in_start = cposition.z() - period_start;
        valuef in_end = position.z() - period_start;

        valuef min_start = min(in_start, in_end);
        valuef max_start = max(in_start, in_end);

        if_e(pi/2 >= min_start && pi/2 <= max_start, [&]
        {
            valuef radial = position[1];

            valuef M = BH_MASS;
            valuef a = BH_SPIN;

            valuef w = pow(M, 1.f/2.f) / (pow(radial, 3.f/2.f) + a * pow(M, 1.f/2.f));

            v4f observer = {radial, 0, 0, w * radial};

            valuef ds = dot_metric(observer, observer, get_metric(cposition));

            ///valid circular geodesic
            if_e(ds < 0 && radial > 0, [&]
            {
                int buffer_size = 2048;
                valuef outer_boundary = 2 * BH_MASS * 50;

                valuef buffer_coordinate = (fabs(radial) / outer_boundary) * buffer_size;
                v3f disk = lookup(accretion_disk, buffer_coordinate, 0.f, valuef(buffer_size - 1));

                disk = disk * clamp(1 - declare_e(opacity), 0.f, 1.f);
                as_ref(opacity) = declare_e(opacity) + energy_of(disk) * 50;

                ///change the parameterisation to proper time
                observer = observer / sqrt(fabs(ds));

                pin(observer);

                #define ACCURATE_REDSHIFT
                #ifdef ACCURATE_REDSHIFT
                valuef temperature_in = lookup(temperature, buffer_coordinate, 0.f, valuef(buffer_size - 1));

                ///temperature == 0 is impossible in our disk, so indicates an invalid area
                if_e(temperature_in >= 1, [&]
                {
                    valuef zp1 = get_zp1(g.position, g.velocity, initial_observer, cposition, cvelocity, observer, get_metric);

                    ///https://www.jb.man.ac.uk/distance/frontiers/cmb/node7.htm
                    valuef shifted_temperature = temperature_in / zp1;

                    valuef old_brightness = energy_of(disk);

                    ///https://arxiv.org/pdf/gr-qc/9505010 12
                    valuef new_brightness = old_brightness / pow(zp1, 4.f);

                    v3f final_colour = lookup(bbody_table, shifted_temperature, 1.f, 100000 - 1.f) * new_brightness;

                    as_ref(colour_out) = declare_e(colour_out) + final_colour;
                });
                #endif

                #ifdef ILLUSTRATIVE_REDSHIFT
                valuef zp1 = get_zp1(g.position, g.velocity, initial_observer, cposition, cvelocity, observer, get_metric);

                v3f shifted = do_redshift(disk, zp1);

                as_ref(colour_out) = declare_e(colour_out) + shifted;
                #endif

                #ifdef RAW_DISK
                as_ref(colour_out) = declare_e(colour_out) + disk;
                #endif // RAW_DISK

                if_e(opacity >= 1, [] {
                    break_e();
                });
            });
        });
        #endif

        if_e(should_terminate(start, as_constant(position), as_constant(velocity)), [&] {
            break_e();
        });
    });

    found.type = result;
    found.position = declare_e(position);
    found.velocity = declare_e(velocity);
    found.sample_colour = declare_e(colour_out);
    found.sample_opacity = clamp(declare_e(opacity), valuef(0.0f), valuef(1.f));

    return found;
}

v3f render_pixel(v2i screen_position, v2i screen_size,
                 const read_only_image<2>& background, const read_only_image<2>& background2, buffer<v3f> accretion_disk, buffer<v3f> bbody_table, buffer<valuef> temperature,
                 v2i background_size, const tetrad& tetrads, v4f start_position, v4f camera_quat, auto&& get_metric)
{
    using namespace single_source;

    v3f ray_direction = get_ray_through_pixel(screen_position, screen_size, 90, camera_quat);

    //so, the tetrad vectors give us a basis, that points in the direction t, r, theta, and phi, because schwarzschild is diagonal
    //we'd like the ray to point towards the black hole: this means we make +z point towards -r, +y point towards +theta, and +x point towards +phi
    geodesic my_geodesic = make_lightlike_geodesic(start_position, ray_direction, tetrads);

    /*value_base se;
    se.type = value_impl::op::SIDE_EFFECT;
    se.abstract_value = "printf(\"%f\\n\"," + value_to_string(my_geodesic.velocity.z()) + ")";

    value_impl::get_context().add(se);*/

    integration_result result = integrate(my_geodesic, tetrads.v[0], accretion_disk, bbody_table, temperature, get_metric);

    valuef theta = result.position[2];
    valuef phi = result.position[3];

    v2f texture_coordinate = angle_to_tex({theta, phi});

    valuei tx = (texture_coordinate[0] * background_size.x().to<float>() + background_size.x().to<float>()).to<int>() % background_size.x();
    valuei ty = (texture_coordinate[1] * background_size.y().to<float>() + background_size.y().to<float>()).to<int>() % background_size.y();

    v4f col = ternary(result.position[1] >= 0, background.read<float, 4>({tx, ty}), background2.read<float,4>({tx, ty}));

    //#define DO_REDSHIFT
    #ifdef DO_REDSHIFT
    {
        valuef zp1 = get_zp1(my_geodesic.position, my_geodesic.velocity, tetrads.v[0], result.position, result.velocity, (v4f){1, 0, 0, 0}, get_metric);

        v3f col3 = do_redshift(col.xyz(), zp1);

        col.x() = col3.x();
        col.y() = col3.y();
        col.z() = col3.z();
    }
    #endif

    mut_v3f colour = declare_mut_e(col.xyz());

    if_e(result.type == 1, [&] {
        as_ref(colour) = (tensor<valuef, 3>){0,0,0};
    });

    return colour.as<valuef>() * (1-result.sample_opacity) + result.sample_colour;
}

template<auto GetMetric>
void opencl_raytrace(execution_context& ectx, literal<valuei> screen_width, literal<valuei> screen_height,
                     read_only_image<2> background, read_only_image<2> background2, write_only_image<2> screen,
                     buffer<v3f> accretion_disk,
                     literal<valuei> background_width, literal<valuei> background_height,
                     buffer<v4f> e0, buffer<v4f> e1, buffer<v4f> e2, buffer<v4f> e3,
                     buffer<v4f> position, literal<v4f> camera_quat,
                     buffer<v3f> bbody_table, buffer<valuef> temperature)
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

    v3f colour = render_pixel(screen_pos, screen_size, background, background2, accretion_disk, bbody_table, temperature, background_size, tetrads, position[0], camera_quat.get(), GetMetric);

    colour = linear_to_srgb_gpu(colour);

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
    using namespace single_source;

    v4f u0 = v0;

    v4f u1 = v1;
    u1 = u1 - gram_project(u0, u1, m);

    pin(u1);

    v4f u2 = v2;
    u2 = u2 - gram_project(u0, u2, m);
    u2 = u2 - gram_project(u1, u2, m);

    pin(u2);

    v4f u3 = v3;
    u3 = u3 - gram_project(u0, u3, m);
    u3 = u3 - gram_project(u1, u3, m);
    u3 = u3 - gram_project(u2, u3, m);

    pin(u3);

    u0 = normalise(u0, m);
    u1 = normalise(u1, m);
    u2 = normalise(u2, m);
    u3 = normalise(u3, m);

    pin(u0);
    pin(u1);
    pin(u2);
    pin(u3);

    return {u0, u1, u2, u3};
}

template<typename T>
void swap(const T& v1, const T& v2)
{
    auto intermediate = single_source::declare_e(v1);
    as_ref(v1) = v2;
    as_ref(v2) = intermediate;
}

///specifically: cartesian minkowski
m44f get_local_minkowski(const tetrad& tetrads, const m44f& met)
{
    m44f minkowski;

    tensor<valuef, 4, 4> m;

    for(int i=0; i < 4; i++)
    {
        m[0, i] = tetrads.v[0][i];
        m[1, i] = tetrads.v[1][i];
        m[2, i] = tetrads.v[2][i];
        m[3, i] = tetrads.v[3][i];
    }

    for(int a=0; a < 4; a++)
    {
        for(int b=0; b < 4; b++)
        {
            valuef sum = 0;

            for(int mu=0; mu < 4; mu++)
            {
                for(int v=0; v < 4; v++)
                {
                    sum += met[mu, v] * m[a, mu] * m[b, v];
                }
            }

            minkowski[a, b] = sum;
        }
    }

    return minkowski;
}

valuei calculate_which_coordinate_is_timelike(const tetrad& tetrads, const m44f& met)
{
    m44f minkowski = get_local_minkowski(tetrads, met);

    using namespace single_source;

    mut<valuei> lowest_index = declare_mut_e(valuei(0));
    mut<valuef> lowest_value = declare_mut_e(valuef(0));

    for(int i=0; i < 4; i++)
    {
        if_e(minkowski[i, i] < lowest_value, [&] {
            as_ref(lowest_index) = valuei(i);
            as_ref(lowest_value) = minkowski[i, i];
        });
    }

    return lowest_index;
}

v4f get_timelike_vector(v3f velocity, const tetrad& tetrads)
{
    v4f coordinate_time = {1, velocity.x(), velocity.y(), velocity.z()};

    valuef lorentz_factor = 1/sqrt(1 - (velocity.x() * velocity.x() + velocity.y() * velocity.y() + velocity.z() * velocity.z()));

    v4f proper_time = lorentz_factor * coordinate_time;

    ///put into curved spacetime
    return proper_time.x() * tetrads.v[0] + proper_time.y() * tetrads.v[1] + proper_time.z() * tetrads.v[2] + proper_time.w() * tetrads.v[3];
}

tetrad boost_tetrad(v3f velocity, const tetrad& tetrads, const metric<valuef, 4, 4>& m)
{
    using namespace single_source;

    v4f u = tetrads.v[0];
    v4f v = get_timelike_vector(velocity, tetrads);

    v4f u_l = m.lower(u);
    v4f v_l = m.lower(v);

    valuef Y = -dot(v_l, u);

    ///https://arxiv.org/pdf/2404.05744 18
    tensor<valuef, 4, 4> B;

    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            valuef kronecker = (i == j) ? 1 : 0;

            B[i, j] = kronecker + ((v[i] + u[i]) * (v_l[j] + u_l[j]) / (1 + Y)) - 2 * v[i] * u_l[j];
        }
    }

    tetrad next;

    for(int a=0; a < 4; a++)
    {
        for(int i=0; i < 4; i++)
        {
            valuef sum = 0;

            for(int j=0; j < 4; j++)
            {
                sum += B[i, j] * tetrads.v[a][j];
            }

            next.v[a][i] = sum;
        }
    }

    return next;
}

v3f project(v3f u, v3f v)
{
    return (dot(u, v) / dot(u, u)) * u;
}

std::array<v3f, 3> orthonormalise(v3f i1, v3f i2, v3f i3)
{
    v3f u1 = i1;
    v3f u2 = i2;
    v3f u3 = i3;

    u2 = u2 - project(u1, u2);

    u3 = u3 - project(u1, u3);
    u3 = u3 - project(u2, u3);

    u1 = u1.norm();
    u2 = u2.norm();
    u3 = u3.norm();

    return {u1, u2, u3};
};

template<auto GetMetric, auto GenericToSpherical, auto SphericalToGeneric>
void build_initial_tetrads(execution_context& ectx, literal<v4f> position,
                           literal<v3f> local_velocity,
                           buffer_mut<v4f> position_out,
                           buffer_mut<v4f> e0_out, buffer_mut<v4f> e1_out, buffer_mut<v4f> e2_out, buffer_mut<v4f> e3_out)
{
    using namespace single_source;

    as_ref(position_out[0]) = position.get();

    v4f v0 = {1, 0, 0, 0};
    v4f v1 = {0, 1, 0, 0};
    v4f v2 = {0, 0, 1, 0};
    v4f v3 = {0, 0, 0, 1};

    m44f metric = GetMetric(position.get());

    //these are actually the column vectors of the metric tensor
    v4f lv0 = metric.lower(v0);
    v4f lv1 = metric.lower(v1);
    v4f lv2 = metric.lower(v2);
    v4f lv3 = metric.lower(v3);

    array_mut<v4f> as_array = declare_mut_array_e<v4f>(4, {v0, v1, v2, v3});
    //we're in theory doing v_mu v^mu, but because only one component of v0 is nonzero, and the lower components are actually
    //the column vectors of the metric tensor, dot(v0, lv0) is actually metric[0,0], dot(v1, lv1) is metric[1,1] etc
    //this method therefore fails if the metric has no nonzero diagonal components
    array_mut<valuef> lengths = declare_mut_array_e<valuef>(4, {dot(v0, lv0), dot(v1, lv1), dot(v2, lv2), dot(v3, lv3)});

    mut<valuei> first_nonzero = declare_mut_e(valuei(0));

    for_e(first_nonzero < 4, assign_b(first_nonzero, first_nonzero+1), [&] {
        auto approx_eq = [](const valuef& v1, const valuef& v2) {
            valuef bound = 0.0001f;

            return v1 >= v2 - bound && v1 < v2 + bound;
        };

        if_e(!approx_eq(lengths[first_nonzero], valuef(0.f)), [&] {
             break_e();
        });
    });

    swap(as_array[0], as_array[first_nonzero]);

    v4f iv0 = declare_e(as_array[0]);
    v4f iv1 = declare_e(as_array[1]);
    v4f iv2 = declare_e(as_array[2]);
    v4f iv3 = declare_e(as_array[3]);

    pin(metric);

    tetrad tetrads = gram_schmidt(iv0, iv1, iv2, iv3, metric);

    array_mut<v4f> tetrad_array = declare_mut_array_e<v4f>(4, {});

    as_ref(tetrad_array[0]) = tetrads.v[0];
    as_ref(tetrad_array[1]) = tetrads.v[1];
    as_ref(tetrad_array[2]) = tetrads.v[2];
    as_ref(tetrad_array[3]) = tetrads.v[3];

    swap(tetrad_array[0], tetrad_array[first_nonzero]);

    valuei timelike_coordinate = calculate_which_coordinate_is_timelike(tetrads, metric);

    swap(tetrad_array[0], tetrad_array[timelike_coordinate]);

    tetrad tet;
    tet.v = {declare_e(tetrad_array[0]), declare_e(tetrad_array[1]), declare_e(tetrad_array[2]), declare_e(tetrad_array[3])};

    bool should_orient = true;

    if(should_orient)
    {
        v4f spher = GenericToSpherical(position.get());
        v3f cart = spherical_to_cartesian(spher.yzw());

        v3f bx = (v3f){1, 0, 0};
        v3f by = (v3f){0, 1, 0};
        v3f bz = (v3f){0, 0, 1};

        v3f sx = convert_velocity(cartesian_to_spherical, cart, bx);
        v3f sy = convert_velocity(cartesian_to_spherical, cart, by);
        v3f sz = convert_velocity(cartesian_to_spherical, cart, bz);

        sx.x() = ternary(spher.y() < 0, -sx.x(), sx.x());
        sy.x() = ternary(spher.y() < 0, -sy.x(), sy.x());
        sz.x() = ternary(spher.y() < 0, -sz.x(), sz.x());

        v4f dx = convert_velocity(SphericalToGeneric, spher, (v4f){0.f, sx.x(), sx.y(), sx.z()});
        v4f dy = convert_velocity(SphericalToGeneric, spher, (v4f){0.f, sy.x(), sy.y(), sy.z()});
        v4f dz = convert_velocity(SphericalToGeneric, spher, (v4f){0.f, sz.x(), sz.y(), sz.z()});

        inverse_tetrad itet = tet.invert();

        pin(dx);
        pin(dy);
        pin(dz);

        v4f lx = itet.into_frame_of_reference(dx);
        v4f ly = itet.into_frame_of_reference(dy);
        v4f lz = itet.into_frame_of_reference(dz);

        pin(lx);
        pin(ly);
        pin(lz);

        std::array<v3f, 3> ortho = orthonormalise(ly.yzw(), lx.yzw(), lz.yzw());

        v4f x_basis = {0, ortho[1].x(), ortho[1].y(), ortho[1].z()};
        v4f y_basis = {0, ortho[0].x(), ortho[0].y(), ortho[0].z()};
        v4f z_basis = {0, ortho[2].x(), ortho[2].y(), ortho[2].z()};

        pin(x_basis);
        pin(y_basis);
        pin(z_basis);

        v4f x_out = tet.into_coordinate_space(x_basis);
        v4f y_out = tet.into_coordinate_space(y_basis);
        v4f z_out = tet.into_coordinate_space(z_basis);

        pin(x_out);
        pin(y_out);
        pin(z_out);

        tet.v[1] = x_out;
        tet.v[2] = y_out;
        tet.v[3] = z_out;
    }

    tetrad boosted = boost_tetrad(local_velocity.get(), tet, metric);

    as_ref(e0_out[0]) = boosted.v[0];
    as_ref(e1_out[0]) = boosted.v[1];
    as_ref(e2_out[0]) = boosted.v[2];
    as_ref(e3_out[0]) = boosted.v[3];
}

template<auto GetMetric>
void trace_geodesic(execution_context& ectx,
                    buffer<v4f> start_position, buffer<v4f> start_velocity,
                    buffer_mut<v4f> positions_out, buffer_mut<v4f> velocity_out,
                    buffer_mut<valuei> written_steps, literal<valuei> max_steps)
{
    using namespace single_source;

    m44f metric = GetMetric(start_position[0]);

    mut<valuei> result = declare_mut_e(valuei(0));

    mut_v4f position = declare_mut_e(start_position[0]);
    mut_v4f velocity = declare_mut_e(start_velocity[0]);

    v4f start_cpos = declare_e(start_position[0]);
    v4f start_cvel = declare_e(start_velocity[0]);

    mut_v4f acceleration = declare_mut_e(calculate_acceleration_of(start_cpos, start_cvel, GetMetric));

    v4f start = declare_e(start_position[0]);

    mut<valuei> idx = declare_mut_e("i", valuei(0));

    for_e(idx < 1024 * 1024 && idx < max_steps.get(), assign_b(idx, idx + 1), [&]
    {
        as_ref(positions_out[idx]) = position;
        as_ref(velocity_out[idx]) = velocity;

        v4f cposition = declare_e(position);
        v4f cvelocity = declare_e(velocity);
        v4f cacceleration = declare_e(acceleration);

        /*value_base se;
        se.type = value_impl::op::SIDE_EFFECT;
        se.abstract_value = "printf(\"%f\\n\"," + value_to_string(cposition.y()) + ")";

        value_impl::get_context().add(se);*/

        valuef dt = get_timelike_timestep(cposition, cvelocity);

        auto [next_position, next_velocity, next_acceleration] = verlet(cposition, cvelocity, cacceleration, valuef(dt), GetMetric);

        as_ref(position) = next_position;
        as_ref(velocity) = next_velocity;
        as_ref(acceleration) = next_acceleration;

        if_e(should_terminate(start, as_constant(position), as_constant(velocity)), [&] {
            break_e();
        });
    });

    as_ref(written_steps[0]) = idx.as_constant() + 1;
}

v4f parallel_transport_get_change(v4f tangent_vector, v4f geodesic_velocity, const tensor<valuef, 4, 4, 4>& christoff2)
{
    v4f dAdt = {};

    for(int a=0; a < 4; a++)
    {
        valuef sum = 0;

        for(int b=0; b < 4; b++)
        {
            for(int s=0; s < 4; s++)
            {
                sum += christoff2[a,b,s] * tangent_vector[b] * geodesic_velocity[s];
            }
        }

        dAdt[a] = -sum;
    }

    return dAdt;
}

v4f transport2(v4f what, v4f position, v4f next_position, v4f velocity, v4f next_velocity, valuef ds, auto&& get_metric)
{
    using namespace single_source;

    tensor<valuef, 4, 4, 4> christoff2 = calculate_christoff2(position, get_metric);

    pin(christoff2);

    v4f f_x = parallel_transport_get_change(what, velocity, christoff2);

    v4f intermediate_next = what + f_x * ds;

    tensor<valuef, 4, 4, 4> nchristoff2 = calculate_christoff2(next_position, get_metric);

    pin(nchristoff2);

    return what + 0.5f * ds * (f_x + parallel_transport_get_change(intermediate_next, next_velocity, nchristoff2));
}

v4f transport1(v4f what, v4f position, v4f velocity, valuef ds, auto&& get_metric)
{
    using namespace single_source;

    tensor<valuef, 4, 4, 4> christoff2 = calculate_christoff2(position, get_metric);

    pin(christoff2);

    v4f f_x = parallel_transport_get_change(what, velocity, christoff2);

    return what + f_x * ds;
}

//note: we already know the value of e0, as its the geodesic velocity
template<auto GetMetric>
void parallel_transport_tetrads(execution_context& ectx,
                                buffer<v4f> e0, buffer<v4f> e1, buffer<v4f> e2, buffer<v4f> e3,
                                buffer<v4f> positions, buffer<v4f> velocities, buffer<valuei> counts,
                                buffer_mut<v4f> e0_out, buffer_mut<v4f> e1_out, buffer_mut<v4f> e2_out, buffer_mut<v4f> e3_out)
{
    using namespace single_source;

    valuei count = declare_e(counts[0]);
    mut<valuei> i = declare_mut_e(valuei(0));

    mut_v4f e0_current = declare_mut_e(e0[0]);
    mut_v4f e1_current = declare_mut_e(e1[0]);
    mut_v4f e2_current = declare_mut_e(e2[0]);
    mut_v4f e3_current = declare_mut_e(e3[0]);

    for_e(i < count - 1, assign_b(i, i+1), [&] {
        as_ref(e0_out[i]) = e0_current;
        as_ref(e1_out[i]) = e1_current;
        as_ref(e2_out[i]) = e2_current;
        as_ref(e3_out[i]) = e3_current;

        v4f current_position = declare_e(positions[i]);
        v4f current_velocity = declare_e(velocities[i]);

        v4f next_position = declare_e(positions[i+1]);
        v4f next_velocity = declare_e(velocities[i+1]);

        v4f e0_cst = declare_e(e0_current);
        v4f e1_cst = declare_e(e1_current);
        v4f e2_cst = declare_e(e2_current);
        v4f e3_cst = declare_e(e3_current);

        valuef dt = get_timelike_timestep(current_position, current_velocity);

        v4f e0_next = transport2(e0_cst, current_position, next_position, current_velocity, next_velocity, valuef(dt), GetMetric);
        v4f e1_next = transport2(e1_cst, current_position, next_position, current_velocity, next_velocity, valuef(dt), GetMetric);
        v4f e2_next = transport2(e2_cst, current_position, next_position, current_velocity, next_velocity, valuef(dt), GetMetric);
        v4f e3_next = transport2(e3_cst, current_position, next_position, current_velocity, next_velocity, valuef(dt), GetMetric);

        as_ref(e0_current) = e0_next;
        as_ref(e1_current) = e1_next;
        as_ref(e2_current) = e2_next;
        as_ref(e3_current) = e3_next;
    });

    if_e(count > 0, [&]{
        as_ref(e0_out[i]) = e0_current;
        as_ref(e1_out[i]) = e1_current;
        as_ref(e2_out[i]) = e2_current;
        as_ref(e3_out[i]) = e3_current;
    });
}

void interpolate(execution_context& ectx, buffer<v4f> positions, buffer<v4f> velocities,
                 buffer<v4f> e0s, buffer<v4f> e1s, buffer<v4f> e2s, buffer<v4f> e3s, buffer<valuei> counts,
                 literal<valuef> desired_proper_time,
                 buffer_mut<v4f> position_out, buffer_mut<v4f> e0_out, buffer_mut<v4f> e1_out, buffer_mut<v4f> e2_out, buffer_mut<v4f> e3_out)
{
    using namespace single_source;

    valuei size = declare_e(counts[0]);

    //somethings gone horribly wrong somewhere
    if_e(size == 0, [&] {
        return_e();
    });

    mut<valuef> elapsed_time = declare_mut_e(valuef(0));

    //fallback if we pick proper time < earliest time
    if_e(desired_proper_time.get() <= 0, [&]{
        as_ref(position_out[0]) = positions[0];
        as_ref(e0_out[0]) = e0s[0];
        as_ref(e1_out[0]) = e1s[0];
        as_ref(e2_out[0]) = e2s[0];
        as_ref(e3_out[0]) = e3s[0];
        return_e();
    });

    mut<valuei> i = declare_mut_e(valuei(0));

    for_e(i < (size - 1), assign_b(i, i+1), [&] {
        valuef dt = get_timelike_timestep(positions[i], velocities[i]);

        if_e(desired_proper_time.get() >= elapsed_time && desired_proper_time.get() < elapsed_time + dt, [&]{
            valuef frac = (desired_proper_time.get() - elapsed_time) / dt;

            as_ref(position_out[0]) = mix(positions[i], positions[i+1], frac);
            as_ref(e0_out[0]) = mix(e0s[i], e0s[i+1], frac);
            as_ref(e1_out[0]) = mix(e1s[i], e1s[i+1], frac);
            as_ref(e2_out[0]) = mix(e2s[i], e2s[i+1], frac);
            as_ref(e3_out[0]) = mix(e3s[i], e3s[i+1], frac);

            return_e();
        });

        as_ref(elapsed_time) = elapsed_time + dt;
    });

    //fallback for if we pick proper time > latest proper time
    as_ref(position_out[0]) = positions[size-1];
    as_ref(e0_out[0]) = e0s[size-1];
    as_ref(e1_out[0]) = e1s[size-1];
    as_ref(e2_out[0]) = e2s[size-1];
    as_ref(e3_out[0]) = e3s[size-1];
}

#endif // SCHWARZSCHILD_SINGLE_SOURCE_HPP_INCLUDED

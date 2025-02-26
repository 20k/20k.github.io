#include "raytrace.hpp"
#include "tensor_algebra.hpp"
#include "bssn.hpp"
#include "formalisms.hpp"
#include "interpolation.hpp"
#include "plugin.hpp"

template<typename T, int N>
struct verlet_context
{
    tensor<mut<T>, N> position;
    mut<T> ds_m;
    tensor<mut<T>, N> last_v_half;
    tensor<mut<T>, N> velocity;

    template<typename dX, typename dV, typename dS, typename State>
    void start(const tensor<T, N>& X_in, const tensor<T, N>& V_in, dX&& get_dX, dV&& get_dV, dS&& get_dS, State&& get_state)
    {
        using namespace single_source;

        position = declare_mut_e(X_in);
        velocity = declare_mut_e(V_in);

        auto x_0 = declare_e(X_in);
        auto v_0 = declare_e(V_in);

        auto st = get_state(x_0);

        auto acceleration = get_dV(x_0, v_0, st);
        pin(acceleration);

        auto ds = get_dS(x_0, v_0, acceleration, st);
        pin(ds);

        auto v_half = v_0 + 0.5f * ds * get_dV(x_0, v_0, st);

        //#define IMPLICIT_V
        #ifdef IMPLICIT_V
        v_half = v_0 + 0.5f * ds * get_dV(x_0, v_half, st);
        #endif

        auto x_full_approx = x_0 + ds * get_dX(x_0, v_0, st);

        auto st_full_approx = get_state(x_full_approx);

        auto x_full = x_0 + 0.5f * ds * (get_dX(x_0, v_half, st) + get_dX(x_full_approx, v_half, st_full_approx));

        //#define IMPLICIT_X
        #ifdef IMPLICIT_X
        auto st_full_implicit = get_state(x_full);
        x_full = cposition + 0.5f * ds * (get_dX(cposition, v_half, st) + get_dX(x_full, v_half, st_full_implicit));
        #endif

        //auto st_full = get_state(x_full);
        //auto v_full = v_half + 0.5f * ds * get_dV(x_full, v_half, st_full);

        last_v_half = declare_mut_e(v_half);
        ds_m = declare_mut_e(ds);
        as_ref(position) = x_full;
    }

    template<typename dX, typename dV, typename dS, typename State>
    auto next(dX&& get_dX, dV&& get_dV, dS&& get_dS, State&& get_state, auto&& enforce_velocity_constraint)
    {
        using namespace single_source;

        auto x_n = declare_e(position);
        auto v_nhalf = declare_e(last_v_half);
        auto ds_n1 = declare_e(ds_m);

        auto st = get_state(x_n);

        auto v_n = v_nhalf + 0.5f * ds_n1 * get_dV(x_n, v_nhalf, st);

        v_n = enforce_velocity_constraint(v_n, st);

        pin(v_n);

        auto acceleration = get_dV(x_n, v_n, st);
        pin(acceleration);

        auto ds = get_dS(x_n, v_n, acceleration, st);
        pin(ds);

        auto v_half = v_n + 0.5f * ds * get_dV(x_n, v_n, st);

        auto x_full_approx = x_n + ds * get_dX(x_n, v_n, st);
        auto st_full_approx = get_state(x_full_approx);

        auto x_full = x_n + 0.5f * ds * (get_dX(x_n, v_half, st) + get_dX(x_full_approx, v_half, st_full_approx));

        as_ref(position) = x_full;
        as_ref(last_v_half) = v_half;
        as_ref(ds_m) = ds;
        //I only update this for rasterisation reasons
        as_ref(velocity) = v_half;

        return get_dX(x_n, v_half, st);
    }
};

template<typename T, int N>
struct euler_context
{
    tensor<mut<T>, N> position;
    tensor<mut<T>, N> velocity;

    template<typename dX, typename dV, typename dS, typename State>
    void start(const tensor<T, N>& position_in, const tensor<T, N>& velocity_in, dX&& get_dX, dV&& get_dV, dS&& get_dS, State&& get_state)
    {
        using namespace single_source;

        position = declare_mut_e(position_in);
        velocity = declare_mut_e(velocity_in);
    }

    template<typename dX, typename dV, typename dS, typename State>
    auto next(dX&& get_dX, dV&& get_dV, dS&& get_dS, State&& get_state, auto&& velocity_postprocess)
    {
        using namespace single_source;

        auto cposition = declare_e(position);
        auto cvelocity = declare_e(velocity);

        auto st = get_state(cposition);

        auto accel = declare_e(get_dV(cposition, cvelocity, st));
        auto dPosition = declare_e(get_dX(cposition, cvelocity, st));

        auto ds = declare_e(get_dS(cposition, cvelocity, accel, st));

        cvelocity = velocity_postprocess(cvelocity, st);

        as_ref(position) = cposition + dPosition * ds;
        as_ref(velocity) = cvelocity + accel * ds;

        return dPosition;
    }
};

v2f angle_to_tex(const v2f& angle)
{
    using namespace single_source;

    float pi = std::numbers::pi_v<float>;

    mut<valuef> thetaf = declare_mut_e(fmod(angle[0], valuef(2 * pi)));
    mut<valuef> phif = declare_mut_e(angle[1]);

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

//calculate Y of XYZ
valuef energy_of(v3f v)
{
    return v.x()*0.2125f + v.y()*0.7154f + v.z()*0.0721f;
}

v3f redshift(v3f v, valuef z)
{
    using namespace single_source;

    {
        valuef iemit = energy_of(v);
        valuef iobs = iemit / pow(z+1, 4.f);

        v = (iobs / iemit) * v;

        pin(v);
    }

    valuef radiant_energy = energy_of(v);

    v3f red = {1/0.2125f, 0.f, 0.f};
    v3f green = {0, 1/0.7154, 0.f};
    v3f blue = {0.f, 0.f, 1/0.0721};

    mut_v3f result = declare_mut_e((v3f){0,0,0});

    if_e(z >= 0, [&]{
        as_ref(result) = mix(v, radiant_energy * red, tanh(z));
    });

    if_e(z < 0, [&]{
        valuef iv1pz = (1/(1 + z)) - 1;

        valuef interpolating_fraction = tanh(iv1pz);

        v3f col = mix(v, radiant_energy * blue, interpolating_fraction);

        //calculate spilling into white
        {
            valuef final_energy = energy_of(clamp(col, 0.f, 1.f));
            valuef real_energy = energy_of(col);

            valuef remaining_energy = real_energy - final_energy;

            col.x() += remaining_energy * red.x();
            col.y() += remaining_energy * green.y();
        }

        as_ref(result) = col;
    });

    as_ref(result) = clamp(result, 0.f, 1.f);

    return declare_e(result);
}

template<typename T>
T linear_to_srgb_gpu(const T& in)
{
    return ternary(in <= T(0.0031308f), in * 12.92f, 1.055f * pow(in, 1.0f / 2.4f) - 0.055f);
}

template<typename T>
tensor<T, 3> linear_to_srgb_gpu(const tensor<T, 3>& in)
{
    tensor<T, 3> ret;

    for(int i=0; i < 3; i++)
        ret[i] = linear_to_srgb_gpu(in[i]);

    return ret;
}

template<typename T>
T srgb_to_linear_gpu(const T& in)
{
    return ternary(in < T(0.04045f), in / 12.92f, pow((in + 0.055f) / 1.055f, T(2.4f)));
}

///https://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
template<typename T>
tensor<T, 3> srgb_to_linear_gpu(const tensor<T, 3>& in)
{
    tensor<T, 3> ret;

    for(int i=0; i < 3; i++)
        ret[i] = srgb_to_linear_gpu(in[i]);

    return ret;
}

valuef get_zp1(v4f position_obs, v4f velocity_obs, v4f ref_obs, v4f position_emit, v4f velocity_emit, v4f ref_emit, auto&& get_metric)
{
    using namespace single_source;

    m44f guv_obs = get_metric(position_obs);
    m44f guv_emit = get_metric(position_emit);

    valuef zp1 = dot_metric(velocity_emit, ref_emit, guv_emit) / dot_metric(velocity_obs, ref_obs, guv_obs);

    pin(zp1);

    return zp1;
}

v3f do_redshift(v3f colour, valuef zp1)
{
    using namespace single_source;

    return redshift(colour, zp1 - 1);
}

auto cartesian_to_spherical = []<typename T>(const tensor<T, 3>& cartesian)
{
    T r = cartesian.length();
    T theta = acos(cartesian[2] / r);
    T phi = atan2(cartesian[1], cartesian[0]);

    return tensor<T, 3>{r, theta, phi};
};

valuef W_f_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    auto W_at = [&](v3i pos)
    {
        return in.W[pos, dim];
    };

    return function_trilinear(W_at, pos);
}

valuef gA_f_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    auto func = [&](v3i pos)
    {
        return in.gA[pos, dim];
    };

    auto val = function_trilinear(func, pos);
    pin(val);
    return val;
}

tensor<valuef, 3> gB_f_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    auto func = [&](v3i pos)
    {
        return adm_at(pos, dim, in).gB;
    };

    auto val = function_trilinear(func, pos);
    pin(val);
    return val;
}

unit_metric<valuef, 3, 3> cY_f_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    auto func = [&](v3i pos)
    {
        bssn_args args(pos, dim, in);
        return args.cY;
    };

    auto val = function_trilinear(func, pos);
    pin(val);
    return val;
}

valuef get_ct_timestep(valuef W)
{
    float X_far = 0.9f;
    float X_near = 0.6f;

    valuef X = W*W;

    valuef my_fraction = (X - X_near) / (X_far - X_near);

    my_fraction = clamp(my_fraction, 0.f, 1.f);

    return mix(valuef(0.1f), valuef(1.f), my_fraction);
}

v3f fix_ray_position_cart(v3f cartesian_pos, v3f cartesian_velocity, valuef sphere_radius)
{
    using namespace single_source;

    cartesian_velocity = cartesian_velocity.norm();

    v3f C = (v3f){0,0,0};

    valuef a = 1;
    valuef b = 2 * dot(cartesian_velocity, (cartesian_pos - C));
    valuef c = dot(C, C) + dot(cartesian_pos, cartesian_pos) - 2 * dot(cartesian_pos, C) - sphere_radius * sphere_radius;

    valuef discrim = b*b - 4 * a * c;

    valuef t0 = (-b - sqrt(discrim)) / (2 * a);
    valuef t1 = (-b + sqrt(discrim)) / (2 * a);

    valuef my_t = ternary(fabs(t0) < fabs(t1), t0, t1);

    v3f result_good = cartesian_pos + my_t * cartesian_velocity;

    return ternary(discrim >= 0, result_good, cartesian_pos);
}

void bssn_to_guv(execution_context& ectx, literal<v3i> upper_dim, literal<v3i> lower_dim,
                 bssn_args_mem<buffer<valuef>> in,
                 std::array<buffer_mut<block_precision_t>, 10> Guv, literal<value<uint64_t>> slice)
{
    using namespace single_source;

    valuei x = value_impl::get_global_id(0);
    valuei y = value_impl::get_global_id(1);
    valuei z = value_impl::get_global_id(2);

    pin(x);
    pin(y);
    pin(z);

    if_e(x >= lower_dim.get().x() || y >= lower_dim.get().y() || z >= lower_dim.get().z(), [&]{
        return_e();
    });

    v3i pos_lo = {x, y, z};

    //builds the 4-metric from the bssn variables
    auto get_metric = [&](v3i posu)
    {
        bssn_args args(posu, upper_dim.get(), in);

        metric<valuef, 3, 3> Yij = args.cY / max(args.W * args.W, valuef(0.0001f));
        metric<valuef, 4, 4> met = calculate_real_metric(Yij, args.gA, args.gB);
        pin(met);

        return met;
    };

    v3i centre_lo = (lower_dim.get() - 1)/2;
    v3i centre_hi = (upper_dim.get() - 1)/2;

    //calculating the scaling factor, to convert from the smaller mesh's coordinate system to the larger mesh
    valuef to_upper = (valuef)centre_hi.x() / (valuef)centre_lo.x();

    //larger mesh coordinate
    v3f f_upper = (v3f)pos_lo * to_upper;

    //calculate and interpolate the metric from the higher resolution mesh, and the bssn variables
    metric<valuef, 4, 4> met = function_trilinear(get_metric, f_upper);

    vec2i indices[10] = {
        {0, 0}, {1, 0}, {2, 0}, {3, 0},
        {1, 1}, {2, 1}, {3, 1},
        {2, 2}, {3, 2},
        {3, 3},
    };

    tensor<value<uint64_t>, 3> p = (tensor<value<uint64_t>, 3>)pos_lo;
    tensor<value<uint64_t>, 3> d = (tensor<value<uint64_t>, 3>)lower_dim.get();

    for(int i=0; i < 10; i++)
    {
        vec2i idx = indices[i];

        //the index coordinate system is t, z, y, x
        value<uint64_t> lidx = p.z() * d.x() * d.y() + p.y() * d.x() + p.x() + slice.get() * d.x() * d.y() * d.z();

        as_ref(Guv[i][lidx]) = (block_precision_t)met[idx.x(), idx.y()];
    }
}

valuef acceleration_to_precision(v4f acceleration, valuef max_acceleration)
{
    valuef diff = acceleration.length() * 0.01f;

    return sqrt(max_acceleration / diff);
}

struct trace4_state
{

};

void trace4x4(execution_context& ectx, literal<v2i> screen_sizel,
            buffer_mut<v4f> positions, buffer_mut<v4f> velocities,
            buffer_mut<valuei> results, buffer_mut<valuef> zshift,
            literal<v3i> dim,
            literal<valuef> scale,
            literal<valuef> universe_size,
            buffer<v4f> e0, buffer<v4f> e1, buffer<v4f> e2, buffer<v4f> e3,
            std::array<buffer<block_precision_t>, 10> Guv_buf,
            literal<valuef> last_time,
            literal<valuei> last_slice,
            literal<valuef> slice_width)
{
    using namespace single_source;

    if_e(last_slice.get() == 0, [&]{
        return_e();
    });

    valuei x = value_impl::get_global_id(0);
    valuei y = value_impl::get_global_id(1);

    //get_global_id() is not a const function, so assign it to an unnamed variable to avoid compilers repeatedly evaluating it
    pin(x);
    pin(y);

    v2i screen_size = screen_sizel.get();

    if_e(y >= screen_size.y(), [&] {
        return_e();
    });

    if_e(x >= screen_size.x(), [&] {
        return_e();
    });

    v2i screen_position = {x, y};

    v4f pos_in = declare_e(positions[screen_position, screen_size]);
    v4f vel_in = declare_e(velocities[screen_position, screen_size]);

    auto world_to_grid4 = [&](v4f position) {
        v3f grid_position = world_to_grid(position.yzw(), dim.get(), scale.get());
        valuef grid_t_frac = position.x() / last_time.get();
        valuef grid_t = grid_t_frac * (valuef)last_slice.get();

        pin(grid_t);

        grid_t = ternary(last_slice.get() >= 5,
                         clamp(grid_t, valuef(1), (valuef)last_slice.get() - 3),
                         grid_t);

        grid_position = clamp(grid_position, (v3f){3,3,3}, (v3f)dim.get() - (v3f){4,4,4});
        pin(grid_position);

        v4f grid_fpos = (v4f){grid_t, grid_position.x(), grid_position.y(), grid_position.z()};
        return grid_fpos;
    };

    auto get_dX = [&](v4f position, v4f velocity, trace4_state st) {
        return velocity;
    };

    auto get_dV = [&](v4f position, v4f velocity, trace4_state st) {
        v4f grid_fpos = world_to_grid4(position);

        auto get_Guvb = [&](v4i pos) {
            auto v = Guv_at(pos, dim.get(), Guv_buf, last_slice.get());
            pin(v);
            return v;
        };

        auto get_guv_at = [&](v4f fpos) {
            return function_quadlinear(get_Guvb, fpos);
        };

        auto Guv = get_guv_at(grid_fpos);

        tensor<valuef, 4, 4, 4> dGuv;

        for(int m=0; m < 4; m++)
        {
            v4f dir;
            dir[m] = 1;

            valuef divisor = 2 * scale.get();

            if(m == 0)
                divisor = 2 * slice_width.get();

            v4f p1 = grid_fpos + dir;
            v4f p2 = grid_fpos - dir;

            pin(p1);
            pin(p2);

            auto g1 = get_guv_at(p1);
            pin(g1);

            auto g2 = get_guv_at(p2);
            pin(g2);

            auto diff = g1 - g2;
            pin(diff);

            auto val = diff / divisor;

            for(int i=0; i < 4; i++)
            {
                for(int j=0; j < 4; j++)
                {
                    dGuv[m, i, j] = val[i, j];
                }
            }
        }

        pin(Guv);
        pin(dGuv);

        auto iGuv = Guv.invert();
        //pin(iGuv);

        auto christoff2 = christoffel_symbols_2(iGuv, dGuv);

        pin(christoff2);

        v4f accel;

        for(int uu=0; uu < 4; uu++)
        {
            valuef sum = 0;

            for(int aa = 0; aa < 4; aa++)
            {
                for(int bb = 0; bb < 4; bb++)
                {
                    sum += velocity[aa] * velocity[bb] * christoff2[uu, aa, bb];
                }
            }

            accel[uu] = -sum;
        }

        pin(accel);

        return accel;
    };

    auto get_dS = [&](v4f position, v4f velocity, v4f acceleration, trace4_state st) {
        return acceleration_to_precision(acceleration, 0.0002f);
    };

    auto get_state = [](v4f position) {
        return trace4_state();
    };

    auto velocity_process = [](v4f v, const trace4_state& st) {
        return v;
    };

    verlet_context<valuef, 4> ctx;
    ctx.start(pos_in, vel_in, get_dX, get_dV, get_dS, get_state);

    mut<valuei> result = declare_mut_e(valuei(2));
    mut<valuei> idx = declare_mut_e("i", valuei(0));

    for_e(idx < 512, assign_b(idx, idx + 1), [&]
    {
        ctx.next(get_dX, get_dV, get_dS, get_state, velocity_process);

        v4f cposition = as_constant(ctx.position);
        v4f cvelocity = as_constant(ctx.velocity);

        valuef radius_sq = dot(cposition.yzw(), cposition.yzw());

        if_e(radius_sq > universe_size.get()*universe_size.get(), [&] {
            //ray escaped
            as_ref(result) = valuei(1);
            break_e();
        });

        if_e(cposition.x() < -150 || fabs(cvelocity.x()) > 30 || cvelocity.yzw().squared_length() < 0.1f * 0.1f ||
                !isfinite(cposition.x()) || !isfinite(cposition.y()) || !isfinite(cposition.z()) || !isfinite(cposition.w()) ||
                !isfinite(cvelocity.x()) || !isfinite(cvelocity.y()) || !isfinite(cvelocity.z()) || !isfinite(cvelocity.w())
                , [&]{
            as_ref(result) = valuei(0);
            break_e();
        });
    });

    v4f final_position = declare_e(ctx.position);
    v4f final_velocity = declare_e(ctx.velocity);

    auto get_Guvb = [&](v4i pos)
    {
        auto val = Guv_at(pos, dim.get(), Guv_buf, last_slice.get());
        pin(val);
        return val;
    };

    auto position_to_metric = [&](v4f fpos)
    {
        auto val = function_quadlinear(get_Guvb, world_to_grid4(fpos));
        pin(val);
        return val;
    };

    valuef zp1 = get_zp1(pos_in, vel_in, e0[0], final_position, final_velocity, (v4f){1, 0, 0, 0}, position_to_metric);

    as_ref(results[screen_position, screen_size]) = as_constant(result);
    as_ref(zshift[screen_position, screen_size]) = zp1 - 1;
    as_ref(positions[screen_position, screen_size]) = final_position;
    as_ref(velocities[screen_position, screen_size]) = final_velocity;
}

void calculate_texture_coordinates(execution_context& ectx, literal<v2i> screen_sizel,
                                   buffer<v4f> positions, buffer<v4f> velocities,
                                   buffer_mut<v2f> out, literal<valuef> universe_size)
{
    using namespace single_source;

    valuei x = value_impl::get_global_id(0);
    valuei y = value_impl::get_global_id(1);

    pin(x);
    pin(y);

    v2i screen_size = screen_sizel.get();

    if_e(y >= screen_size.y(), [&] {
        return_e();
    });

    if_e(x >= screen_size.x(), [&] {
        return_e();
    });

    v2i screen_position = {x, y};

    v3f position3 = positions[screen_position, screen_size].yzw();
    v3f velocity3 = velocities[screen_position, screen_size].yzw();

    position3 = fix_ray_position_cart(position3, velocity3, universe_size.get());

    v2f angle = cartesian_to_spherical(position3).yz();

    v2f texture_coordinate = angle_to_tex(angle);

    v2f normed = fmod(texture_coordinate + (v2f){1, 1}, (v2f){1, 1});

    as_ref(out[screen_position, screen_size]) = normed;
}

valuef circular_diff(valuef f1, valuef f2, valuef period)
{
    f1 = f1 * (2 * M_PI/period);
    f2 = f2 * (2 * M_PI/period);

    //return period * fast_pseudo_atan2(f2 - f1) / (2 * M_PI);
    return period * atan2(sin(f2 - f1), cos(f2 - f1)) / (2 * M_PI);
}

v2f circular_diff2(v2f f1, v2f f2)
{
    return {circular_diff(f1.x(), f2.x(), 1.f), circular_diff(f1.y(), f2.y(), 1.f)};
}

v3f read_mipmap(read_only_image_array<2> img, v3f coord)
{
    using namespace single_source;

    ///I don't think this is technically necessary
    coord.z() = max(coord.z(), valuef(0.f));

    std::vector<std::string> flags = {sampler_flag::NORMALIZED_COORDS_TRUE, sampler_flag::FILTER_LINEAR, sampler_flag::ADDRESS_REPEAT};

    valuef mip_lower = floor(coord.z());
    valuef mip_upper = ceil(coord.z());

    valuef lower_divisor = pow(valuef(2.f), mip_lower);
    valuef upper_divisor = pow(valuef(2.f), mip_upper);

    v2f lower_coord = coord.xy() / lower_divisor;
    v2f upper_coord = coord.xy() / upper_divisor;

    v3f full_lower_coord = (v3f){lower_coord.x(), lower_coord.y(), mip_lower};
    v3f full_upper_coord = (v3f){upper_coord.x(), upper_coord.y(), mip_upper};

    valuef weight = coord.z() - mip_lower;

    v3f val = mix(img.read<float, 4>(full_lower_coord, flags), img.read<float, 4>(full_upper_coord, flags), weight).xyz();
    pin(val);
    return srgb_to_linear_gpu(val);
}

void render(execution_context& ectx, literal<v2i> screen_sizel,
            buffer<v4f> positions, buffer<v4f> velocities, buffer<valuei> results, buffer<valuef> zshift, buffer<v4f> matter_colour,
            buffer<v2f> texture_coordinates,
            read_only_image_array<2> background, write_only_image<2> screen,
            literal<v2i> background_size,
            literal<valuei> background_array_length)
{
    using namespace single_source;

    valuei x = value_impl::get_global_id(0);
    valuei y = value_impl::get_global_id(1);

    pin(x);
    pin(y);

    v2i screen_size = screen_sizel.get();

    if_e(y >= screen_size.y(), [&] {
        return_e();
    });

    if_e(x >= screen_size.x(), [&] {
        return_e();
    });

    v2i screen_position = {x, y};

    if_e(results[screen_position, screen_size] == 0 || results[screen_position, screen_size] == 2, [&]{
        v4f colour = matter_colour[screen_position, screen_size];

        colour = clamp(colour, v4f{0,0,0,0}, v4f{1,1,1,1});

        valuef zp1 = declare_e(zshift[screen_position, screen_size]) + 1;

        v3f cvt = colour.xyz();

        cvt = clamp(cvt, v3f{0,0,0}, v3f{1,1,1});

        cvt = linear_to_srgb_gpu(cvt);

        screen.write(ectx, {x, y}, (v4f){cvt.x(),cvt.y(),cvt.z(),1});
        return_e();
    });

    /*if_e(results[screen_position, screen_size] == 2, [&]{
        screen.write(ectx, {x, y}, (v4f){0,0,0,1});
        return_e();
    });*/

    //#define BILINEAR
    #ifdef BILINEAR
    v2f texture_coordinate = texture_coordinates[screen_position, screen_size];

    v3f to_read = {texture_coordinate.x(), texture_coordinate.y(), 0.f};

    v3f col = background.read<float, 4>(to_read, {sampler_flag::NORMALIZED_COORDS_TRUE, sampler_flag::FILTER_LINEAR, sampler_flag::ADDRESS_REPEAT}).xyz();

    v3f cvt = col;
    #endif

    #define ANISOTROPIC
    #ifdef ANISOTROPIC
    mut<valuei> dx = declare_mut_e(valuei(1));
    mut<valuei> dy = declare_mut_e(valuei(1));

    ///check if we're on a non terminating boundary
    {
        v2i vdx = {declare_e(dx), valuei(0)};
        v2i vdy = {valuei(0), declare_e(dy)};

        if_e(results[screen_position + vdx, screen_size] == 0, [&]{
            as_ref(dx) = -as_constant(dx);
        });

        if_e(results[screen_position + vdy, screen_size] == 0, [&]{
            as_ref(dy) = -as_constant(dy);
        });
    }

    if_e(x == screen_size.x() - 1, [&]{
        as_ref(dx) = valuei(-1);
    });

    if_e(y == screen_size.y() - 1, [&]{
        as_ref(dy) = valuei(-1);
    });

    if_e(x == 0, [&]{
        as_ref(dx) = valuei(1);
    });

    if_e(y == 0, [&]{
        as_ref(dy) = valuei(1);
    });

    valuei cdx = declare_e(dx);
    valuei cdy = declare_e(dy);

    v2f tl = texture_coordinates[screen_position, screen_size];
    v2f tr = texture_coordinates[screen_position + (v2i){cdx, 0}, screen_size];
    v2f bl = texture_coordinates[screen_position + (v2i){0, cdy}, screen_size];

    pin(tl);
    pin(tr);
    pin(bl);

    float bias_frac = 1.3;

    v2f dx_vtc = (valuef)cdx * circular_diff2(tl, tr) / bias_frac;
    v2f dy_vtc = (valuef)cdy * circular_diff2(tl, bl) / bias_frac;

    dx_vtc = dx_vtc * (v2f)background_size.get();
    dy_vtc = dy_vtc * (v2f)background_size.get();

    ///http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1002.1336&rep=rep1&type=pdf
    valuef dv_dx = dx_vtc.y();
    valuef dv_dy = dy_vtc.y();

    valuef du_dx = dx_vtc.x();
    valuef du_dy = dy_vtc.x();

    valuef Ann = dv_dx * dv_dx + dv_dy * dv_dy;
    valuef Bnn = -2 * (du_dx * dv_dx + du_dy * dv_dy);
    valuef Cnn = du_dx * du_dx + du_dy * du_dy; ///only tells lies

    ///hecc
    #define HECKBERT
    #ifdef HECKBERT
    Ann = dv_dx * dv_dx + dv_dy * dv_dy + 1;
    Cnn = du_dx * du_dx + du_dy * du_dy + 1;
    #endif // HECKBERT

    valuef F = Ann * Cnn - Bnn * Bnn / 4;
    valuef A = Ann / F;
    valuef B = Bnn / F;
    valuef C = Cnn / F;

    valuef root = sqrt((A - C) * (A - C) + B*B);
    valuef a_prime = (A + C - root) / 2;
    valuef c_prime = (A + C + root) / 2;

    valuef majorRadius = 1/sqrt(a_prime);
    valuef minorRadius = 1/sqrt(c_prime);

    valuef theta = atan2(B, (A - C)/2);

    majorRadius = max(majorRadius, valuef(1.f));
    minorRadius = max(minorRadius, valuef(1.f));

    majorRadius = max(majorRadius, minorRadius);

    valuef fProbes = 2 * (majorRadius / minorRadius) - 1;
    mut<valuei> iProbes = declare_mut_e((valuei)floor(fProbes + 0.5f));

    valuei maxProbes = 32;

    as_ref(iProbes) = min(declare_e(iProbes), maxProbes);

    pin(iProbes);

    //if(iProbes < fProbes)
    //    minorRadius = 2 * majorRadius / (iProbes + 1);

    minorRadius = ternary((valuef)iProbes < fProbes,
                          2 * majorRadius / ((valuef)iProbes + 1),
                          minorRadius);

    mut<valuef> levelofdetail = declare_mut_e(log2(minorRadius));

    valuef maxLod = (valuef)background_array_length.get() - 1;

    if_e(as_constant(levelofdetail) > maxLod, [&]{
        as_ref(levelofdetail) = maxLod;
        as_ref(iProbes) = valuei(1);
    });

    mut_v3f end_result = declare_mut_e((v3f){0,0,0});

    std::vector<std::string> flags = {sampler_flag::NORMALIZED_COORDS_TRUE, sampler_flag::FILTER_LINEAR, sampler_flag::ADDRESS_REPEAT};

    value<bool> is_trilinear = iProbes == 1 || iProbes <= 1;

    if_e(is_trilinear, [&]{
        if_e(iProbes < 1, [&]{
            as_ref(levelofdetail) = maxLod;
        });

        v3f coord = {tl.x(), tl.y(), as_constant(levelofdetail)};

        as_ref(end_result) = read_mipmap(background, coord);
    });

    if_e(!is_trilinear, [&]{
        valuef lineLength = 2 * (majorRadius - minorRadius);
        valuef du = cos(theta) * lineLength / ((valuef)iProbes - 1);
        valuef dv = sin(theta) * lineLength / ((valuef)iProbes - 1);

        mut_v3f totalWeight = declare_mut_e((v3f){0,0,0});
        mut<valuef> accumulatedProbes = declare_mut_e(valuef(0.f));

        mut<valuei> startN = declare_mut_e(valuei(0));

        value<bool> is_odd = (iProbes % 2) == 1;

        if_e(is_odd, [&]{
            valuei probeArm = (iProbes - 1) / 2;

            as_ref(startN) = -2 * probeArm;
        });

        if_e(!is_odd, [&]{
            valuei probeArm = iProbes / 2;

            as_ref(startN) = -2 * probeArm - 1;
        });

        mut<valuei> currentN = declare_mut_e(startN);
        float alpha = 2;

        valuef sU = du / (valuef)background_size.get().x();
        valuef sV = dv / (valuef)background_size.get().y();

        mut<valuei> idx = declare_mut_e(valuei(0));

        for_e(idx < iProbes, assign_b(idx, idx + 1), [&] {
            valuef d_2 = ((valuef)(currentN * currentN) / 4.f) * (du * du + dv * dv) / (majorRadius * majorRadius);

            valuef relativeWeight = exp(-alpha * d_2);

            valuef centreu = tl.x();
            valuef centrev = tl.y();

            valuef cu = centreu + ((valuef)currentN / 2.f) * sU;
            valuef cv = centrev + ((valuef)currentN / 2.f) * sV;

            v3f fval = read_mipmap(background, {cu, cv, as_constant(levelofdetail)});

            as_ref(totalWeight) = declare_e(totalWeight) + relativeWeight * fval;
            as_ref(accumulatedProbes) = declare_e(accumulatedProbes) + relativeWeight;
            as_ref(currentN) = declare_e(currentN) + valuei(2);
        });

        as_ref(end_result) = as_constant(totalWeight) / accumulatedProbes;
    });

    v3f cvt = declare_e(end_result);
    #endif // ANISOTROPIC

    v4f colour = matter_colour[screen_position, screen_size];

    colour = clamp(colour, v4f{0,0,0,0}, v4f{1,1,1,1});

    valuef zp1 = declare_e(zshift[screen_position, screen_size]) + 1;

    cvt = do_redshift(cvt, zp1);

    cvt = cvt * (1 - colour.w());

    cvt += colour.xyz();

    cvt = clamp(cvt, v3f{0,0,0}, v3f{1,1,1});

    cvt = linear_to_srgb_gpu(cvt);

    v4f crgba = {cvt[0], cvt[1], cvt[2], 1.f};

    screen.write(ectx, {x, y}, crgba);
}

struct trace3_state
{
    valuef W;
    unit_metric<valuef, 3, 3> cY;
    tensor<valuef, 3, 3> Kij;
    valuef gA;
    tensor<valuef, 3> gB;

    v3f grid_position;
};

void build_raytrace_kernels(cl::context ctx, const std::vector<plugin*>& plugins, bool use_colour)
{
    auto trace3 = [plugins, use_colour]
                  (execution_context& ectx, literal<v2i> screen_sizel,
                   literal<v4f> camera_quat,
                   buffer_mut<v4f> positions, buffer_mut<v4f> velocities,
                   buffer_mut<valuei> results, buffer_mut<valuef> zshift, buffer_mut<v4f> matter_colour,
                   literal<v3i> dim,
                   literal<valuef> scale,
                   literal<valuef> universe_size,
                   bssn_args_mem<buffer<valuef>> in,
                   bssn_derivatives_mem<buffer<derivative_t>> derivatives,
                   value_impl::builder::placeholder plugin_ph)
    {
        all_adm_args_mem plugin_data = make_arg_provider(plugins);
        plugin_ph.add(plugin_data);

        using namespace single_source;

        valuei x = value_impl::get_global_id(0);
        valuei y = value_impl::get_global_id(1);

        //get_global_id() is not a const function, so assign it to an unnamed variable to avoid compilers repeatedly evaluating it
        pin(x);
        pin(y);

        v2i screen_size = screen_sizel.get();

        if_e(y >= screen_size.y(), [&] {
            return_e();
        });

        if_e(x >= screen_size.x(), [&] {
            return_e();
        });

        v2i screen_position = {x, y};

        v3f pos_in = declare_e(positions[screen_position, screen_size]).yzw();
        v3f vel_in = declare_e(velocities[screen_position, screen_size]).yzw();

        mut<valuei> result = declare_mut_e(valuei(2));

        auto fix_velocity = [](v3f velocity, const trace3_state& args)
        {
            auto cY = args.cY;
            auto W = args.W;

            auto Yij = cY / (W*W);
            //pin(Yij);

            valuef length_sq = dot(velocity, Yij.lower(velocity));
            valuef length = sqrt(fabs(length_sq));

            velocity = velocity / length;

            pin(velocity);

            return velocity;
        };

        auto get_dX = [](v3f position, v3f velocity, const trace3_state& args)
        {
            v3f d_X = args.gA * velocity - args.gB;
            pin(d_X);

            return d_X;
        };

        auto get_dV = [&](v3f position, v3f velocity, const trace3_state& args)
        {
            v3f grid_position = args.grid_position;

            auto dgA_at = [&](v3i pos) {
                bssn_derivatives derivs(pos, dim.get(), derivatives);
                return derivs.dgA;
            };

            auto dgB_at = [&](v3i pos) {
                bssn_derivatives derivs(pos, dim.get(), derivatives);
                return derivs.dgB;
            };

            auto dcY_at = [&](v3i pos) {
                bssn_derivatives derivs(pos, dim.get(), derivatives);
                return derivs.dcY;
            };

            auto dW_at = [&](v3i pos) {
                bssn_derivatives derivs(pos, dim.get(), derivatives);
                return derivs.dW;
            };

            tensor<valuef, 3> dgA = function_trilinear(dgA_at, grid_position);
            tensor<valuef, 3, 3> dgB = function_trilinear(dgB_at, grid_position);
            tensor<valuef, 3, 3, 3> dcY = function_trilinear(dcY_at, grid_position);
            tensor<valuef, 3> dW = function_trilinear(dW_at, grid_position);

            pin(dgA);
            pin(dgB);
            pin(dcY);
            pin(dW);

            auto cY = args.cY;
            auto W = args.W;

            auto icY = cY.invert();
            pin(icY);

            auto iYij = icY * (W*W);

            auto christoff2_cfl = christoffel_symbols_2(icY, dcY);
            pin(christoff2_cfl);

            auto christoff2 = get_full_christoffel2(W, dW, cY, icY, christoff2_cfl);
            pin(christoff2);

            tensor<valuef, 3> d_V;

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    valuef kjvk = 0;

                    for(int k=0; k < 3; k++)
                    {
                        kjvk += args.Kij[j, k] * velocity[k];
                    }

                    valuef christoffel_sum = 0;

                    for(int k=0; k < 3; k++)
                    {
                        christoffel_sum += christoff2[i, j, k] * velocity[k];
                    }

                    valuef dlog_gA = dgA[j] / args.gA;

                    d_V[i] += args.gA * velocity[j] * (velocity[i] * (dlog_gA - kjvk) + 2 * raise_index(args.Kij, iYij, 0)[i, j] - christoffel_sum)
                            - iYij[i, j] * dgA[j] - velocity[j] * dgB[j, i];

                }
            }

            return d_V;
        };

        auto get_dS = [&](v3f position, v3f velocity, v3f acceleration, const trace3_state& args)
        {
            return -3.5f * get_ct_timestep(args.W);
        };

        auto get_state = [&](v3f position) -> trace3_state
        {
            trace3_state out;

            v3f grid_position = world_to_grid(position, dim.get(), scale.get());

            grid_position = clamp(grid_position, (v3f){3,3,3}, (v3f)dim.get() - (v3f){4,4,4});
            pin(grid_position);

            auto W = W_f_at(grid_position, dim.get(), in);
            auto cY = cY_f_at(grid_position, dim.get(), in);

            pin(W);
            pin(cY);

            adm_variables args = admf_at(grid_position, dim.get(), in);

            out.W = W;
            out.cY = cY;
            out.Kij = args.Kij;
            out.gA = args.gA;
            out.gB = args.gB;
            out.grid_position = grid_position;

            return out;
        };

        verlet_context<valuef, 3> ctx;
        ctx.start(pos_in, vel_in, get_dX, get_dV, get_dS, get_state);

        mut<valuei> idx = declare_mut_e("i", valuei(0));
        mut_v3f accumulated_colour = declare_mut_e((v3f){0,0,0});
        mut<valuef> accumulated_occlusion = declare_mut_e(valuef());

        for_e(idx < 512, assign_b(idx, idx + 1), [&]
        {
            v3f cposition = declare_e(ctx.position);
            v3f cvelocity = declare_e(ctx.velocity);

            valuef radius_sq = dot(cposition, cposition);

            //terminate if we're out of the sim
            if_e(radius_sq > universe_size.get()*universe_size.get(), [&] {
                as_ref(result) = valuei(1);
                break_e();
            });

            //terminate if our rays become non finite
            if_e(!isfinite(cvelocity.x()) || !isfinite(cvelocity.y()) || !isfinite(cvelocity.z()), [&]{
                as_ref(result) = valuei(0);
                break_e();
            });

            v3f diff = ctx.next(get_dX, get_dV, get_dS, get_state, fix_velocity);

            auto get_dbg = [&](v3i pos)
            {
                derivative_data d;
                d.pos = pos;
                d.dim = dim.get();
                d.scale = scale.get();

                bssn_args args = bssn_at(pos, dim.get(), in);

                valuef p = trace(plugin_data.adm_W2_Sij(args, d), args.cY.invert());
                pin(p);

                return p;
            };

            v3f grid_position = world_to_grid(cposition, dim.get(), scale.get());

            grid_position = clamp(grid_position, (v3f){3,3,3}, (v3f)dim.get() - (v3f){4,4,4});
            pin(grid_position);

            //valuef rho = function_trilinear(get_rho, grid_position);

            v3f colour = {0,0,0};
            valuef density = 0.f;

            if(use_colour)
            {
                auto get_col = [&](v3i pos)
                {
                    derivative_data d;
                    d.pos = pos;
                    d.dim = dim.get();
                    d.scale = scale.get();

                    bssn_args args = bssn_at(pos, dim.get(), in);

                    v3f c = plugin_data.get_colour(args, d);
                    pin(c);

                    return c;
                };

                colour = function_trilinear(get_col, grid_position) * 1000;
                density = colour.length();
            }
            else
            {
                auto get_rho = [&](v3i pos)
                {
                    derivative_data d;
                    d.pos = pos;
                    d.dim = dim.get();
                    d.scale = scale.get();

                    bssn_args args = bssn_at(pos, dim.get(), in);

                    valuef p = plugin_data.adm_p(args, d);
                    pin(p);

                    return p;
                };

                valuef rho = function_trilinear(get_rho, grid_position);

                colour = {rho, rho, rho};
                colour = fabs(colour);
                colour = colour * 1000;

                density = rho * 1000;;
            }

            /*if_e(screen_position.x() == screen_size.x()/2 && screen_position.y() == screen_size.y()/2, [&]{
                valuef S = function_trilinear(get_dbg, grid_position);

                value_base se;
                se.type = value_impl::op::SIDE_EFFECT;
                se.abstract_value = "printf(\"rho: %f\\n\"," + value_to_string(rho) + ")";

                value_impl::get_context().add(se);
            });*/

            valuef sample_length = diff.length();

            as_ref(accumulated_occlusion) += density * sample_length;

            valuef transparency = exp(-as_constant(accumulated_occlusion));

            ///assuming that the intrinsic brightness is a function of density
            as_ref(accumulated_colour) += density * colour * sample_length * transparency;

            if_e(transparency <= 0.001f, [&]{
                as_ref(result) = valuei(1);
                break_e();
            });

            //terminate if the movement of our ray through coordinate space becomes trapped, its likely hit an event horizon
            if_e(diff.squared_length() < 0.1f * 0.1f, [&]
            {
                as_ref(result) = valuei(0);
                break_e();
            });
        });

        valuef final_occlusion = declare_e(accumulated_occlusion);
        v3f final_colour = declare_e(accumulated_colour);

        v3f final_position = declare_e(ctx.position);
        v3f final_velocity = declare_e(ctx.velocity);

        as_ref(positions[screen_position, screen_size]) = (v4f){0, final_position.x(), final_position.y(), final_position.z()};
        as_ref(velocities[screen_position, screen_size]) = (v4f){0, final_velocity.x(), final_velocity.y(), final_velocity.z()};
        as_ref(results[screen_position, screen_size]) = as_constant(result);
        as_ref(matter_colour[screen_position, screen_size]) = (v4f){final_colour.x(), final_colour.y(), final_colour.z(), final_occlusion};
    };

    cl::async_build_and_cache(ctx, [trace3]{
        return value_impl::make_function(trace3, "trace3");
    }, {"trace3"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(bssn_to_guv, "bssn_to_guv");
    }, {"bssn_to_guv"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(trace4x4, "trace4x4");
    }, {"trace4x4"}, "-cl-fast-relaxed-math");

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(calculate_texture_coordinates, "calculate_texture_coordinates");
    }, {"calculate_texture_coordinates"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(render, "render");
    }, {"render"});
}

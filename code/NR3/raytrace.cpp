#include "raytrace.hpp"
#include "tensor_algebra.hpp"

#define UNIVERSE_SIZE 29

///todo: take a function in that returns a state
#define METHOD_1
//#define METHOD_3
template<typename T, int N>
struct verlet_context
{
    tensor<mut<T>, N> position;

    #ifdef METHOD_1
    tensor<mut<T>, N> accel;
    mut<T> ds;
    #endif

    tensor<mut<T>, N> velocity;

    template<typename dX, typename dV, typename dS, typename State>
    void start(const tensor<T, N>& position_in, const tensor<T, N>& velocity_in, dX&& get_dX, dV&& get_dV, dS&& get_dS, State&& get_state)
    {
        using namespace single_source;

        #ifdef METHOD_1
        auto st = get_state(position_in);

        auto acceleration = get_dV(position_in, velocity_in, st);
        pin(acceleration);

        ds = declare_mut_e(get_dS(position_in, velocity_in, acceleration, st));

        accel = declare_mut_e(acceleration);

        auto v_half = velocity_in + 0.5f * acceleration * ds;
        pin(v_half);

        auto dPosition = get_dX(position_in, v_half, st);

        position = declare_mut_e(position_in + dPosition * ds);
        velocity = declare_mut_e(velocity_in);
        #endif

        #ifdef METHOD_2
        position = declare_mut_e(position_in);
        velocity = declare_mut_e(velocity_in);
        #endif

        #ifdef METHOD_3
        position = declare_mut_e(position_in);
        velocity = declare_mut_e(velocity_in);

        auto st = get_state(position_in);

        acceleration = declare_mut_e(get_dV(position_in, velocity_in, st));
        #endif
    }

    template<typename dX, typename dV, typename dS, typename State>
    void next(dX&& get_dX, dV&& get_dV, dS&& get_dS, State&& get_state)
    {
        using namespace single_source;

        #ifdef METHOD_1
        auto cposition = declare_e(position);
        auto cvelocity = declare_e(velocity);
        auto cds = declare_e(ds);
        auto caccel = declare_e(accel);

        auto cv_half = cvelocity + 0.5f * caccel * cds;

        pin(cv_half);

        auto cv_full_approx = cv_half + 0.5f * caccel * cds;

        auto st = get_state(cposition);

        auto AFull_approx = get_dV(cposition, cv_full_approx, st);
        pin(AFull_approx);

        auto VFull_next = cv_half + 0.5f * AFull_approx * cds;
        pin(VFull_next);

        as_ref(velocity) = VFull_next;
        as_ref(ds) = get_dS(cposition, cv_full_approx, AFull_approx, st);
        as_ref(accel) = get_dV(cposition, as_constant(velocity), st);

        auto XDiff = get_dX(cposition, cv_half, st);

        as_ref(position) = cposition + XDiff * as_constant(ds);
        #endif

        #ifdef METHOD_2
        auto cposition = declare_e(position);
        auto cvelocity = declare_e(velocity);

        auto ds = get_dS(cposition, cvelocity, AFull_approx);

        auto v_half = cvelocity + 0.5f * get_dV(cposition, cvelocity) * ds;
        auto x_full = cposition + v_half * ds;

        auto v_full_approx = cvelocity + get_dV(cposition, cvelocity) * ds;

        auto a_full = get_dV(x_full, v_full_approx);

        auto v_next = v_half + 0.5f * a_full * ds;

        as_ref(position) = x_full;
        as_ref(velocity) = v_next;
        #endif
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
    void next(dX&& get_dX, dV&& get_dV, dS&& get_dS, State&& get_state)
    {
        using namespace single_source;

        auto cposition = declare_e(position);
        auto cvelocity = declare_e(velocity);

        auto st = get_state(cposition);

        auto accel = declare_e(get_dV(cposition, cvelocity, st));
        auto dPosition = declare_e(get_dX(cposition, cvelocity, st));

        auto ds = declare_e(get_dS(cposition, cvelocity, accel, st));

        as_ref(position) = cposition + dPosition * ds;
        as_ref(velocity) = cvelocity + accel * ds;
    }
};

template<auto GetMetric, typename... T>
inline
void build_initial_tetrads(execution_context& ectx, literal<v4f> position,
                           literal<v3f> local_velocity,
                           buffer_mut<v4f> position_out,
                           buffer_mut<v4f> e0_out, buffer_mut<v4f> e1_out, buffer_mut<v4f> e2_out, buffer_mut<v4f> e3_out,
                           T... extra)
{
    using namespace single_source;

    as_ref(position_out[0]) = position.get();

    v4f v0 = {1, 0, 0, 0};
    v4f v1 = {0, 1, 0, 0};
    v4f v2 = {0, 0, 1, 0};
    v4f v3 = {0, 0, 0, 1};

    m44f metric = GetMetric(position.get(), extra...);

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
        v3f cart = position.get().yzw();

        v4f dx = (v4f){0, 1, 0, 0};
        v4f dy = (v4f){0, 0, 1, 0};
        v4f dz = (v4f){0, 0, 0, 1};

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

//calculate Y of XYZ
inline
valuef energy_of(v3f v)
{
    return v.x()*0.2125f + v.y()*0.7154f + v.z()*0.0721f;
}

inline
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
inline
T linear_to_srgb_gpu(const T& in)
{
    return ternary(in <= T(0.0031308f), in * 12.92f, 1.055f * pow(in, 1.0f / 2.4f) - 0.055f);
}

template<typename T>
inline
tensor<T, 3> linear_to_srgb_gpu(const tensor<T, 3>& in)
{
    tensor<T, 3> ret;

    for(int i=0; i < 3; i++)
        ret[i] = linear_to_srgb_gpu(in[i]);

    return ret;
}

template<typename T>
inline
T srgb_to_linear_gpu(const T& in)
{
    return ternary(in < T(0.04045f), in / 12.92f, pow((in + 0.055f) / 1.055f, T(2.4f)));
}

///https://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
template<typename T>
inline
tensor<T, 3> srgb_to_linear_gpu(const tensor<T, 3>& in)
{
    tensor<T, 3> ret;

    for(int i=0; i < 3; i++)
        ret[i] = srgb_to_linear_gpu(in[i]);

    return ret;
}

inline
valuef get_zp1(v4f position_obs, v4f velocity_obs, v4f ref_obs, v4f position_emit, v4f velocity_emit, v4f ref_emit, auto&& get_metric)
{
    using namespace single_source;

    m44f guv_obs = get_metric(position_obs);
    m44f guv_emit = get_metric(position_emit);

    valuef zp1 = dot_metric(velocity_emit, ref_emit, guv_emit) / dot_metric(velocity_obs, ref_obs, guv_obs);

    pin(zp1);

    return zp1;
}

inline
v3f do_redshift(v3f colour, valuef zp1)
{
    using namespace single_source;

    return redshift(colour, zp1 - 1);
}

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

inline
bssn_args bssn_at(v3i pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    bssn_args args(pos, dim, in);
    return args;
}

inline
valuef W_f_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    auto W_at = [&](v3i pos)
    {
        bssn_args args(pos, dim, in);
        return args.W;
    };

    return function_trilinear(W_at, pos);
}

inline
valuef gA_f_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    auto func = [&](v3i pos)
    {
        return adm_at(pos, dim, in).gA;
    };

    auto val = function_trilinear(func, pos);
    pin(val);
    return val;
}

inline
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


inline
unit_metric<valuef, 3, 3> cY_f_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    auto func = [&](v3i pos)
    {
        return bssn_at(pos, dim, in).cY;
    };

    auto val = function_trilinear(func, pos);
    pin(val);
    return val;
}

///this is totally pointless, velocity = 1
valuef get_ct_timestep(v3f position, v3f velocity, valuef W)
{
    float X_far = 0.9f;
    float X_near = 0.6f;

    valuef X = W*W;

    valuef my_fraction = (clamp(X, X_near, X_far) - X_near) / (X_far - X_near);

    my_fraction = clamp(my_fraction, 0.f, 1.f);

    return mix(valuef(0.4f), valuef(4.f), my_fraction) * 0.4f;
}

///so. I think the projection is wrong, and that we should have -dt
///but i need to test the 4-iteration realistically
void init_rays3(execution_context& ectx, literal<valuei> screen_width, literal<valuei> screen_height,
               buffer_mut<v4f> positions_out,
               buffer_mut<v4f> velocities_out,
               buffer<v4f> e0, buffer<v4f> e1, buffer<v4f> e2, buffer<v4f> e3,
               buffer<v4f> position, literal<v4f> camera_quat,
               literal<v3i> dim, literal<valuef> scale, literal<valuei> is_adm,
               bssn_args_mem<buffer<valuef>> in)
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

    v2i screen_position = {x, y};
    v2i screen_size = {screen_width.get(), screen_height.get()};

    tetrad tetrads = {e0[0], e1[0], e2[0], e3[0]};

    v3f grid_position = world_to_grid(position[0].yzw(), dim.get(), scale.get());

    grid_position = clamp(grid_position, (v3f){2,2,2}, (v3f)dim.get() - (v3f){3,3,3});

    pin(grid_position);

    v3f ray_direction = get_ray_through_pixel(screen_position, screen_size, 90, camera_quat.get());

    geodesic my_geodesic = make_lightlike_geodesic(position[0], ray_direction, tetrads);

    v2i out_dim = {screen_width.get(), screen_height.get()};
    v2i out_pos = {x, y};

    as_ref(positions_out[out_pos, out_dim]) = my_geodesic.position;

    if_e(is_adm.get() == 1, [&]{

        adm_variables init_adm = admf_at(grid_position, dim.get(), in);

        tensor<valuef, 4> normal = get_adm_hypersurface_normal_raised(init_adm.gA, init_adm.gB);
        tensor<valuef, 4> normal_lowered = get_adm_hypersurface_normal_lowered(init_adm.gA);

        valuef E = sum_multiply(my_geodesic.velocity, normal_lowered);

        ///98% sure this is wrong, but past me had a lot of qualms about this and was careful so...
        tensor<valuef, 4> adm_velocity = -((my_geodesic.velocity / E) - normal);

        as_ref(velocities_out[out_pos, out_dim]) = adm_velocity;
    });

    if_e(is_adm.get() == 0, [&]{
        as_ref(velocities_out[out_pos, out_dim]) = my_geodesic.velocity;
    });
}

v3f fix_ray_position_cart(v3f cartesian_pos, v3f cartesian_velocity, float sphere_radius)
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

struct trace3_state
{
    valuef W;
    unit_metric<valuef, 3, 3> cY;
    tensor<valuef, 3, 3> Kij;
    valuef gA;
    tensor<valuef, 3> gB;

    v3f grid_position;
};

void trace3(execution_context& ectx, literal<valuei> screen_width, literal<valuei> screen_height,
                     literal<v4f> camera_quat,
                     buffer_mut<v4f> positions, buffer_mut<v4f> velocities,
                     buffer_mut<valuei> results, buffer_mut<valuef> zshift,
                     literal<v3i> dim,
                     literal<valuef> scale,
                     bssn_args_mem<buffer<valuef>> in,
                     bssn_derivatives_mem<buffer<derivative_t>> derivatives)
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

    v2i screen_position = {x, y};
    v2i screen_size = {screen_width.get(), screen_height.get()};

    v3f pos_in = declare_e(positions[screen_position, screen_size]).yzw();
    v3f vel_in = declare_e(velocities[screen_position, screen_size]).yzw();

    mut<valuei> result = declare_mut_e(valuei(0));
    v3f final_position;
    mut_v3f final_velocity = declare_mut_e((v3f){});
    //mut_v3f final_dX = declare_mut_e((v3f){});

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

    auto get_dX = [&](v3f position, v3f velocity, const trace3_state& args)
    {
        v3f d_X = args.gA * fix_velocity(velocity, args) - args.gB;
        pin(d_X);

        //so. This recovers the performance, but its pretty hacky
        /*if_e(dot(d_X, d_X) < 0.2f * 0.2f, [&]
        {
            as_ref(result) = valuei(0);
            break_e();
        });*/

        return d_X;
    };

    auto get_dV = [&](v3f position, v3f velocity, const trace3_state& args)
    {
        v3f grid_position = args.grid_position;

        auto dgA_at = [&](v3i pos)
        {
            bssn_derivatives derivs(pos, dim.get(), derivatives);
            return derivs.dgA;
        };

        auto dgB_at = [&](v3i pos)
        {
            bssn_derivatives derivs(pos, dim.get(), derivatives);
            return derivs.dgB;
        };

        auto dcY_at = [&](v3i pos)
        {
            bssn_derivatives derivs(pos, dim.get(), derivatives);
            return derivs.dcY;
        };

        auto dW_at = [&](v3i pos)
        {
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

        velocity = fix_velocity(velocity, args);

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
        return -1.f * get_ct_timestep(position, velocity, args.W);
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

    #if 1
    verlet_context<valuef, 3> ctx;
    ctx.start(pos_in, vel_in, get_dX, get_dV, get_dS, get_state);

    mut<valuei> idx = declare_mut_e("i", valuei(0));

    for_e(idx < 1024, assign_b(idx, idx + 1), [&]
    {
        v3f cposition = declare_e(ctx.position);
        v3f cvelocity = declare_e(ctx.velocity);

        valuef radius_sq = dot(cposition, cposition);

        if_e(radius_sq > UNIVERSE_SIZE*UNIVERSE_SIZE, [&] {
            as_ref(final_velocity) = cvelocity;
            as_ref(result) = valuei(1);
            break_e();
        });

        ctx.next(get_dX, get_dV, get_dS, get_state);

        if_e(dot((cvelocity), (cvelocity)) < 0.3f * 0.3f, [&]
        {
            as_ref(result) = valuei(0);
            break_e();
        });
    });

    final_position = declare_e(ctx.position);;
    //final_velocity = declare_e(ctx.velocity);;
    #endif

    #if 0
    {
        mut_v3f position = declare_mut_e(pos_in);
        mut_v3f velocity = declare_mut_e(vel_in);

        mut<valuei> idx = declare_mut_e("i", valuei(0));

        for_e(idx < 2048, assign_b(idx, idx + 1), [&]
        {
            v3f cposition = declare_e(position);
            v3f cvelocity = declare_e(velocity);

            valuef radius_sq = dot(cposition, cposition);

            if_e(radius_sq > UNIVERSE_SIZE*UNIVERSE_SIZE, [&] {
                as_ref(final_velocity) = cvelocity;
                as_ref(result) = valuei(1);
                break_e();
            });

            v3f grid_position = world_to_grid(cposition, dim.get(), scale.get());

            grid_position = clamp(grid_position, (v3f){3,3,3}, (v3f)dim.get() - (v3f){4,4,4});
            pin(grid_position);

            auto dgA_at = [&](v3i pos)
            {
                bssn_derivatives derivs(pos, dim.get(), derivatives);
                return derivs.dgA;
            };

            auto dgB_at = [&](v3i pos)
            {
                bssn_derivatives derivs(pos, dim.get(), derivatives);
                return derivs.dgB;
            };

            auto dcY_at = [&](v3i pos)
            {
                bssn_derivatives derivs(pos, dim.get(), derivatives);
                return derivs.dcY;
            };

            auto dW_at = [&](v3i pos)
            {
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


            auto W = W_f_at(grid_position, dim.get(), in);
            auto cY = cY_f_at(grid_position, dim.get(), in);

            pin(W);
            pin(cY);

            adm_variables args = admf_at(grid_position, dim.get(), in);

            auto Yij = cY / (W*W);
            pin(Yij);

            valuef length_sq = dot(cvelocity, Yij.lower(cvelocity));
            valuef length = sqrt(fabs(length_sq));

            cvelocity = cvelocity / length;

            pin(cvelocity);

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
                        kjvk += args.Kij[j, k] * cvelocity[k];
                    }

                    valuef christoffel_sum = 0;

                    for(int k=0; k < 3; k++)
                    {
                        christoffel_sum += christoff2[i, j, k] * cvelocity[k];
                    }

                    valuef dlog_gA = dgA[j] / args.gA;

                    d_V[i] += args.gA * cvelocity[j] * (cvelocity[i] * (dlog_gA - kjvk) + 2 * raise_index(args.Kij, iYij, 0)[i, j] - christoffel_sum)
                            - iYij[i, j] * dgA[j] - cvelocity[j] * dgB[j, i];

                }
            }

            v3f d_X = args.gA * cvelocity - args.gB;
            pin(d_X);

            if_e(dot(d_X, d_X) < 0.2f * 0.2f, [&]
            {
                as_ref(result) = valuei(0);
                break_e();
            });

            valuef dt = -1.f * get_ct_timestep(cposition, cvelocity, W);

            as_ref(position) = cposition + d_X * dt;
            as_ref(velocity) = cvelocity + d_V * dt;
        });

        final_position = declare_e(position);
    }
    #endif

    v3f vel = declare_e(final_velocity);

    as_ref(positions[screen_position, screen_size]) = (v4f){0, final_position.x(), final_position.y(), final_position.z()};
    as_ref(velocities[screen_position, screen_size]) = (v4f){0, vel.x(), vel.y(), vel.z()};
    as_ref(results[screen_position, screen_size]) = as_constant(result);
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

    valuef to_upper = (valuef)centre_hi.x() / (valuef)centre_lo.x();

    v3f f_upper = (v3f)pos_lo * to_upper;

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

        value<uint64_t> lidx = p.z() * d.x() * d.y() + p.y() * d.x() + p.x() + slice.get() * d.x() * d.y() * d.z();

        as_ref(Guv[i][lidx]) = (block_precision_t)met[idx.x(), idx.y()];
    }
}

///todo: fixme
valuef acceleration_to_precision(v4f acceleration, valuef max_acceleration, valuef* next_ds_out)
{
    valuef current_acceleration_err = acceleration.length() * 0.01f;

    valuef experienced_acceleration_change = current_acceleration_err;

    valuef err = max_acceleration;

    //#define MIN_STEP 0.00001f
    //#define MIN_STEP 0.000001f

    valuef max_timestep = 100000;

    valuef diff = experienced_acceleration_change;

    diff = max(diff, err / pow(max_timestep, 2.f));

    ///of course, as is tradition, whatever works for kerr does not work for alcubierre
    ///the sqrt error calculation is significantly better for alcubierre, largely in terms of having no visual artifacts at all
    ///whereas the pow version is nearly 2x faster for kerr
    valuef next_ds = sqrt(err / diff);

    *next_ds_out = next_ds;

    return diff;
}

metric<valuef, 4, 4> get_Guv(v4i grid_pos, v3i dim, std::array<buffer<block_precision_t>, 10> Guv_buf, valuei last_slice)
{
    grid_pos.x() = clamp(grid_pos.x(), valuei(0), last_slice - 1);

    tensor<value<uint64_t>, 3> p = (tensor<value<uint64_t>, 3>)grid_pos.yzw();
    tensor<value<uint64_t>, 3> d = (tensor<value<uint64_t>, 3>)dim;

    ///this might be the problem?
    value<uint64_t> idx = ((value<uint64_t>)grid_pos.x()) * d.x() * d.y() * d.z() + p.z() * d.x() * d.y() + p.y() * d.x() + p.x();

    int indices[16] = {
        0, 1, 2, 3,
        1, 4, 5, 6,
        2, 5, 7, 8,
        3, 6, 8, 9,
    };

    metric<valuef, 4, 4> met;

    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            met[i, j] = (valuef)Guv_buf[indices[j * 4 + i]][idx];
        }
    }

    return met;
}

struct trace4_state
{

};

void trace4x4(execution_context& ectx, literal<valuei> screen_width, literal<valuei> screen_height,
            buffer_mut<v4f> positions, buffer_mut<v4f> velocities,
            buffer_mut<valuei> results, buffer_mut<valuef> zshift,
            literal<v3i> dim,
            literal<valuef> scale,
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

    if_e(y >= screen_height.get(), [&] {
        return_e();
    });

    if_e(x >= screen_width.get(), [&] {
        return_e();
    });

    v2i screen_position = {x, y};
    v2i screen_size = {screen_width.get(), screen_height.get()};

    v4f pos_in = declare_e(positions[screen_position, screen_size]);
    v4f vel_in = declare_e(velocities[screen_position, screen_size]);

    auto world_to_grid4 = [&](v4f position)
    {
        v3f grid_position = world_to_grid(position.yzw(), dim.get(), scale.get());
        valuef grid_t_frac = position.x() / last_time.get();
        valuef grid_t = grid_t_frac * (valuef)last_slice.get();

        pin(grid_t);

        /*if_e(last_slice.get() >= 3, [&]{
            grid_t = clamp()
        });*/

        grid_t = ternary(last_slice.get() > 5,
                         clamp(grid_t, valuef(2), (valuef)last_slice.get() - 3),
                         grid_t);

        grid_position = clamp(grid_position, (v3f){3,3,3}, (v3f)dim.get() - (v3f){4,4,4});
        pin(grid_position);

        v4f grid_fpos = (v4f){grid_t, grid_position.x(), grid_position.y(), grid_position.z()};
        return grid_fpos;
    };

    auto get_dX = [&](v4f position, v4f velocity, trace4_state st)
    {
        return velocity;
    };

    auto get_dV = [&](v4f position, v4f velocity, trace4_state st)
    {
        v4f grid_fpos = world_to_grid4(position);

        auto get_Guvb = [&](v4i pos)
        {
            auto v = get_Guv(pos, dim.get(), Guv_buf, last_slice.get());
            pin(v);
            return v;
        };

        auto get_guv_at = [&](v4f fpos)
        {
            //v4f p = {fpos.y(), fpos.z(), fpos.w(), fpos.x()};

            return function_quadlinear(get_Guvb, fpos);
        };

        auto Guv = get_guv_at(grid_fpos);

        tensor<valuef, 4, 4, 4> dGuv;

        for(int m=0; m < 4; m++)
        {
            v4f dir;
            dir[m] = 1;

            ///oh. The timelike direction obviously doesn't have a gap of scale
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

    auto get_dS = [&](v4f position, v4f velocity, v4f acceleration, trace4_state st)
    {
        //valuef pos = position.yzw().length();
        //return ternary(pos < valuef(10), valuef(1.f), valuef(1.5f));

        valuef dt;
        acceleration_to_precision(acceleration, 0.0001f, &dt);
        return dt;
    };

    auto get_state = [](v4f position)
    {
        return trace4_state();
    };

    verlet_context<valuef, 4> ctx;
    ctx.start(pos_in, vel_in, get_dX, get_dV, get_dS, get_state);

    mut<valuei> result = declare_mut_e(valuei(0));
    v4f final_position;
    v4f final_velocity;

    ///todo: I think what's happening is that the clamping is breaking my time derivatives
    ///which means we need to change the initial conditions to construct our rays from an earlier point, rather than from the end?
    {
        mut<valuei> idx = declare_mut_e("i", valuei(0));

        for_e(idx < 1024, assign_b(idx, idx + 1), [&]
        {
            #if 0
            tensor<m44f, 3, 3, 3, 3> block;

            for(int t=0; t < 3; t++)
            {
                for(int k=0; k < 3; k++)
                {
                    for(int j=0; j < 3; j++)
                    {
                        for(int i = 0; i < 3; i++)
                        {
                            v4i off = {t-1, i-1, j-1, k-1};

                            //valuei slice_up = clamp(time_floor + t - 1, valuei(0), last_slice.get() - 1);

                            block[t, i, j, k] = get_Guv((v4i)grid_fpos + off);
                            //pin(block[t, i, j, k]);
                        }
                    }
                }
            }

            auto funcl = [&]<typename T>(T&& func, v4f frac)
            {
                auto a000 = func({0,0,0,0});
                auto a100 = func({1,0,0,0});

                auto a010 = func({0,1,0,0});
                auto a110 = func({1,1,0,0});

                auto a001 = func({0,0,1,0});
                auto a101 = func({1,0,1,0});

                auto a011 = func({0,1,1,0});
                auto a111 = func({1,1,1,0});

                auto a00 = a000 - frac.x() * (a000 - a100);
                auto a01 = a001 - frac.x() * (a001 - a101);

                auto a10 = a010 - frac.x() * (a010 - a110);
                auto a11 = a011 - frac.x() * (a011 - a111);

                auto a0 = a00 - frac.y() * (a00 - a10);
                auto a1 = a01 - frac.y() * (a01 - a11);

                auto linear_1 = a0 - frac.z() * (a0 - a1);

                auto c000 = func({0,0,0,1});
                auto c100 = func({1,0,0,1});

                auto c010 = func({0,1,0,1});
                auto c110 = func({1,1,0,1});

                auto c001 = func({0,0,1,1});
                auto c101 = func({1,0,1,1});

                auto c011 = func({0,1,1,1});
                auto c111 = func({1,1,1,1});

                auto c00 = c000 - frac.x() * (c000 - c100);
                auto c01 = c001 - frac.x() * (c001 - c101);

                auto c10 = c010 - frac.x() * (c010 - c110);
                auto c11 = c011 - frac.x() * (c011 - c111);

                auto c0 = c00 - frac.y() * (c00 - c10);
                auto c1 = c01 - frac.y() * (c01 - c11);

                auto linear_2 = c0 - frac.z() * (c0 - c1);

                return linear_1 - frac.w() * (linear_1 - linear_2);
            };

            m44f Guv = funcl([&](tensor<int, 4> pos)
            {
                tensor<int, 4> ipos = pos + (tensor<int, 4>){1, 1, 1, 1};

                return block[ipos.x(), ipos.y(), ipos.z(), ipos.w()];
            }, grid_fpos - floor(grid_fpos));

            //m44f Guv = block[1, 1, 1, 1];

            tensor<valuef, 4, 4, 4> dGuv;

            for(int m=0; m < 4; m++)
            {
                tensor<int, 4> dir;
                dir[m] = 1;

                tensor<int, 4> centre = {1,1,1,1};

                tensor<int, 4> r = centre + dir;
                tensor<int, 4> l = centre - dir;

                metric<valuef, 4, 4> right = block[r.x(), r.y(), r.z(), r.w()];
                metric<valuef, 4, 4> left = block[l.x(), l.y(), l.z(), l.w()];

                valuef divisor = 2 * scale.get();

                if(m == 0)
                    divisor = 2 * slice_width.get();

                /*auto dr = (right - Guv) / scale.get();
                auto dl = (Guv - left) / scale.get();

                auto dG = mix(dl, dr, grid_fpos[m] - floor(grid_fpos[m]));*/

                auto dG = (right - left) / divisor;

                for(int i=0; i < 4; i++)
                {
                    for(int j=0; j < 4; j++)
                    {
                        dGuv[m, i, j] = dG[i, j];
                    }
                }
            }
            #endif

            ctx.next(get_dX, get_dV, get_dS, get_state);

            v4f cposition = as_constant(ctx.position);
            v4f cvelocity = as_constant(ctx.velocity);

            valuef radius_sq = dot(cposition.yzw(), cposition.yzw());

            if_e(radius_sq > UNIVERSE_SIZE*UNIVERSE_SIZE, [&] {
                //ray escaped
                as_ref(result) = valuei(1);
                break_e();
            });

            /*if_e(x == screen_width.get()/2 && y == screen_height.get()/2, [&]{
                value_base se;
                se.type = value_impl::op::SIDE_EFFECT;
                se.abstract_value = "printf(\"pos vel x: %f %f\\n\"," + value_to_string(cposition.x()) + "," + value_to_string(cvelocity.x()) + ")";
                //se.abstract_value = "printf(\"val %f\\n\"," + value_to_string(d_V[0]) + ")";
                //se.abstract_value = "printf(\"adm: %i\\n\"," + value_to_string(result) + ")";

                value_impl::get_context().add(se);
            });*/

            /*if_e(as_constant(ctx.velocity).yzw().length() < 0.5f, [&]{
                as_ref(result) = valuei(1);
                break_e();
            });*/

            if_e(cposition.x() < -150 || fabs(cvelocity.x()) > 30 || cvelocity.yzw().squared_length() < 0.1f * 0.1f ||
                 !isfinite(cposition.x()) || !isfinite(cposition.y()) || !isfinite(cposition.z()) || !isfinite(cposition.w()) ||
                 !isfinite(cvelocity.x()) || !isfinite(cvelocity.y()) || !isfinite(cvelocity.z()) || !isfinite(cvelocity.w())
                 , [&]{
                as_ref(result) = valuei(0);
                break_e();
            });

            /*if_e(dot(accel, accel) < 0.2f * 0.2f, [&]
            {
                //as_ref(result) = valuei(1);
                //break_e();
            });*/
        });

        final_position = declare_e(ctx.position);
        final_velocity = declare_e(ctx.velocity);
    }

    auto get_Guvb = [&](v4i pos)
    {
        auto val = get_Guv(pos, dim.get(), Guv_buf, last_slice.get());
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

void calculate_texture_coordinates(execution_context& ectx, literal<valuei> screen_width, literal<valuei> screen_height,
                                   buffer<v4f> positions, buffer<v4f> velocities,
                                   buffer_mut<v2f> out)
{
    using namespace single_source;

    valuei x = value_impl::get_global_id(0);
    valuei y = value_impl::get_global_id(1);

    pin(x);
    pin(y);

    if_e(y >= screen_height.get(), [&] {
        return_e();
    });

    if_e(x >= screen_width.get(), [&] {
        return_e();
    });

    v2i screen_position = {x, y};
    v2i screen_size = {screen_width.get(), screen_height.get()};

    v3f position3 = positions[screen_position, screen_size].yzw();
    v3f velocity3 = velocities[screen_position, screen_size].yzw();

    position3 = fix_ray_position_cart(position3, velocity3, UNIVERSE_SIZE);

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

void render(execution_context& ectx, literal<valuei> screen_width, literal<valuei> screen_height,
            buffer<v4f> positions, buffer<v4f> velocities, buffer<valuei> results, buffer<valuef> zshift,
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

    if_e(y >= screen_height.get(), [&] {
        return_e();
    });

    if_e(x >= screen_width.get(), [&] {
        return_e();
    });

    v2i screen_position = {x, y};
    v2i screen_size = {screen_width.get(), screen_height.get()};

    if_e(results[screen_position, screen_size] == 0, [&]{
        screen.write(ectx, {x, y}, (v4f){0,0,0,1});
        return_e();
    });

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

    valuef zp1 = declare_e(zshift[screen_position, screen_size]) + 1;

    cvt = linear_to_srgb_gpu(do_redshift(cvt, zp1));

    v4f crgba = {cvt[0], cvt[1], cvt[2], 1.f};

    screen.write(ectx, {x, y}, crgba);
}

void build_raytrace_kernels(cl::context ctx)
{
    cl::async_build_and_cache(ctx, []{
        auto get_metric = [](v4f position, bssn_args_mem<buffer<valuef>> in, literal<v3i> dim, literal<valuef> scale)
        {
            using namespace single_source;

            v3f grid = world_to_grid(position.yzw(), dim.get(), scale.get());

            grid = clamp(grid, (v3f){2,2,2}, (v3f)dim.get() - (v3f){3,3,3});

            pin(grid);

            adm_variables adm = admf_at(grid, dim.get(), in);

            auto met = calculate_real_metric(adm.Yij, adm.gA, adm.gB);
            pin(met);
            return met;
        };

        return value_impl::make_function(build_initial_tetrads<get_metric, bssn_args_mem<buffer<valuef>>, literal<v3i>, literal<valuef>>, "init_tetrads3");
    }, {"init_tetrads3"});

    cl::async_build_and_cache(ctx, []{
        auto get_metric = [](v4f position, std::array<buffer<block_precision_t>, 10> in, literal<v3i> dim, literal<valuef> scale, literal<valuef> slice_time_end, literal<valuei> last_slice)
        {
            using namespace single_source;

            v3f grid = world_to_grid(position.yzw(), dim.get(), scale.get());

            grid = clamp(grid, (v3f){2,2,2}, (v3f)dim.get() - (v3f){3,3,3});

            auto guv = [&](v4i pos)
            {
                return get_Guv(pos, dim.get(), in, last_slice.get());
            };

            valuef slice = (position.x() / (valuef)slice_time_end.get()) * (valuef)last_slice.get();

            slice = clamp(slice, valuef(0.f), (valuef)last_slice.get() - 1);

            v4f fpos = {slice, grid.x(), grid.y(), grid.z()};

            pin(fpos);

            auto met = function_quadlinear(guv, fpos);
            pin(met);
            return met;
        };

        return value_impl::make_function(build_initial_tetrads<get_metric, std::array<buffer<block_precision_t>, 10>, literal<v3i>, literal<valuef>, literal<valuef>, literal<valuei>>, "init_tetrads4");
    }, {"init_tetrads4"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(trace3, "trace3");
    }, {"trace3"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(init_rays3, "init_rays3");
    }, {"init_rays3"});

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

#include "raytrace.hpp"
#include "tensor_algebra.hpp"

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
metric<valuef, 3, 3> Yij_f_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    auto func = [&](v3i pos)
    {
        return adm_at(pos, dim, in).Yij;
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

inline
inverse_metric<valuef, 3, 3> icY_f_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    auto func = [&](v3i pos)
    {
        return bssn_at(pos, dim, in).cY;
    };

    auto val = function_trilinear(func, pos);
    pin(val);
    return val.invert();
}

inline
inverse_metric<valuef, 3, 3> iYij_f_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    auto W = W_f_at(pos, dim, in);

    pin(W);

    return icY_f_at(pos, dim, in) * W * W;
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

    ///need to differentiate some of these variables in the regular adm stuff sighghghgh
    v3f ray_direction = get_ray_through_pixel(screen_position, screen_size, 90, camera_quat.get());

    geodesic my_geodesic = make_lightlike_geodesic(position[0], ray_direction, tetrads);

    adm_variables init_adm = admf_at(grid_position, dim.get(), in);

    tensor<valuef, 4> normal = get_adm_hypersurface_normal_raised(init_adm.gA, init_adm.gB);
    tensor<valuef, 4> normal_lowered = get_adm_hypersurface_normal_lowered(init_adm.gA);

    valuef E = -sum_multiply(my_geodesic.velocity, normal_lowered);

    ///98% sure this is wrong, but past me had a lot of qualms about this and was careful so...
    tensor<valuef, 4> adm_velocity = -((my_geodesic.velocity / E) - normal);

    /*if_e(x == 0 && y == 0, [&]{
        value_base se;
        se.type = value_impl::op::SIDE_EFFECT;
        se.abstract_value = "printf(\"adm: %f\\n\"," + value_to_string(adm_velocity.x()) + ")";

        value_impl::get_context().add(se);
    });*/

    v2i out_dim = {screen_width.get(), screen_height.get()};
    v2i out_pos = {x, y};

    as_ref(positions_out[out_pos, out_dim]) = my_geodesic.position;

    if_e(is_adm.get() == 1, [&]{
        as_ref(velocities_out[out_pos, out_dim]) = adm_velocity;
    });

    if_e(is_adm.get() == 0, [&]{
        as_ref(velocities_out[out_pos, out_dim]) = my_geodesic.velocity;
    });
}

///todo: figure out the projection
void trace3(execution_context& ectx, literal<valuei> screen_width, literal<valuei> screen_height,
                     read_only_image<2> background, write_only_image<2> screen,
                     literal<valuei> background_width, literal<valuei> background_height,
                     literal<v4f> camera_quat,
                     buffer<v4f> positions, buffer<v4f> velocities,
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
    v2i background_size = {background_width.get(), background_height.get()};

    v3f pos_in = positions[screen_position, screen_size].yzw();
    v3f vel_in = velocities[screen_position, screen_size].yzw();

    mut<valuei> result = declare_mut_e(valuei(1));
    v3f final_position;

    {
        mut_v3f position = declare_mut_e(pos_in);
        mut_v3f velocity = declare_mut_e(vel_in);

        mut<valuei> idx = declare_mut_e("i", valuei(0));

        for_e(idx < 1024, assign_b(idx, idx + 1), [&]
        {
            v3f cposition = declare_e(position);
            v3f cvelocity = declare_e(velocity);

            v3f grid_position = world_to_grid(cposition, dim.get(), scale.get());

            grid_position = clamp(grid_position, (v3f){3,3,3}, (v3f)dim.get() - (v3f){4,4,4});
            pin(grid_position);

            #if 1
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

            auto Yij = cY / (W*W);
            pin(Yij);

            adm_variables args = admf_at(grid_position, dim.get(), in);

            valuef length_sq = dot(cvelocity, Yij.lower(cvelocity));
            valuef length = sqrt(fabs(length_sq));

            cvelocity = cvelocity / length;

            pin(cvelocity);

            v3f d_X = args.gA * cvelocity - args.gB;

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

            ///todo: fix/checkme
            ///98% sure this is wrong, because ray back in time dt/dT divided gives flipped ray dir
            ///but then. Rsy going wrong way in coordinate time. bad?
            valuef dt = 1.f * get_ct_timestep(cposition, cvelocity, W);
            #endif

            #if 0
            auto metric_at = [&](v3i pos)
            {
                adm_variables adm = adm_at(pos, dim.get(), in);

                return calculate_real_metric(adm.Yij, adm.gA, adm.gB);
            };

            auto metric_f_at = [&](v3f pos)
            {
                return function_trilinear(metric_at, pos);
            };

            auto Guv = metric_f_at(grid_position);
            tensor<valuef, 4, 4, 4> dGuv;

            for(int m=0; m < 4; m++)
            {
                if(m == 0)
                {
                    for(int j=0; j < 4; j++)
                    {
                        for(int k=0; k < 4; k++)
                            dGuv[m, j, k] = 0;
                    }
                }
                else
                {
                    v3i dir;
                    dir[m - 1] = 1;

                    auto diff = (metric_f_at(grid_position + dir) - metric_f_at(grid_position - dir)) / (2 * scale);

                    for(int j=0; j < 4; j++)
                    {
                        for(int k=0; k < 4; k++)
                        {
                            dGuv[m, j, k] = diff[j, k];
                        }
                    }
                }
            }

            valuef dt = 0.1f;
            #endif

            as_ref(position) = cposition + d_X * dt;
            as_ref(velocity) = cvelocity + d_V * dt;

            valuef radius_sq = dot(cposition, cposition);

            if_e(radius_sq > 29*29, [&] {
                //ray escaped
                as_ref(result) = valuei(0);
                break_e();
            });

            if_e(dot(d_X, d_X) < 0.2f * 0.2f, [&]
            {
                as_ref(result) = valuei(1);
                break_e();
            });
        });

        final_position = declare_e(position);
    }

    mut_v3f col = declare_mut_e((v3f){0,0,0});

    if_e(result == 0, [&] {
        v3f spherical = cartesian_to_spherical(final_position);

        valuef theta = spherical[1];
        valuef phi = spherical[2];

        v2f texture_coordinate = angle_to_tex({theta, phi});

        valuei tx = (texture_coordinate[0] * background_size.x().to<float>() + background_size.x().to<float>()).to<int>() % background_size.x();
        valuei ty = (texture_coordinate[1] * background_size.y().to<float>() + background_size.y().to<float>()).to<int>() % background_size.y();

        ///linear colour
        as_ref(col) = linear_to_srgb_gpu(background.read<float, 4>({tx, ty}).xyz());
    });

    v4f crgba = {col[0], col[1], col[2], 1.f};

    screen.write(ectx, {x, y}, crgba);
}

#if 0
void trace4(execution_context& ectx, literal<valuei> screen_width, literal<valuei> screen_height,
            read_only_image<2> background, write_only_image<2> screen,
            literal<valuei> background_width, literal<valuei> background_height,
            literal<v4f> camera_quat,
            buffer_mut<v4f> positions, buffer_mut<v3f> velocities,
            literal<v3i> dim,
            literal<valuef> scale,
            literal<valuef> time_lower,
            literal<valuef> time_upper,
            bssn_args_mem<buffer<valuef>> lower,
            bssn_args_mem<buffer<valuef>> upper,
            bssn_derivatives_mem<buffer<derivative_t>> lower_derivatives,
            bssn_derivatives_mem<buffer<derivative_t>> upper_derivatives,
            buffer_mut<valuei> full_result)
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
    v2i background_size = {background_width.get(), background_height.get()};

    if_e(full_result[screen_position, screen_size] != 0, [&]{
        return_e();
    });

    valuef position_t_in = positions[screen_position, screen_size].x();
    v3f pos_in = declare_e(positions[screen_position, screen_size].yzw());
    v3f vel_in = declare_e(velocities[screen_position, screen_size]);

    mut<valuei> result = declare_mut_e(valuei(1));

    mut_v3f position = declare_mut_e(pos_in);
    mut_v3f velocity = declare_mut_e(vel_in);
    mut<valuef> position_t = declare_mut_e(position_t_in);

    {
        mut<valuei> idx = declare_mut_e("i", valuei(0));

        for_e(idx < 1024, assign_b(idx, idx + 1), [&]
        {
            v3f cposition = declare_e(position);
            v3f cvelocity = declare_e(velocity);
            valuef cposition_t = declare_e(position_t);

            v3f grid_position = world_to_grid(cposition, dim.get(), scale.get());
            valuef time_frac = (cposition_t - time_lower.get()) / (time_upper.get() - time_lower.get());

            time_frac = min(time_frac, valuef(1.f));

            /*if_e(time_frac < 0, [&]{
                 as_ref(result) = valuei(2);
                break_e();
            });*/

            time_frac = clamp(time_frac, 0.f, 1.f);

            time_frac = 1;

            grid_position = clamp(grid_position, (v3f){3,3,3}, (v3f)dim.get() - (v3f){4,4,4});
            pin(grid_position);

            auto dgA_at1 = [&](v3i pos)
            {
                bssn_derivatives derivs(pos, dim.get(), lower_derivatives);
                return derivs.dgA;
            };

            auto dgB_at1 = [&](v3i pos)
            {
                bssn_derivatives derivs(pos, dim.get(), lower_derivatives);
                return derivs.dgB;
            };

            auto dcY_at1 = [&](v3i pos)
            {
                bssn_derivatives derivs(pos, dim.get(), lower_derivatives);
                return derivs.dcY;
            };

            auto dW_at1 = [&](v3i pos)
            {
                bssn_derivatives derivs(pos, dim.get(), lower_derivatives);
                return derivs.dW;
            };

            tensor<valuef, 3> dgA1 = function_trilinear(dgA_at1, grid_position);
            tensor<valuef, 3, 3> dgB1 = function_trilinear(dgB_at1, grid_position);
            tensor<valuef, 3, 3, 3> dcY1 = function_trilinear(dcY_at1, grid_position);
            tensor<valuef, 3> dW1 = function_trilinear(dW_at1, grid_position);

            auto dgA_at2 = [&](v3i pos)
            {
                bssn_derivatives derivs(pos, dim.get(), upper_derivatives);
                return derivs.dgA;
            };

            auto dgB_at2 = [&](v3i pos)
            {
                bssn_derivatives derivs(pos, dim.get(), upper_derivatives);
                return derivs.dgB;
            };

            auto dcY_at2 = [&](v3i pos)
            {
                bssn_derivatives derivs(pos, dim.get(), upper_derivatives);
                return derivs.dcY;
            };

            auto dW_at2 = [&](v3i pos)
            {
                bssn_derivatives derivs(pos, dim.get(), upper_derivatives);
                return derivs.dW;
            };

            tensor<valuef, 3> dgA2 = function_trilinear(dgA_at2, grid_position);
            tensor<valuef, 3, 3> dgB2 = function_trilinear(dgB_at2, grid_position);
            tensor<valuef, 3, 3, 3> dcY2 = function_trilinear(dcY_at2, grid_position);
            tensor<valuef, 3> dW2 = function_trilinear(dW_at2, grid_position);

            auto dgA = mix(dgA1, dgA2, time_frac);
            auto dgB = mix(dgB1, dgB2, time_frac);
            auto dcY = mix(dcY1, dcY2, time_frac);
            auto dW = mix(dW1, dW2, time_frac);

            pin(dgA);
            pin(dgB);
            pin(dcY);
            pin(dW);

            auto W1 = W_f_at(grid_position, dim.get(), lower);
            auto cY1 = cY_f_at(grid_position, dim.get(), lower);

            auto W2 = W_f_at(grid_position, dim.get(), upper);
            auto cY2 = cY_f_at(grid_position, dim.get(), upper);

            auto W = mix(W1, W2, time_frac);
            auto cY = mix(cY1, cY2, time_frac);

            pin(W);
            pin(cY);

            auto Yij = cY / (W*W);
            pin(Yij);

            adm_variables args1 = admf_at(grid_position, dim.get(), lower);
            adm_variables args2 = admf_at(grid_position, dim.get(), upper);

            adm_variables args;
            args.gA = mix(args1.gA, args2.gA, time_frac);
            args.gB = mix(args1.gB, args2.gB, time_frac);
            //args.Yij = mix(args1.Yij, args2.Yij, time_frac);
            args.Kij = mix(args1.Kij, args2.Kij, time_frac);

            valuef length_sq = dot(cvelocity, Yij.lower(cvelocity));
            valuef length = sqrt(fabs(length_sq));

            //cvelocity = cvelocity / length;

            pin(cvelocity);

            v3f d_X = args.gA * cvelocity - args.gB;

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

            ///todo: fix/checkme
            ///98% sure this is wrong, because ray back in time dt/dT divided gives flipped ray dir
            ///but then. Rsy going wrong way in coordinate time. bad?
            valuef dt = 1.f * get_ct_timestep(cposition, cvelocity, W);

            if_e(x == 400 && y == 400, [&]{
                value_base se;
                se.type = value_impl::op::SIDE_EFFECT;
                se.abstract_value = "printf(\"test: %f %f %f\\n\"," + value_to_string(vel_in.x()) + "," + value_to_string(vel_in.y()) + "," + value_to_string(vel_in.z()) + ")";
                //se.abstract_value = "printf(\"val %f\\n\"," + value_to_string(d_V[0]) + ")";
                //se.abstract_value = "printf(\"adm: %i\\n\"," + value_to_string(result) + ")";

                value_impl::get_context().add(se);
            });

            as_ref(position) = cposition + d_X * dt;
            as_ref(velocity) = cvelocity + d_V * dt;
            as_ref(position_t) = cposition_t - fabs(dt);

            valuef radius_sq = dot(cposition, cposition);

            if_e(radius_sq > 29*29, [&] {
                //ray escaped
                as_ref(result) = valuei(0);
                break_e();
            });

            if_e(dot(d_X, d_X) < 0.2f * 0.2f, [&]
            {
                as_ref(result) = valuei(1);
                break_e();
            });
        });
    }

    v3f final_position = declare_e(position);
    valuef final_time = declare_e(position_t);

    as_ref(positions[screen_position, screen_size]) = (v4f){final_time, final_position.x(), final_position.y(), final_position.z()};
    as_ref(velocities[screen_position, screen_size]) = velocity;

    mut_v3f col = declare_mut_e((v3f){0,0,0});

    if_e(result == 0, [&] {
        v3f spherical = cartesian_to_spherical(final_position);

        valuef theta = spherical[1];
        valuef phi = spherical[2];

        v2f texture_coordinate = angle_to_tex({theta, phi});

        valuei tx = (texture_coordinate[0] * background_size.x().to<float>() + background_size.x().to<float>()).to<int>() % background_size.x();
        valuei ty = (texture_coordinate[1] * background_size.y().to<float>() + background_size.y().to<float>()).to<int>() % background_size.y();

        ///linear colour
        as_ref(col) = linear_to_srgb_gpu(background.read<float, 4>({tx, ty}).xyz());
    });

    if_e(result == 1 || result == 0, [&] {
        as_ref(full_result[screen_position, screen_size]) = valuei(1);

        v4f crgba = {col[0], col[1], col[2], 1.f};

        screen.write(ectx, {x, y}, crgba);
    });

    if_e(x == 400 && y == 400, [&]{
        value_base se;
        se.type = value_impl::op::SIDE_EFFECT;
        se.abstract_value = "printf(\"pos: %f %f %f\\n\"," + value_to_string(final_position.x()) + "," + value_to_string(final_position.y()) + "," + value_to_string(final_position.z()) + ")";
        //se.abstract_value = "printf(\"adm: %i\\n\"," + value_to_string(result) + ")";

        value_impl::get_context().add(se);
    });
}
#endif
void bssn_to_guv(execution_context& ectx, literal<v3i> dim, literal<valuef> scale,
                 bssn_args_mem<buffer<valuef>> in,
                 std::array<buffer_mut<valueh>, 10> Guv, literal<value<uint64_t>> offset)
{
    using namespace single_source;

    valuei x = value_impl::get_global_id(0);
    valuei y = value_impl::get_global_id(1);
    valuei z = value_impl::get_global_id(2);

    pin(x);
    pin(y);
    pin(z);

    if_e(x >= dim.get().x() || y >= dim.get().y() || z >= dim.get().z(), [&]{
        return_e();
    });

    v3i pos = {x, y, z};

    bssn_args args(pos, dim.get(), in);
    adm_variables adm = bssn_to_adm(args);

    metric<valuef, 4, 4> met = calculate_real_metric(adm.Yij, adm.gA, adm.gB);

    vec2i indices[10] = {
        {0, 0}, {1, 0}, {2, 0}, {3, 0},
        {1, 1}, {2, 1}, {3, 1},
        {2, 2}, {3, 2},
        {3, 3},
    };

    tensor<value<uint64_t>, 3> p = (tensor<value<uint64_t>, 3>)pos;
    tensor<value<uint64_t>, 3> d = (tensor<value<uint64_t>, 3>)dim.get();

    for(int i=0; i < 10; i++)
    {
        vec2i idx = indices[i];

        value<uint64_t> lidx = p.z() * d.x() * d.y() + p.y() * d.x() + p.x() + offset.get();

        //std::cout << "Vidx " << value_to_string(lidx) << std::endl;

        as_ref(Guv[i][lidx]) = (valueh)met[idx.x(), idx.y()];
    }
}

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

valuef get_ct_timestep2(valuef W)
{
    float X_far = 0.9f;
    float X_near = 0.6f;

    valuef X = W*W;

    valuef my_fraction = (clamp(X, X_near, X_far) - X_near) / (X_far - X_near);

    my_fraction = clamp(my_fraction, 0.f, 1.f);

    return mix(valuef(1.f), valuef(4.f), my_fraction);
}

///tomorrow me: try just 4x4ing one slice
void trace4x4(execution_context& ectx, literal<valuei> screen_width, literal<valuei> screen_height,
            read_only_image<2> background, write_only_image<2> screen,
            literal<valuei> background_width, literal<valuei> background_height,
            buffer_mut<v4f> positions, buffer_mut<v4f> velocities,
            literal<v3i> dim,
            literal<valuef> scale,
            std::array<buffer<valueh>, 10> Guv_buf,
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
    v2i background_size = {background_width.get(), background_height.get()};

    v4f pos_in = declare_e(positions[screen_position, screen_size]);
    v4f vel_in = declare_e(velocities[screen_position, screen_size]);

    mut<valuei> result = declare_mut_e(valuei(1));
    v4f final_position;

    ///todo: I think what's happening is that the clamping is breaking my time derivatives
    ///which means we need to change the initial conditions to construct our rays from an earlier point, rather than from the end?
    {
        mut_v4f position = declare_mut_e(pos_in);
        mut_v4f velocity = declare_mut_e(vel_in);

        mut<valuei> idx = declare_mut_e("i", valuei(0));

        for_e(idx < 1024, assign_b(idx, idx + 1), [&]
        {
            v4f cposition = declare_e(position);
            v4f cvelocity = declare_e(velocity);

            v3f grid_position = world_to_grid(cposition.yzw(), dim.get(), scale.get());
            valuef grid_t_frac = cposition.x() / last_time.get();
            valuef grid_t = grid_t_frac * (valuef)last_slice.get();

            grid_position = clamp(grid_position, (v3f){3,3,3}, (v3f)dim.get() - (v3f){4,4,4});
            pin(grid_position);

            v4f grid_fpos = (v4f){grid_t, grid_position.x(), grid_position.y(), grid_position.z()};

            auto get_Guv = [&](v4i pos)
            {
                //pos = {pos.w(), pos.x(), pos.y(), pos.z()};

                pos.x() = clamp(pos.x(), valuei(0), last_slice.get() - 1);

                tensor<value<uint64_t>, 3> p = (tensor<value<uint64_t>, 3>)pos.yzw();
                tensor<value<uint64_t>, 3> d = (tensor<value<uint64_t>, 3>)dim.get();

                ///this might be the problem?
                value<uint64_t> idx = ((value<uint64_t>)pos.x()) * d.x() * d.y() * d.z() + p.z() * d.x() * d.y() + p.y() * d.x() + p.x();

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
            };

            auto get_guv_at = [&](v4f fpos)
            {
                //v4f p = {fpos.y(), fpos.z(), fpos.w(), fpos.x()};

                return function_quadlinear(get_Guv, fpos);
            };

            #if 1
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

                auto val = (get_guv_at(grid_fpos + dir) - get_guv_at(grid_fpos - dir)) / divisor;

                for(int i=0; i < 4; i++)
                {
                    for(int j=0; j < 4; j++)
                    {
                        dGuv[m, i, j] = val[i, j];
                    }
                }
            }
            #endif

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

            pin(Guv);
            pin(dGuv);

            auto christoff2 = christoffel_symbols_2(Guv.invert(), dGuv);

            pin(christoff2);

            v4f accel;

            for(int uu=0; uu < 4; uu++)
            {
                valuef sum = 0;

                for(int aa = 0; aa < 4; aa++)
                {
                    for(int bb = 0; bb < 4; bb++)
                    {
                        sum += cvelocity[aa] * cvelocity[bb] * christoff2[uu, aa, bb];
                    }
                }

                accel[uu] = -sum;
            }

            /*metric<valuef, 3, 3> Yij;

            for(int i=1; i < 4; i++)
            {
                for(int j=1; j < 4; j++)
                {
                    Yij[i - 1, j - 1] = Guv[i, j];
                }
            }

            valuef W = pow(Yij.det(), -1/6.f);*/

            valuef dt = 1.f;// * get_ct_timestep2(W);

            //valuef dt = 0.01f/accel.length();

            as_ref(position) = cposition + cvelocity * dt;
            as_ref(velocity) = cvelocity + accel * dt;

            valuef radius_sq = dot(cposition.yzw(), cposition.yzw());

            if_e(radius_sq > 29*29, [&] {
                //ray escaped
                as_ref(result) = valuei(0);
                break_e();
            });

            if_e(dot(accel, accel) < 0.2f * 0.2f, [&]
            {
                //as_ref(result) = valuei(1);
                //break_e();
            });
        });

        final_position = declare_e(position);
    }

    mut_v3f col = declare_mut_e((v3f){0,0,0});

    if_e(result == 0, [&] {
        v3f spherical = cartesian_to_spherical(final_position.yzw());

        valuef theta = spherical[1];
        valuef phi = spherical[2];

        v2f texture_coordinate = angle_to_tex({theta, phi});

        valuei tx = (texture_coordinate[0] * background_size.x().to<float>() + background_size.x().to<float>()).to<int>() % background_size.x();
        valuei ty = (texture_coordinate[1] * background_size.y().to<float>() + background_size.y().to<float>()).to<int>() % background_size.y();

        ///linear colour
        as_ref(col) = linear_to_srgb_gpu(background.read<float, 4>({tx, ty}).xyz());
    });

    if_e(result == 1 || result == 0, [&] {
        v4f crgba = {col[0], col[1], col[2], 1.f};

        screen.write(ectx, {x, y}, crgba);
    });
}

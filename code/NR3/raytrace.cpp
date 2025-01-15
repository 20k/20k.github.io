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
    pos = clamp(pos, (v3i){1,1,1}, dim - (v3i){2,2,2});

    bssn_args args(pos, dim, in);
    return args;
}

inline
valuef W_f_at(v3f pos, v3i dim, bssn_args_mem<buffer<valuef>> in)
{
    using namespace single_source;

    auto W_at = [&](v3i pos)
    {
        pos = clamp(pos, (v3i){1,1,1}, dim - (v3i){2,2,2});

        bssn_args args(pos, dim, in);
        pin(args.W);
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
        auto val = adm_at(pos, dim, in).gA;
        pin(val);
        return val;
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
        auto val = adm_at(pos, dim, in).gB;
        pin(val);
        return val;
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
        auto val = adm_at(pos, dim, in).Yij;
        pin(val);
        return val;
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
        auto val = bssn_at(pos, dim, in).cY;
        pin(val);
        return val;
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
        auto val = bssn_at(pos, dim, in).cY;
        pin(val);
        return val;
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

    valuef X = sqrt(W);

    valuef my_fraction = (clamp(X, X_near, X_far) - X_near) / (X_far - X_near);

    my_fraction = clamp(my_fraction, 0.f, 1.f);

    return mix(valuef(0.4f), valuef(4.f), my_fraction) * 0.1f;
}

void trace3(execution_context& ectx, literal<valuei> screen_width, literal<valuei> screen_height,
                     read_only_image<2> background, write_only_image<2> screen,
                     literal<valuei> background_width, literal<valuei> background_height,
                     buffer<v4f> e0, buffer<v4f> e1, buffer<v4f> e2, buffer<v4f> e3,
                     buffer<v4f> position, literal<v4f> camera_quat,
                     literal<v3i> dim,
                     literal<valuef> scale,
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
    v2i background_size = {background_width.get(), background_height.get()};

    tetrad tetrads = {e0[0], e1[0], e2[0], e3[0]};

    v4f lpos = position[0];

    pin(lpos);

    ///need to differentiate some of these variables in the regular adm stuff sighghghgh
    v3i ipos = (v3i)lpos.yzw();

    v3f ray_direction = get_ray_through_pixel(screen_position, screen_size, 90, camera_quat.get());

    geodesic my_geodesic = make_lightlike_geodesic(position[0], ray_direction, tetrads);

    adm_variables init_adm = admf_at(position[0].yzw(), dim.get(), in);

    tensor<valuef, 4> normal = get_adm_hypersurface_normal_raised(init_adm.gA, init_adm.gB);

    metric<valuef, 4, 4> init_full = calculate_real_metric(init_adm.Yij, init_adm.gA, init_adm.gB);

    tensor<valuef, 4> velocity_lowered = init_full.lower(my_geodesic.velocity);

    valuef E = -sum_multiply(velocity_lowered, normal);

    tensor<valuef, 4> adm_velocity = -((my_geodesic.velocity / E) - normal);

    mut<valuei> result = declare_mut_e(valuei(1));
    v3f final_position;

    {
        mut_v3f position = declare_mut_e(my_geodesic.position.yzw());
        mut_v3f velocity = declare_mut_e(adm_velocity.yzw());

        mut<valuei> idx = declare_mut_e("i", valuei(0));

        for_e(idx < 1024, assign_b(idx, idx + 1), [&]
        {
            v3f cposition = declare_e(position);
            v3f cvelocity = declare_e(velocity);

            v3f grid_position = world_to_grid(cposition, dim.get(), scale.get());

            pin(grid_position);

            tensor<valuef, 3> dgA;
            tensor<valuef, 3, 3> dgB;
            tensor<valuef, 3, 3, 3> dcY;

            for(int m=0; m < 3; m++)
            {
                v3f dir = {0,0,0};
                dir[m] = 1;

                auto gA_r = gA_f_at(grid_position + dir, dim.get(), in);
                auto gA_l = gA_f_at(grid_position - dir, dim.get(), in);

                auto gB_r = gB_f_at(grid_position + dir, dim.get(), in);
                auto gB_l = gB_f_at(grid_position - dir, dim.get(), in);

                auto Yij_r = Yij_f_at(grid_position + dir, dim.get(), in);
                auto Yij_l = Yij_f_at(grid_position - dir, dim.get(), in);

                dgA[m] = (gA_r - gA_l) / (2 * scale.get());

                for(int j=0; j < 3; j++)
                {
                    dgB[m, j] = (gB_r[j] - gB_l[j]) / (2 * scale.get());

                    for(int k=0; k < 3; k++)
                    {
                        dcY[m, j, k] = (Yij_r[j, k] - Yij_l[j, k]) / (2 * scale.get());
                    }
                }
            }

            pin(dgA);
            pin(dgB);
            pin(dcY);

            adm_variables args = admf_at(grid_position, dim.get(), in);

            valuef length_sq = dot(cvelocity, args.Yij.lower(cvelocity));
            valuef length = sqrt(fabs(length_sq));

            cvelocity = cvelocity / length;

            pin(cvelocity);

            v3f d_X = args.gA * cvelocity - args.gB;

            auto iYij = iYij_f_at(grid_position, dim.get(), in);
            //auto iYij = args.Yij.invert();
            pin(iYij);

            auto christoff2 = christoffel_symbols_2(iYij, dcY);
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

            valuef fW = W_f_at(grid_position, dim.get(), in);

            valuef dt = -1.f * get_ct_timestep(cposition, cvelocity, fW);

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

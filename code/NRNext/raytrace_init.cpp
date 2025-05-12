#include "raytrace_init.hpp"
#include "bssn.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include "tensor_algebra.hpp"
#include "formalisms.hpp"
#include "interpolation.hpp"

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

tetrad calculate_tetrad(m44f metric, v3f local_velocity, bool should_orient)
{
    using namespace single_source;

    v4f v0 = {1, 0, 0, 0};
    v4f v1 = {0, 1, 0, 0};
    v4f v2 = {0, 0, 1, 0};
    v4f v3 = {0, 0, 0, 1};

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

    if(should_orient)
    {
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

    tetrad boosted = boost_tetrad(local_velocity, tet, metric);

    return boosted;
}

template<auto GetMetric, typename... T>
void build_initial_tetrads(execution_context& ectx, literal<v4f> position,
                           literal<v3f> local_velocity,
                           buffer_mut<v4f> position_out,
                           buffer_mut<v4f> e0_out, buffer_mut<v4f> e1_out, buffer_mut<v4f> e2_out, buffer_mut<v4f> e3_out,
                           T... extra)
{
    using namespace single_source;

    as_ref(position_out[0]) = position.get();

    m44f metric = GetMetric(position.get(), extra...);

    tetrad boosted = calculate_tetrad(metric, local_velocity.get(), true);

    as_ref(e0_out[0]) = boosted.v[0];
    as_ref(e1_out[0]) = boosted.v[1];
    as_ref(e2_out[0]) = boosted.v[2];
    as_ref(e3_out[0]) = boosted.v[3];
}

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

void init_rays_generic(execution_context& ectx, literal<v2i> screen_sizel,
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

    v2i screen_size = screen_sizel.get();

    if_e(y >= screen_size.y(), [&] {
        return_e();
    });

    if_e(x >= screen_size.x(), [&] {
        return_e();
    });

    v2i screen_position = {x, y};

    tetrad tetrads = {e0[0], e1[0], e2[0], e3[0]};

    v3f grid_position = world_to_grid(position[0].yzw(), dim.get(), scale.get());

    grid_position = clamp(grid_position, (v3f){2,2,2}, (v3f)dim.get() - (v3f){3,3,3});

    pin(grid_position);

    v3f ray_direction = get_ray_through_pixel(screen_position, screen_size, 90, camera_quat.get());

    geodesic my_geodesic = make_lightlike_geodesic(position[0], ray_direction, tetrads);

    v2i out_dim = screen_sizel.get();
    v2i out_pos = {x, y};

    as_ref(positions_out[out_pos, out_dim]) = my_geodesic.position;

    if_e(is_adm.get() == 1, [&]{
        adm_variables init_adm = admf_at(grid_position, dim.get(), in);

        tensor<valuef, 4> normal = get_adm_hypersurface_normal_raised(init_adm.gA, init_adm.gB);
        tensor<valuef, 4> normal_lowered = get_adm_hypersurface_normal_lowered(init_adm.gA);

        valuef E = -sum_multiply(my_geodesic.velocity, normal_lowered);

        tensor<valuef, 4> adm_velocity = ((my_geodesic.velocity / E) - normal);

        as_ref(velocities_out[out_pos, out_dim]) = adm_velocity;
    });

    if_e(is_adm.get() == 0, [&]{
        as_ref(velocities_out[out_pos, out_dim]) = my_geodesic.velocity;
    });
}

void build_raytrace_init_kernels(cl::context ctx)
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
                return Guv_at(pos, dim.get(), in, last_slice.get());
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
        return value_impl::make_function(init_rays_generic, "init_rays_generic");
    }, {"init_rays_generic"});
}

#include "particles.hpp"
#include "integration.hpp"
#include "bssn.hpp"
#include "init_general.hpp"

//3d
valuef dirac_delta_v(const valuef& r, const valuef& radius)
{
    valuef frac = r / radius;

    valuef mult = 1/(M_PI * pow(radius, 3.f));

    valuef result = 0;

    valuef branch_1 = (1.f/4.f) * pow(2.f - frac, 3.f);
    valuef branch_2 = 1.f - (3.f/2.f) * pow(frac, 2.f) + (3.f/4.f) * pow(frac, 3.f);

    result = ternary(frac <= 2, mult * branch_1, 0.f);
    result = ternary(frac <= 1, mult * branch_2, result);

    return result;
}

//3d
float dirac_delta_f(const float& r, const float& radius)
{
    float frac = r / radius;

    float mult = 1/(M_PI * pow(radius, 3.f));

    float result = 0;

    float branch_1 = (1.f/4.f) * pow(2.f - frac, 3.f);
    float branch_2 = 1.f - (3.f/2.f) * pow(frac, 2.f) + (3.f/4.f) * pow(frac, 3.f);

    //result = frac <= 2 ? mult * branch_1 : 0.f;
    //result = frac <= 1 ? mult * branch_2 : result;

    if(frac <= 1)
        return mult * branch_2;
    if(frac <= 2)
        return mult * branch_1;

    return 0.f;
}

//3d
float dirac_delta_cells_without_prefix(const float& r_cells, const float& radius_cells)
{
    float frac = r_cells / radius_cells;

    float result = 0;

    float branch_1 = (1.f/4.f) * pow(2.f - frac, 3.f);
    float branch_2 = 1.f - (3.f/2.f) * pow(frac, 2.f) + (3.f/4.f) * pow(frac, 3.f);

    //result = frac <= 2 ? mult * branch_1 : 0.f;
    //result = frac <= 1 ? mult * branch_2 : result;

    if(frac <= 1)
        return branch_2;
    if(frac <= 2)
        return branch_1;

    return 0.f;
}

//1d
float dirac_delta_1d(const float& r)
{
    if(r >= 1)
        return 0.f;

    return 1 - r;
}

template<typename T>
inline
T get_dirac(auto&& func, tensor<T, 3> world_pos, tensor<T, 3> dirac_location, T radius, T scale)
{
    T r = (world_pos - dirac_location).length();

    #ifdef GET_DIRAC_STANDARD
    return func(r, radius);
    #endif // GET_DIRAC_STANDARD

    #define GET_DIRAC_CORRECTED
    #ifdef GET_DIRAC_CORRECTED
    tensor<T, 3> scale3 = {scale, scale, scale};

    auto im1 = world_pos - scale3 / 2;
    auto ip1 = world_pos + scale3 / 2;

    return integrate_3d_trapezoidal([&](T x, T y, T z)
    {
        tensor<T, 3> pos = {x, y, z};

        return func((pos - dirac_location).length(), radius);
    }, 10, ip1, im1) / (scale*scale*scale);
    #endif // GET_DIRAC_CORRECTED
}

//so. det(cA) = c^n det(A). For us, c^3
//in a conformally flat spacetime, det(cfl) = 1
//we have the quantity E = mass dirac lorentz / sqrt(det(Y))
//we're calculating the quantity E * sqrt(det(Y))
//need to calculate what sqrt(det(Y)) is in terms of phis
//Y_ij = phi^4 cfl_flat
//det(Y_ij) = det(phi^4 cfl_flat)
//det(Y_ij) = (phi^4)^3 = phi^12?
//sqrt(det(Y_ij)) = phi^12^0.5 = phi^6
//E = non_cfl_E  phi^-6

//https://arxiv.org/pdf/1611.07906 16
void calculate_particle_nonconformal_E(execution_context& ectx, particle_base_args<buffer<valuef>> in,
                                       buffer_mut<valuei64> nonconformal_E_out,
                                       literal<v3i> dim, literal<valuef> scale, literal<value<size_t>> particle_count,
                                       literal<valued> fixed_scale)
{
    using namespace single_source;

    value<size_t> id = value_impl::get_global_id_us(0);
    pin(id);

    if_e(id >= particle_count.get(), [&]{
        return_e();
    });

    int radius_cells = 3;
    valuef radius_world = radius_cells * scale.get();

    //valuef dirac_prefix = 1/(M_PI * pow(radius_world, 3.f));

    valuef lorentz = 1;
    valuef mass = in.get_mass(id);
    v3f pos = in.get_position(id);
    v3f vel = in.get_velocity(id);

    v3i cell = (v3i)floor(world_to_grid(pos, dim.get(), scale.get()));

    valuef E = in.mass[id] * lorentz;

    int spread = radius_cells + 1;

    for(int z = -spread; z <= spread; z++)
    {
        for(int y = -spread; y <= spread; y++)
        {
            for(int x = -spread; x <= spread; x++)
            {
                v3i offset = {x, y, z};
                offset += cell;

                offset = clamp(offset, (v3i){0,0,0}, dim.get() - 1);

                v3f world_offset = (v3f)offset * scale.get();

                valuef dirac = dirac_delta_v((world_offset - pos).length(), radius_world);

                if_e(dirac > 0, [&]{
                    //the mass may be extremely small
                    //E might be absolutely tiny
                    valued E_d = (valued)E;
                    valued E_scaled = E_d * fixed_scale.get();

                    valuei64 as_i64 = (valuei64)E_scaled;

                    ///[offset, dim.get()]

                    valuei idx = offset.z() * dim.get().y() * dim.get().x() + offset.y() * dim.get().x() + offset.x();

                    nonconformal_E_out.atom_add_e(idx, as_i64);
                });
            }
        }
    }
}

void fixed_to_float(execution_context& ectx, buffer<valuei64> in, buffer<valuef> out, literal<valued> fixed_scale, literal<valuei> count)
{
    using namespace single_source;

    valuei id = value_impl::get_global_id(0);
    pin(id);

    if_e(id >= count.get(), [&]{
        return_e();
    });

    valued as_double = (valued)in[id] / fixed_scale.get();

    out[id] = (valuef)as_double;
}

void boot_particle_kernels(cl::context ctx)
{
    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(calculate_particle_nonconformal_E, "calculate_particle_nonconformal_E");
    }, {"calculate_particle_nonconformal_E"});

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(fixed_to_float, "fixed_to_float");
    }, {"fixed_to_float"});
}

//so. I need to calculate E, without the conformal factor
//https://arxiv.org/pdf/1611.07906 16
void particle_initial_conditions(cl::context& ctx, cl::command_queue& cqueue, discretised_initial_data& to_fill, particle_data& data, t3i dim, float scale)
{
    cl::buffer intermediate(ctx);
    intermediate.alloc(sizeof(cl_long) * dim.x() * dim.y() * dim.z());

    ///assume that our total mass is K
    ///to correctly sum it, we want the scale to be.. well, each particle's mass is K/N
    ///and we want.. 3 digits ( = log2(10^3) = 10 bits) of precision? as many as possible?

    double approx_total_mass = 1;
    double fixed_scale = ((double)data.count / approx_total_mass) * pow(10., 4.);

    {
        cl_ulong count = data.count;

        cl::args args;
        args.push_back(data.positions[0], data.positions[1], data.positions[2]);
        args.push_back(data.velocities[0], data.velocities[1], data.velocities[2]);
        args.push_back(data.masses);
        args.push_back(intermediate);
        args.push_back(dim);
        args.push_back(scale);
        args.push_back(count);
        args.push_back(fixed_scale);

        cqueue.exec("calculate_particle_nonconformal_E", args, {dim.x(), dim.y(), dim.z()}, {8, 8, 1});
    }

    {
        int size = dim.x() * dim.y() * dim.z();

        cl::args args;
        args.push_back(intermediate);
        args.push_back(to_fill.particles_contrib);
        args.push_back(fixed_scale);
        args.push_back(size);

        cqueue.exec("fixed_to_float", args, {size}, {128});
    }
}

void dirac_test()
{
    t3f dirac_location = {0, 0, 0.215f};

    int grid_size = 5;
    float world_width = 5;
    float scale = (world_width / (grid_size - 1));

    std::vector<float> values;
    values.resize(grid_size * grid_size * grid_size);

    int centre = (grid_size - 1)/2;

    auto w2g = [&](t3f world)
    {
        return (world / scale) + (t3f){centre, centre, centre};
    };

    auto g2w = [&](t3f grid)
    {
        return (grid - (t3f){centre, centre, centre}) * scale;
    };

    for(int z=0; z < grid_size; z++)
    {
        for(int y=0; y < grid_size; y++)
        {
            for(int x=0; x < grid_size; x++)
            {
                t3i gpos = {x, y, z};
                t3f wpos = g2w((t3f)gpos);

                float dirac = get_dirac(dirac_delta_f, wpos, dirac_location, 1.f, scale);

                values[z * grid_size * grid_size + y * grid_size + x] = dirac;
            }
        }
    }

    float integrated = 0.f;

    for(auto& i : values)
    {
        integrated += i * scale * scale * scale;
    }

    std::cout << "Integrated " << integrated << std::endl;

    #ifdef DIRAC_1D
    float dirac_location = 0.215f;

    int grid_size = 5;
    float world_width = 5;
    float scale = (world_width / (grid_size - 1));

    std::vector<float> values;
    values.resize(grid_size);

    int centre = (grid_size - 1)/2;

    auto w2g = [&](float world)
    {
        return (world / scale) + centre;
    };

    auto g2w = [&](float grid)
    {
        return (grid - centre) * scale;
    };

    for(int i=0; i < values.size(); i++)
    {
        float world = g2w(i);

        float im1 = world - scale / 2.f;
        float ip1 = world + scale / 2.f;

        //float dirac = dirac_delta2(fabs(world - dirac_location));

        /*float dirac = integrate_1d_trapezoidal([&](float in)
        {
            return dirac_delta2(fabs(in - dirac_location));
        }, 10, ip1, im1) / scale;*/

        float dirac = test_dirac1(fabs(world - dirac_location));

        printf("Dirac %f\n", dirac);

        /*float dirac = integrate_1d_trapezoidal([&](float in)
        {
            return test_dirac1(fabs(in - dirac_location));
        }, 100, ip1, im1) / scale;*/

        values[i] = dirac;
    }

    float integrated = 0.f;

    for(auto& i : values)
    {
        integrated += i * scale;
    }

    std::cout << "Integrated " << integrated << std::endl;
    #endif
}

std::vector<buffer_descriptor> particle_buffers::get_description()
{
    buffer_descriptor p0;
    p0.name = "p0";
    p0.sommerfeld_enabled = false;

    buffer_descriptor p1;
    p1.name = "p1";
    p1.sommerfeld_enabled = false;

    buffer_descriptor p2;
    p2.name = "p2";
    p2.sommerfeld_enabled = false;

    buffer_descriptor v0;
    v0.name = "v0";
    v0.sommerfeld_enabled = false;

    buffer_descriptor v1;
    v1.name = "v1";
    v1.sommerfeld_enabled = false;

    buffer_descriptor v2;
    v2.name = "v2";
    v2.sommerfeld_enabled = false;

    buffer_descriptor mass;
    mass.name = "mass";
    mass.sommerfeld_enabled = false;

    return {p0, p1, p2, v0, v1, v2, mass};
}

std::vector<cl::buffer> particle_buffers::get_buffers()
{
    return {pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], mass};
}

void particle_buffers::allocate(cl::context ctx, cl::command_queue cqueue, t3i size)
{
    for(int i=0; i < 3; i++)
    {
        pos[i].alloc(sizeof(cl_float) * particle_count);
        vel[i].alloc(sizeof(cl_float) * particle_count);
    }

    mass.alloc(sizeof(cl_float) * particle_count);
};

void particle_plugin::add_args_provider(all_adm_args_mem& mem)
{
    mem.add(full_particle_args<buffer<valuef>>());
}

buffer_provider* particle_plugin::get_buffer_factory(cl::context ctx)
{
    return new particle_buffers(ctx, particle_count);
}

buffer_provider* particle_plugin::get_utility_buffer_factory(cl::context ctx)
{
    return new particle_utility_buffers;
}

particle_plugin::particle_plugin(cl::context ctx, uint64_t _particle_count) : particle_count(_particle_count)
{
    boot_particle_kernels(ctx);
}

//consider implementing 3.2 https://arxiv.org/pdf/1905.08890

template struct full_particle_args<buffer<valuef>>;
template struct full_particle_args<buffer_mut<valuef>>;

template<typename T>
valuef full_particle_args<T>::adm_p(bssn_args& args, const derivative_data& d)
{
    return 0.f;
}

template<typename T>
tensor<valuef, 3> full_particle_args<T>::adm_Si(bssn_args& args, const derivative_data& d)
{
    return {};
}

template<typename T>
tensor<valuef, 3, 3> full_particle_args<T>::adm_W2_Sij(bssn_args& args, const derivative_data& d)
{
    return {};
}

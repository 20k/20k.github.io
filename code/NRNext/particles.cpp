#include "particles.hpp"
#include "integration.hpp"
#include "bssn.hpp"

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

/*void sum_rest_mass(execution_context& ectx, bssn_args_mem<buffer<valuef>> in,
                    hydrodynamic_base_args<buffer<valuef>> hydro,
                    hydrodynamic_utility_args<buffer<valuef>> util,
                    literal<v3i> idim,
                    literal<valuei> positions_length,
                    literal<valuef> scale, buffer_mut<value<std::int64_t>> sum)*/

//https://arxiv.org/pdf/1611.07906 16
void calculate_particle_nonconformal_E(execution_context& ectx, particle_base_args<buffer<valuef>> in,
                                       buffer<valuei64> nonconformal_E_out,
                                       literal<v3i> dim, literal<valuef> scale, literal<value<size_t>> particle_count)
{
    using namespace single_source;

    value<size_t> id = value_impl::get_global_id_us(0);

    if_e(id >= particle_count.get(), [&]{
        return_e();
    });

    int radius_cells = 3;
    valuef radius_world = radius_cells * scale.get();

    valuef dirac_prefix = 1/(M_PI * pow(radius_world, 3.f));

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
                t3f offset_cpu = {x, y, z};
                float dirac_len = offset_cpu.length();

                float dirac_suffix = dirac_delta_cells_without_prefix(dirac_len, radius_cells);

                if(dirac_suffix == 0)
                    continue;

                v3i offset = {x, y, z};
                offset += cell;

                offset = clamp(offset, (v3i){0,0,0}, dim.get() - 1);

                v3f world_offset = (v3f)offset * scale.get();

                valuef dirac = dirac_prefix * dirac_suffix;
            }
        }
    }
}

void boot_particle_kernels(cl::context ctx)
{
    /*
    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(init_hydro, "init_hydro", use_colour);
    }, {"init_hydro"});
    */
}

//so. I need to calculate E, without the conformal factor
//https://arxiv.org/pdf/1611.07906 16
void initialise_particles(discretised_initial_data& to_fill, particle_data& data, cl::command_queue& cqueue, t3i dim, float scale)
{

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

particle_plugin::particle_plugin(cl::context ctx)
{
    boot_particle_kernels(ctx);
}

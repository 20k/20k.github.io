#include "particles.hpp"
#include "integration.hpp"

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

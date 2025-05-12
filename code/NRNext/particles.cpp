#include "particles.hpp"
#include "integration.hpp"

float dirac_delta(const float& r, const float& radius)
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

float dirac_delta3(float x, float dx)
{
    if(x <= 0.5f * dx)
    {
        return 3.f/4.f + pow(x / dx, 2.f);
    }

    if(x <= (3.f/2.f) * dx)
    {
        return 0.5f * pow(3.f/2.f - x/dx, 2.f);
    }

    return 0.f;
}

float test_dirac1(float r)
{
    return (4.f/3.f) * M_PI * dirac_delta(r, 1);
}

float dirac_delta2(const float& r)
{
    if(r >= 1)
        return 0.f;

    return 1 - r;
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

                #ifdef DIRAC_STANDARD
                t3f rel = wpos - dirac_location;
                float dirac = dirac_delta(rel.length(), 1.f);
                #endif

                #define DIRAC_EXACT
                #ifdef DIRAC_EXACT
                t3f im1 = wpos - (t3f){scale/2, scale/2, scale/2};
                t3f ip1 = wpos + (t3f){scale/2, scale/2, scale/2};

                float dirac = integrate_3d_trapezoidal([&](float x, float y, float z)
                {
                    t3f pos = {x, y, z};

                    return dirac_delta((pos - dirac_location).length(), 1.f);
                }, 100, ip1, im1) / (scale*scale*scale);
                #endif

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

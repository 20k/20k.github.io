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

float dirac_delta2(const float& r)
{
    if(r >= 1)
        return 0.f;

    return 1 - r;
}

void dirac_test()
{
    //float radius = 2.f;
    /*float dirac_location = 0.215f;

    float h = (10 / 1000.);

    auto func = [&](float in)
    {
        float x = fabs(in - dirac_location);

        return dirac_delta(x, 1.f);
        //return dirac_delta3(x, h);
    };

    //valuef result = integrate_1d_trapezoidal(func, 100, 0.f, 1.f);
    float result = integrate_1d_trapezoidal(func, 1000, 5.f, -5.f);

    std::cout << "Result " << result << std::endl;

    assert(false);*/

    float dirac_location = 0.215f;

    int grid_size = 101;
    float world_width = 3;
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

        float dirac = dirac_delta2(fabs(world - dirac_location));

        //printf("")

        values[i] = dirac;
    }

    float integrated = 0.f;

    for(auto& i : values)
    {
        integrated += i * scale;
    }

    std::cout << "Integrated " << integrated << std::endl;

    //std::cout << value_to_string(result) << std::endl;
}

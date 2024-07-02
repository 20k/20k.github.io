#include "accretion_disk.hpp"
#include <cmath>
#include <numbers>
#include <vec/vec.hpp>
#include <iostream>
#include <SFML/Graphics.hpp>
#include "../common/vec/tensor.hpp"
#include "blackbody.hpp"

template<typename T>
T linear_to_srgb(const T& in)
{
    if(in <= 0.0031308)
        return in * 12.92;
    else
        return 1.055 * pow(in, 1.0 / 2.4) - 0.055;
}

template<typename T>
tensor<T, 3> linear_to_srgb(const tensor<T, 3>& in)
{
    tensor<T, 3> ret;

    for(int i=0; i < 3; i++)
        ret[i] = linear_to_srgb(in[i]);

    return ret;
}

double get_isco(double M, double a)
{
    double a_star = a/M;

    double Z1 = 1 + pow(1 - a_star*a_star, 1./3.) * (pow(1 + a_star, 1./3.) + pow(1 - a_star, 1./3.));
    double Z2 = sqrt(3 * a_star * a_star + Z1 * Z1);

    double r0_M = 3 + Z2 - sqrt((3 - Z1) * (3 + Z1 + 2 * Z2));

    return r0_M * M;
}

///we don't actually use this
double get_event_horizon(double M, double a)
{
    double rs = 2 * M;

    return (rs + (sqrt(rs*rs - 4*a*a))) / 2.;
}

constexpr double cst_G = 6.6743 * pow(10., -11.);
constexpr double cst_C = 299792458;

double kg_to_geom(double M)
{
    return M * cst_G / (cst_C * cst_C);
}

double geom_to_kg(double M)
{
    return M * cst_C * cst_C / cst_G;
}

///https://www-astro.physics.ox.ac.uk/~garret/teaching/lecture7-2012.pdf 12
///we're not using this, but you might if you want an accretion rate in kg
double eddington_limit_kg_s(double M_kg)
{
    double pi = std::numbers::pi_v<double>;

    double sigmaT = 6.65245863216 * pow(10., -29.);
    double proton_mass = 1.67262192595 * pow(10., -27.);

    double pm_t = proton_mass / sigmaT;;

    double e = 0.1;

    double kg_ps = 4 * pi * (cst_G * M_kg * pm_t) / (e * cst_C);

    return kg_ps;
}

namespace region_type
{
    enum region_t
    {
        INNER,
        MIDDLE,
        OUTER,
    };
};

///https://arxiv.org/pdf/1110.6556 is broken
///https://www.emis.de/journals/LRG/Articles/lrr-2013-1/articlese5.html
accretion_disk make_accretion_disk_kerr(float mass, float a)
{
    double Msol_kg = 1.988 * pow(10., 30.);

    double pi = std::numbers::pi_v<double>;

    double Msol_natural = kg_to_geom(Msol_kg);

    double m_star = mass / Msol_natural;

    //double eddington_kg_ps = eddington_limit_kg_s(geom_to_kg(mass));

    double mdot_star = 0.3;

    double a_star = a/mass;
    double assq = pow(a_star, 2.);

    double isco = get_isco(mass, a);

    double x1 = 2 * cos(acos(a_star)/3 - pi/3);
    double x2 = 2 * cos(acos(a_star)/3 + pi/3);
    double x3 = -2 * cos(acos(a_star)/3);

    ///unused
    //double horizon = get_event_horizon(mass, a);

    double outer_boundary = 2 * mass * 50;

    ///plunge, edge, inner, middle, outer
    region_type::region_t region = region_type::INNER;

    int max_steps = 1000;

    ///Viscosity
    double alpha = 0.1;

    double x0 = sqrt(isco/mass);

    std::vector<double> radius;
    std::vector<double> brightness;
    std::vector<double> temperature;

    for(int steps = 0; steps < max_steps; steps++)
    {
        double r = mix(isco, outer_boundary, steps / (double)max_steps);
        double x = sqrt(r/mass);

        double r_star = r / mass;

        double x_pow_m2 = mass/r;
        double x_pow_m4 = pow(mass/r, 2.);

        double A = 1 + assq * x_pow_m4 + 2 * assq * pow(x, -6.);
        double B = 1 + a_star * pow(x, -3.);
        double C = 1 - 3 * x_pow_m2 + 2 * assq * pow(x, -3.);
        double D = 1 - 2 * x_pow_m2 + assq * pow(x, -4.);
        double E = 1 + 4 * assq * x_pow_m4 - 4 * assq * pow(x,-6.) + 3 * pow(a_star, 4.) * pow(x, -8.);
        double Q = B * pow(C, -1/2.) * (1/x) * (x - x0 - (3/2.) * a_star * log(x/x0)
                                                - (3 * pow(x1 - a_star, 2.) / (x1 * (x1 - x2) * (x1 - x3))) * log((x - x1) / (x0 - x1))
                                                - (3 * pow(x2 - a_star, 2.) / (x2 * (x2 - x1) * (x2 - x3))) * log((x - x2) / (x0 - x2))
                                                - (3 * pow(x3 - a_star, 2.) / (x3 * (x3 - x1) * (x3 - x2))) * log((x - x3) / (x0 - x3)));

        if(region == region_type::INNER)
        {
            ///This is B/(1-B)
            double p_gas_p_rad = 4 * pow(10., -6.) * pow(alpha, -1/4.) * pow(m_star, -1/4.) * pow(mdot_star, -2.) * pow(r_star, 21/8.) * pow(A, -5/2.) * pow(B, 9/2.) * D * pow(E, 5/4.) * pow(Q, -2.);

            ///in the edge region, gas pressure dominates over radiation pressure
            ///in the inner region, gas pressure is less than radiation pressure
            ///in the middle region, gas pressure is greater than radiation pressure
            if(p_gas_p_rad > 1)
                region = region_type::MIDDLE;
        }

        if(region == region_type::MIDDLE)
        {
            //in the outer region opacity is free-free
            //in the middle region, opacity is electron scattering
            double Tff_Tes = (2 * pow(10., -6.)) * (pow(mdot_star, -1.)) * pow(r_star, 3/2.) * pow(A, -1.) * pow(B, 2.) * pow(D, 1/2.) * pow(E, 1/2.) * pow(Q, -1.);

            if(Tff_Tes >= 1)
                region = region_type::OUTER;
        }

        double surface_flux = 7 * pow(10., 26.) * pow(m_star, -1.) * mdot_star * pow(r_star, -3.) * pow(B, -1.) * pow(C, -1/2.) * Q;
        double T = 0;

        if(region == region_type::INNER)
            T = 5 * pow(10., 7.) * pow(alpha, -1/4.) * pow(m_star, -1/4.) * pow(r_star, -3/8.) * pow(A, -1/2.) * pow(B, 1/2.) * pow(E, 1/4.);

        ///'edge' region as well in the more developed model
        if(region == region_type::MIDDLE)
            T = 7 * pow(10., 8.) * pow(alpha, -1/5.) * pow(m_star, -1/5.) * pow(mdot_star, 2/5.) * pow(r_star, -9/10.) * pow(B, -2/5.) * pow(D, -1/5.) * pow(Q, 2/5.);

        if(region == region_type::OUTER)
            T = 2 * pow(10., 8.) * pow(alpha, -1/5.) * pow(m_star, -1/5.) * pow(mdot_star, 3/10.) * pow(r_star, -3/4.) * pow(A, -1/10.) * pow(B, -1/5.) * pow(D, -3/20.) * pow(E, 1/20.) * pow(Q, 3/10.);

        radius.push_back(r);
        brightness.push_back(surface_flux);
        temperature.push_back(T);
    }

    //brightness normalisation
    {
        double min_val = FLT_MAX;
        double max_val = 0;

        for(auto& b : brightness)
        {
            max_val = std::max(b, max_val);
            min_val = std::min(b, min_val);
        }

        for(auto& b : brightness)
        {
            #ifdef MAX_CONTRAST
            b -= min_val;
            b /= (max_val - min_val);
            #else
            b /= max_val;
            #endif
        }
    }

    std::vector<tensor<float, 3>> radial_colour;

    auto temperature_to_linear_rgb = [](double in)
    {
        #define BLACKBODY_EXACT
        #ifndef BLACKBODY_EXACT
        return blackbody_temperature_to_approximate_linear_rgb(in);
        #else
        return blackbody_temperature_to_linear_rgb(in);
        #endif
    };

    for(auto& T : temperature)
    {
        radial_colour.push_back(temperature_to_linear_rgb(T));
    }

    int tex_size = 2048;

    auto lookup_radius = [&](double coordinate_radius)
    {
        double my_brightness = 0;
        double my_temperature = 0;
        tensor<float, 3> my_linear_rgb;

        ///iterate from outside in, as there's a gap in the middle of our accretion disk
        for(int i=(int)radius.size() - 2; i >= 0; i--)
        {
            auto& rad = radius[i];

            ///we've found our value, interpolate
            if(coordinate_radius >= rad)
            {
                double upper = brightness[i + 1];
                double lower = brightness[i];

                coordinate_radius = std::clamp(coordinate_radius, lower, upper);

                float frac = (coordinate_radius - lower) / (upper - lower);

                my_brightness = mix(brightness[i], brightness[i + 1], frac);
                my_linear_rgb = mix(radial_colour[i], radial_colour[i + 1], frac);
                my_temperature = mix(temperature[i], temperature[i + 1], frac);
                break;
            }
        }

        return std::tuple{(float)my_brightness, my_linear_rgb, (float)my_temperature};
    };

    //purely for article purposes, img is never used
    {
        sf::Image img;
        img.create(tex_size, tex_size);

        ///generate a texture
        for(int j=0; j < tex_size; j++)
        {
            for(int i=0; i < tex_size; i++)
            {
                int centre = tex_size/2;

                float dj = (j - centre) / (float)centre;
                float di = (i - centre) / (float)centre;

                float rad = sqrt(di * di + dj * dj);

                auto [my_brightness, my_linear_rgb, _] = lookup_radius(rad * outer_boundary);

                assert(my_brightness >= 0 && my_brightness <= 1);

                tensor<float, 3> srgb = linear_to_srgb(my_brightness * my_linear_rgb);
                sf::Color col(255 * srgb.x(), 255 * srgb.y(), 255 * srgb.z(), 255);
                img.setPixel(i, j, col);
            }
        }

        img.saveToFile("out.png");
    }

    std::vector<tensor<float, 3>> brightness_out;
    std::vector<float> temperature_out;

    for(int i=0; i < tex_size; i++)
    {
        float rad = (float)i / tex_size;

        auto [my_brightness, _, my_temperature] = lookup_radius(rad * outer_boundary);

        brightness_out.push_back({my_brightness*5, my_brightness*5, my_brightness*5});
        temperature_out.push_back(my_temperature);
    }


    accretion_disk disk;

    disk.brightness = brightness_out;
    disk.temperature = temperature_out;

    return disk;
}

#include "accretion_disk.hpp"
#include <cmath>
#include <numbers>
#include <vec/vec.hpp>
#include <iostream>
#include <SFML/Graphics.hpp>

template<typename T>
T linear_to_srgb(const T& in)
{
    if(in <= 0.0031308)
        return in * 12.92;
    else
        return 1.055 * pow(in, 1.0 / 2.4) - 0.055;
}

constexpr double cpow(double a, double b)
{
    return pow(a, b);
}

double cpow(double a, int b)
{
    assert(false);
}

double get_isco(double M, double a)
{
    double a_star = a/M;

    double Z1 = 1 + cpow(1 - a_star*a_star, 1./3.) * (cpow(1 + a_star, 1./3.) + cpow(1 - a_star, 1./3.));
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

constexpr double cst_G = 6.6743 * cpow(10., -11.);
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

    double sigmaT = 6.65245863216 * cpow(10., -29.);
    double proton_mass = 1.67262192595 * pow(10., -27.);

    double pm_t = proton_mass / sigmaT;;

    double e = 0.1;

    double kg_ps = 4 * pi * (cst_G * M_kg * pm_t) / (e * cst_C);

    return kg_ps;
}

///https://arxiv.org/pdf/1110.6556 is broken
///https://www.emis.de/journals/LRG/Articles/lrr-2013-1/articlese5.html
accretion_disk make_accretion_disk_kerr(float mass, float a)
{
    double Msol_kg = 1.988 * cpow(10., 30.);

    double pi = std::numbers::pi_v<double>;

    double Msol_natural = kg_to_geom(Msol_kg);

    double m_star = mass / Msol_natural;

    //double eddington_kg_ps = eddington_limit_kg_s(geom_to_kg(mass));

    double mdot_star = 0.3;

    double a_star = a/mass;
    double assq = cpow(a_star, 2.);

    double isco = get_isco(mass, a);

    double x1 = 2 * cos(acos(a_star)/3 - pi/3);
    double x2 = 2 * cos(acos(a_star)/3 + pi/3);
    double x3 = -2 * cos(acos(a_star)/3);

    ///unused
    //double horizon = get_event_horizon(mass, a);

    double outer_boundary = 2 * mass * 50;

    ///0 = plunge, 1 = edge, 2 = inner, 3 = middle, 4 = outer
    int region = 2;

    int max_steps = 1000;

    ///Viscosity
    double alpha = 0.1;

    double x0 = sqrt(isco/mass);

    std::vector<std::pair<double, double>> brightness;

    for(int steps = 0; steps < max_steps; steps++)
    {
        double r = mix(isco, outer_boundary, steps / (double)max_steps);
        double x = sqrt(r/mass);

        float r_star = r / mass;

        float x_pow_m2 = mass/r;
        float x_pow_m4 = cpow(mass/r, 2.);

        double A = 1 + assq * x_pow_m4 + 2 * assq * cpow(x, -6.);
        double B = 1 + a_star * cpow(x, -3.);
        double C = 1 - 3 * x_pow_m2 + 2 * assq * cpow(x, -3.);
        double D = 1 - 2 * x_pow_m2 + assq * cpow(x, -4.);
        double E = 1 + 4 * assq * x_pow_m4 - 4 * assq * cpow(x,-6.) + 3 * cpow(a_star, 4.) * cpow(x, -8.);
        double Q = B * cpow(C, -1/2.) * (1/x) * (x - x0 - (3/2.) * a_star * log(x/x0)
                                                - (3 * cpow(x1 - a_star, 2.) / (x1 * (x1 - x2) * (x1 - x3))) * log((x - x1) / (x0 - x1))
                                                - (3 * cpow(x2 - a_star, 2.) / (x2 * (x2 - x1) * (x2 - x3))) * log((x - x2) / (x0 - x2))
                                                - (3 * cpow(x3 - a_star, 2.) / (x3 * (x3 - x1) * (x3 - x2))) * log((x - x3) / (x0 - x3))
                                                );
        ///This is B/(1-B)
        double p_gas_p_rad = 0;

        if(region == 4)
            p_gas_p_rad = 3 * cpow(alpha, -1/10.) * cpow(m_star, -1/10.) * cpow(mdot_star, -7/20.) * cpow(r_star, 3/8.) * cpow(A, -11/20.) * cpow(B, 9/10.) * cpow(D, 7/40.) * cpow(E, 11/40.) * cpow(Q, -7/20.);

        if(region == 3)
            p_gas_p_rad = 7 * cpow(10., -3.) * cpow(alpha, -1/10.) * cpow(m_star, -1/10.) * cpow(mdot_star, -4/5.) * cpow(r_star, 21/20.) * cpow(A, -1.) * cpow(B, 9/5.) * cpow(D, 2/5.) * cpow(E, 1/2.) * cpow(Q, -4/5.);

        if(region == 2)
            p_gas_p_rad = 4 * cpow(10., -6.) * cpow(alpha, -1/4.) * cpow(m_star, -1/4.) * cpow(mdot_star, -2.) * cpow(r_star, 21/8.) * cpow(A, -5/2.) * cpow(B, 9/2.) * D * cpow(E, 5/4.) * cpow(Q, -2.);

        ///in the edge region, gas pressure dominates over radiation pressure
        ///in the inner region, gas pressure is less than radiation pressure
        ///in the middle region, gas pressure is greater than radiation pressure
        if(region == 2 && p_gas_p_rad > 1)
            region = 3;

        //in the outer region opacity is free-free
        //in the middle region, opacity is electron scattering
        double Tff_Tes = (2 * cpow(10., -6.)) * (cpow(mdot_star, -1.)) * cpow(r_star, 3/2.) * cpow(A, -1.) * cpow(B, 2.) * cpow(D, 1/2.) * cpow(E, 1/2.) * cpow(Q, -1.);

        if(region == 3 && Tff_Tes >= 1)
            region = 4;

        double surface_flux = 0;
        double temperature = 0;

        if(region == 2)
        {
            surface_flux = 7 * cpow(10., 26.) * cpow(m_star, -1.) * mdot_star * cpow(r_star, -3.) * cpow(B, -1.) * cpow(C, -1/2.) * Q;
        }

        if(region == 1 || region == 3)
        {
            surface_flux = 7 * cpow(10., 26.) * cpow(m_star, -1.) * mdot_star * cpow(r_star, -3.) * cpow(B, -1.) * cpow(C, -1/2.) * Q;
        }

        if(region == 4)
        {
            surface_flux = 7 * cpow(10., 26.) * cpow(m_star, -1.) * mdot_star * cpow(r_star, -3.) * cpow(B, -1.) * cpow(C, -1/2.) * Q;
        }

        if(!std::isnan(surface_flux))
        {
            brightness.push_back({r, surface_flux});
        }
    }

    //brightness normalisation
    {
        double max_bright = 0;
        double min_bright = FLT_MAX;

        for(auto& [a, b] : brightness)
        {
            max_bright = std::max(b, max_bright);
            min_bright = std::min(b, min_bright);
        }

        for(auto& [a, b] : brightness)
        {
            #ifdef MAX_CONTRAST
            b -= min_bright;
            b /= (max_bright - min_bright);
            #else
            b /= max_bright;
            #endif
        }
    }

    int tex_size = 2048;

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

            double max_coordinate_boundary = 1;

            double max_physical_boundary = outer_boundary;

            double my_physical_radius = rad * (max_physical_boundary / max_coordinate_boundary);

            double my_brightness = 0;

            ///iterate from outside in, as there's a gap in the middle of our accretion disk
            for(int i=(int)brightness.size() - 2; i >= 0; i--)
            {
                auto& [pr, b] = brightness[i];

                ///we've found our value, interpolate
                if(my_physical_radius >= pr)
                {
                    double upper = brightness[i + 1].first;
                    double lower = pr;

                    my_physical_radius = std::clamp(my_physical_radius, lower, upper);

                    double frac = (my_physical_radius - lower) / (upper - lower);

                    my_brightness = mix(b, brightness[i + 1].second, frac);
                    break;
                }
            }

            assert(my_brightness >= 0 && my_brightness <= 1);

            double my_srgb = linear_to_srgb(my_brightness);

            sf::Color col(255 * my_srgb, 255 * my_srgb, 255 * my_srgb, 255);

            img.setPixel(i, j, col);
        }
    }

    img.saveToFile("out.png");

    accretion_disk disk;

    disk.normalised_brightness = img;

    return disk;
}

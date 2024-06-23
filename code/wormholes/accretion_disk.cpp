#include "accretion_disk.hpp"
#include <cmath>
#include <numbers>
#include <vec/vec.hpp>
#include <iostream>


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

double geom_to_time(double t)
{
    return t / cst_C;
}

double geom_to_itime(double t)
{
    return t * cst_C;
}

double itime_to_geom(double t)
{
    return t / cst_C;
}

double eddington_limit_kg_s(double M)
{
    double pi = std::numbers::pi_v<double>;

    double M_in_kg = M * cst_C * cst_C / cst_G;

    double sigmaT = 6.65245863216 * cpow(10., -29.);
    double proton_mass = 1.67262192595 * pow(10., -27.);

    double pm_t = proton_mass / sigmaT;;

    double e = 0.1;

    double kg_ps = 4 * pi * (cst_G * M_in_kg * pm_t) / (e * cst_C);

    return kg_ps;

    //return itime_to_geom(kg_to_geom(kg_ps));
}

///https://arxiv.org/pdf/1110.6556
accretion_disk make_accretion_disk_kerr(float mass, float a)
{
    mass *= 0.1;
    a = 0;

    double Msol = 1.988 * cpow(10., 30.);

    double pi = std::numbers::pi_v<double>;

    double Msol_natural = (Msol * cst_G) / (cst_C*cst_C);

    //mass = 10 * Msol_natural;

    double m_star = mass / (3 * Msol_natural);

    //std::cout << m_star << std::endl;

    double eddington_kg_ps = eddington_limit_kg_s(mass);

    std::cout << "Eddington " << eddington_kg_ps << std::endl;

    ///sure why not
    //double mdot_star = 0.001;

    ///they want grams per second
    double mdot_star = 0.3 * eddington_kg_ps * 1000 / pow(10., 17.);

    //a = 0;

    double a_star = a/mass;
    double assq = cpow(a_star, 2.);

    double isco = get_isco(mass, a);

    std::cout << "ISCOF " << isco / (2 * mass) << std::endl;



    double x1 = 2 * cos(acos(a_star)/3 - pi/3);
    double x2 = 2 * cos(acos(a_star)/3 + pi/3);
    double x3 = -2 * cos(acos(a_star)/3);

    double horizon = get_event_horizon(mass, a);

    ///10 radii out??
    double outer_boundary = 2 * mass * 200;

    ///0 = plunge, 1 = edge, 2 = inner, 3 = middle, 4 = outer
    int region = 0;

    int max_steps = 1000;

    ///VISCOSITY
    double alpha = 0.1;

    std::cout << "ISCO " << isco << " horizon " << horizon << " Boundary " << outer_boundary << " mass " << mass << std::endl;

    double x0 = sqrt(isco/mass);
    double F0 = 1 - 2 * a_star * cpow(x0, -3.) + assq * cpow(x0, -4.);
    double G0 = 1 - 2 * cpow(x0, -2.) + a_star * cpow(x0, -3.);
    double C0 = 1 - 3 * cpow(x0, -2.) + 2 * assq * cpow(x0, -3.);
    double D0 = 1 - 2 * cpow(x0, -2.) + assq * cpow(x0, -4.);
    double V0 = cpow(D0, -1.) * (1 + cpow(x0, -4.) * (assq - cpow(x0, 2.) * cpow(F0, 2.) * cpow(G0, -2.)) + 2 * cpow(x0, -6.) * (a_star - x0 * F0 * cpow(G0, -1.)));

    std::cout << "C0 " << C0 << " G0 " << G0 << std::endl;

    std::cout << "F0 " << F0 << std::endl;

    std::cout << "X0 " << x0 << std::endl;

    for(int steps = 0; steps < max_steps; steps++)
    {
        double r = mix(horizon, outer_boundary, steps / (double)max_steps);

        if(region == 0 && r >= isco)
            region = 1;

        double x = sqrt(r/mass);

        std::cout << "X " << x << std::endl;

        ///so, when r/m > 2, we're outside the event horizon
        ///because rs = 2m


        float x_pow_m2 = mass/r;
        float x_pow_m4 = cpow(mass/r, 2.);

        ///ok so, x_pow_m2 has a value of 1/2 at the event horizon
        ///x_pow_m4 has a value of 1/4

        double A = 1 + assq * x_pow_m4 + 2 * assq * cpow(x, -6.);
        double B = 1 + a_star * cpow(x, -3.);
        double C = 1 - 3 * x_pow_m2 + 2 * assq * cpow(x, -3.);
        double D = 1 - 2 * (mass/r) + assq * cpow(x, -4.);
        //double D = 1 - 2 * cpow(x, -2.) + assq * cpow(x, -4.);
        double E = 1 + 4 * assq * x_pow_m4 - 4 * assq * cpow(x,-6.) + 3 * cpow(a_star, 4.) * cpow(x, -8.);
        double F = 1 - 2 * a_star * cpow(x, -3.) + assq * x_pow_m4;
        double G = 1 - 2 * x_pow_m2 + a_star * cpow(x, -3.);
        double H = 1 - 2 * x_pow_m2 + 2 * a_star * cpow(x, -2.) * cpow(x0, -1.) * cpow(F0, -1.) * G0;
        double I = A - 2 * a_star * cpow(x, -6.) * x0 * F0 * cpow(G0, -1.);
        double O = H * cpow(I, -1.);
        double J = O - cpow(x, -2.) * cpow(I, -1.) * (1 - a_star * cpow(x0, -1.) * cpow(F0, -1.) * G0 + assq * cpow(x, -2.) * H * cpow(I, -1.) * (1 + 3 * cpow(x, -2.) - 3 * cpow(a_star, -1.) * cpow(x, -2.) * x0 * F0 * cpow(G0, -1.)));
        double K = fabs(A * J * cpow(1 - cpow(x, -4.) * cpow(A, 2.) * cpow(D, -1.) * cpow(x0 * F0 * cpow(G0, -1.) * O - 2 * a_star * cpow(x, -2.) * cpow(A, -1.), 2.), -1.));
        double Q = B * cpow(C, -1/2.) * (1/x) * (x - x0 - (3/2.) * a_star * log(x/x0)
                                                - (3 * cpow(x1 - a_star, 2.) / (x1 * (x1 - x2) * (x1 - x3))) * log((x - x1) / (x0 - x1))
                                                - (3 * cpow(x2 - a_star, 2.) / (x2 * (x2 - x1) * (x2 - x3))) * log((x - x2) / (x0 - x2))
                                                - (3 * cpow(x3 - a_star, 2.) / (x3 * (x3 - x1) * (x3 - x2))) * log((x - x3) / (x0 - x3))
                                                );
        double R = cpow(F, 2.) * cpow(C, -1.) - assq * cpow(x, -2.) * (G * cpow(C, -1./2.) - 1);
        double S = cpow(A, 2.) * cpow(B, -2.) * C * cpow(D, -1.) * R;
        double V = cpow(D, -1.) * (1 + cpow(x, -4.) * (assq - cpow(x0, 2.) * cpow(F0, 2.) * cpow(G0, -2.)) + 2 * cpow(x, -6.) * (a_star - x0 * F0 * cpow(G0, -1.)));

        double Phi = Q + (0.02) * (cpow(alpha, 9/8.) * cpow(m_star, -3/8.) * cpow(mdot_star, 1/4.) * cpow(x, -1.) * B * cpow(C, -1/2.) * (cpow(x0, 9/8.) * cpow(C0, -5/8.) * G0 * cpow(V0, 1/2.)));

        double p_gas_p_rad = (5 * cpow(10., -5.)) * (cpow(alpha, -1/4.) * cpow(m_star, 7/4.) * cpow(mdot_star, -2.)) * cpow(x, 21/4.) * cpow(A, -5/2.) * cpow(B, 9/2.) * D * cpow(S, 5/4.) * cpow(Phi, -2.);

        double v_star_inner = (cpow(C0, -1.) * cpow(G0, 2.) * V - 1) + 7 * cpow(10., -3.) * cpow(alpha, 1/4.) * cpow(m_star, -3/4.) * cpow(mdot_star, 1/2.) * cpow(x0, -7/4.) * cpow(C0, -5/4.) * cpow(D0, -1.) * cpow(G0, 2.) * V;

        //v_star_inner = fabs(v_star_inner);

        std::cout << "Test " << (1/(x*x*x*x)) * -x0*x0 * F0*F0 * 1/(G0*G0) << std::endl;
        std::cout << "Test2 " << 2 * cpow(x, -6.) * -x0 * F0 / G0 << std::endl;
        std::cout << "D " << D << std::endl;

        std::cout << "A " << A << " B " << B << " C " << C << " D " << D << " E " << E << std::endl;
        std::cout << "V " << V << std::endl;

        //std::cout << "VTest " << cpow(D, -1.) << std::endl;

        //std::cout << "DOM " << (cpow(C0, -1.) * cpow(G0, 2.) * V - 1) << std::endl;

        double v_star = sqrt(v_star_inner);

        ///in the edge region, gas pressure dominates over radiation pressure
        ///in the inner region, gas pressure is less than radiation pressure
        ///in the middle region, gas pressure is greater than radiation pressure
        if(region == 1 && p_gas_p_rad < 1)
            region = 2;

        if(region == 2 && p_gas_p_rad > 1)
            region = 3;

        //in the outer region opacity is free-free
        //in the middle region, opacity is electron scattering
        double Tff_Tes = (0.6 * cpow(10.,-5.)) * (m_star * cpow(mdot_star, -1.)) * cpow(x, 3.) * cpow(A, -1.) * cpow(B, 2.) * cpow(D, 1/2.) * cpow(S, 1/2.) * cpow(Phi, -1.);

        //std::cout << "Tff " << Tff_Tes << std::endl;

        if(Tff_Tes >= 1)
            region = 4;

        double surface_flux = 0;
        double temperature = 0;
        double density = 0;
        double surface_density = 0;

        if(region == 0)
        {
            //std::cout << "Vs? " << v_star << " in " << v_star_inner << std::endl;
            /*std::cout << "C " << C0 << " D " << D0 << " V " << V << " x " << x << std::endl;

            std::cout << "assq " << assq << std::endl;

            std::cout << "VP1 " << 1 + cpow(x, -4.) * (assq - cpow(x0, 2.) * cpow(F0, 2.) * cpow(G0, -2.)) << std::endl;
            std::cout << "VP2 " << 2 * cpow(x, -6.) * (a_star - x0 * F0 * cpow(G0, -1.)) << std::endl;
            std::cout << "D? " << cpow(D, -1.) << std::endl;
            std::cout << "D " << D << std::endl;*/


            //std::cout << "VSI " << v_star_inner << std::endl;

            //erg/cm^2 sec
            surface_flux = 2 * cpow(10., 18.) * (cpow(alpha, 4/3.) * cpow(m_star, -3.) * cpow(mdot_star, 5/3.)) * cpow(x, -26/3.) * cpow(x0, 4/3.) * cpow(D, -5/6.) * cpow(K, 4/3.) * cpow(F0, 4/3.) * cpow(G0, -4/3.) * cpow(v_star, -5/3.);
            surface_density = 1 * cpow(m_star, -1.) * mdot_star * cpow(x, -2.) * cpow(D, -1/2.) * cpow(v_star, -1.);
        }

        if(region == 1 || region == 3)
        {
            surface_flux = 0.6 * cpow(10., 26.) * (cpow(m_star, -2.) * mdot_star) * cpow(x, -6.) * cpow(B, -1.) * cpow(C,-1/2.) * Phi;
            surface_density = 5 * cpow(10., 4.) * cpow(alpha, -4/5.) * cpow(m_star, -2/5.) * cpow(mdot_star, 3/5.) * cpow(x, -6/5.) * cpow(B, -4/5.) * cpow(C, -1/2.) * cpow(D, -4/5.) * cpow(Phi, 3/5.);
        }

        if(region == 2)
        {
            surface_flux = 0.6 * cpow(10., 26.) * (cpow(m_star, -2.) * mdot_star) * cpow(x, -6.) * cpow(B, -1.) * cpow(C,-1/2.) * Phi;
            surface_density = 20 * cpow(alpha, -1.) * m_star * cpow(mdot_star, -1.) * cpow(x, 3.) * cpow(A, -2.) * cpow(B, 3.) * cpow(C, 1/2.) * S * cpow(Phi, -1.);
        }

        if(region == 4)
        {
            surface_flux = 0.6 * cpow(10., 26.) * (cpow(m_star, -2.) * mdot_star) * cpow(x, -6.) * cpow(B, -1.) * cpow(C,-1/2.) * Phi;
            surface_density = 2 * cpow(10., 5.) * cpow(alpha, -4/5.) * cpow(m_star, -1/2.) * cpow(mdot_star, 7/10.) * cpow(x, -3/2.) * cpow(A, 1/10.) * cpow(B, -4/5.) * cpow(C, 1/2.) * cpow(D, -17/20.) * cpow(S, -1/20.) * cpow(Phi, 7/10.);
        }

        //printf("At %i step %i r %f flux %f surface %f\n", region, steps, r / (2 * mass), surface_flux, surface_density);
    }

    accretion_disk disk;

    ///we use units of c=g=1

    assert(false);

    return disk;
}

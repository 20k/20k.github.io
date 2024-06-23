#include "accretion_disk.hpp"
#include <cmath>
#include <numbers>
#include <vec/vec.hpp>
#include <iostream>


double cpow(double a, double b)
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

///https://arxiv.org/pdf/1110.6556
accretion_disk make_accretion_disk_kerr(float M, float a)
{
    double Msol = 1.988 * cpow(10., 30.);

    double cst_G = 6.6743 * cpow(10., -11.);
    double cst_C = 299792458;

    double Msol_natural = Msol * cst_G / (cst_C*cst_C);

    double m_star = M / (3 * Msol_natural);

    ///sure why not
    double mdot_star = 0.01;

    double a_star = a/M;
    double assq = cpow(a_star, 2.);

    double isco = get_isco(M, a);

    double pi = std::numbers::pi_v<double>;

    double x1 = 2 * cos(acos(a_star)/3 - pi/3);
    double x2 = 2 * cos(acos(a_star)/3 + pi/3);
    double x3 = -2 * cos(acos(a_star)/3);

    double horizon = get_event_horizon(M, a);

    ///10 radii out??
    double outer_boundary = 2 * M * 10;

    ///0 = plunge, 1 = edge, 2 = inner, 3 = middle, 4 = outer
    int region = 0;

    int max_steps = 100;

    ///VISCOSITY
    double alpha = 0.1;

    for(int steps = 0; steps < max_steps; steps++)
    {
        double r = mix(horizon, outer_boundary, steps / (double)max_steps);

        if(region == 0 && r >= isco)
            region = 1;

        double x = sqrt(r/M);

        double x0 = sqrt(isco/M);
        double F0 = 1 - 2 * a_star * cpow(x0, -3.) + assq * cpow(x0, -4.);
        double G0 = 1 - 2 * cpow(x0, -2.) + a_star * cpow(x0, -3.);
        double C0 = 1 - 3 * cpow(x0, -2.) + 2 * assq * cpow(x0, -3.);
        double D0 = 1 - 2 * cpow(x0, -2.) + assq * cpow(x0, -4.);
        double V0 = cpow(D0, -1.) * (1 + cpow(x0, -4.) * (assq - cpow(x0, 2.) * cpow(F0, 2.) * cpow(G0, -2.)) + 2 * cpow(x, -6.) * (a_star - x0 * F0 * cpow(G0, -1.)));

        double A = 1 + assq * cpow(x, -4.) + 2 * assq * cpow(x, -6.);
        double B = 1 + a_star * cpow(x, -3.);
        double C = 1 - 3 * cpow(x, -2.) + 2 * assq * cpow(x, -3.);
        double D = 1 - 2 * cpow(x, -2.) + assq * cpow(x, -4.);
        double E = 1 + 4 * assq * cpow(x, -4.) - 4 * assq * cpow(x,-6.) + 3 * cpow(a_star, 4.) * cpow(x, -8.);
        double F = 1 - 2 * a_star * cpow(x, -3.) + assq * cpow(x, -4.);
        double G = 1 - 2 * cpow(x, -2.) + a_star * cpow(x, -4.);
        double H = 1 - 2 * cpow(x, -2.) + 2 * a_star * cpow(x, -2.) * cpow(x0, -1.) * cpow(F0, -1.) * G0;
        double I = A - 2 * a_star * cpow(x, -6.) * x0 * F0 * cpow(G0, -1.);
        double O = H * cpow(I, -1.);
        double J = O - cpow(x, -2.) * cpow(I, -1.) * (1 - a_star * cpow(x0, -1.) * cpow(F0, -1.) * G0 + assq * cpow(x, -2.) * H * cpow(J, -1.) * (1 + 3 * cpow(x, -2.) - 3 * cpow(a_star, -1.) * cpow(x, -2.) * x0 * F0 * cpow(G0, -1.)));
        double K = fabs(cpow(A * J * (1 - cpow(x, -4.) * cpow(A, 2.) * cpow(D, -1.) * cpow(x0 * F0 * cpow(G0, -1.) * O - 2 * a_star * cpow(x, -2.) * cpow(A, -1.), 2.)), -1.));
        double Q = B * cpow(C, -1/2.) * (1/x) * (x - x0 - 3/2. * a_star * log(x/x0)
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

        if(Tff_Tes >= 1)
            region = 4;

        double surface_flux = 0;
        double temperature = 0;
        double density = 0;

        if(region == 0)
        {
            //std::cout << "Vs? " << v_star << " in " << v_star_inner << std::endl;
            std::cout << "C " << C0 << " D " << D0 << " V " << V << " x " << x << std::endl;

            //erg/cm^2 sec
            surface_flux = 2 * cpow(10., 18.) * (cpow(alpha, 4/3.) * cpow(m_star, -3.) * cpow(mdot_star, 5/3.)) * cpow(x, -26/3.) * cpow(x0, 4/3.) * cpow(D, -5/6.) * cpow(K, 4/3.) * cpow(F0, 4/3.) * cpow(G0, -4/3.) * cpow(v_star, -5/3.);
        }

        if(region == 1 || region == 3)
        {
            surface_flux = 0.6 * cpow(10., 26.) * (cpow(m_star, -2.) * mdot_star) * cpow(x, -6.) * cpow(B, -1.) * cpow(C,-1/2.) * Phi;
        }

        if(region == 2)
        {
            surface_flux = 0.6 * cpow(10., 26.) * (cpow(m_star, -2.) * mdot_star) * cpow(x, -6.) * cpow(B, -1.) * cpow(C,-1/2.) * Phi;
        }

        if(region == 4)
        {
            surface_flux = 0.6 * cpow(10., 26.) * (cpow(m_star, -2.) * mdot_star) * cpow(x, -6.) * cpow(B, -1.) * cpow(C,-1/2.) * Phi;
        }

        printf("At %i r %f flux %f\n", region, r, surface_flux);
    }

    accretion_disk disk;

    ///we use units of c=g=1

    assert(false);

    return disk;
}

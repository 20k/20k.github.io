#include "accretion_disk.hpp"

double get_isco(double M, double a)
{
    double a_star = a/M;

    double Z1 = 1 + pow(1 - a_star*a_star, 1./3.) * (pow(1 + a_star, 1./3.) + pow(1 - a_star, 1./3.));
    double Z2 = sqrt(3 * a_star * a_star + Z1 * Z1);

    double r0_M = 3 + Z2 - sqrt((3 - Z1) * (3 + Z1 + 2 * Z2));

    return r0_M * M;
}

///https://arxiv.org/pdf/1110.6556
accretion_disk make_for_kerr(float M, float a)
{
    double Msol = 1.988 * pow(10., 30.);

    double G = 6.6743 * pow(10., -11.);
    double C = 299792458;

    double Msol_natural = Msol * G / (C*C);

    double M_star = M / (3 * Msol_natural);

    ///sure why not
    double Mdot_star = 0.01;

    double a_star = a/M;

    double isco = get_isco(M, a);

    double x1 =

    accretion_disk disk;

    ///we use units of c=g=1

    return disk;
}

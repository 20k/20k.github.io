#include "accretion_disk.hpp"
#include <cmath>
#include <numbers>

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

    double cst_G = 6.6743 * pow(10., -11.);
    double cst_C = 299792458;

    double Msol_natural = Msol * cst_G / (cst_C*cst_C);

    double M_star = M / (3 * Msol_natural);

    ///sure why not
    double Mdot_star = 0.01;

    double a_star = a/M;
    double assq = pow(a_star, 2);

    double isco = get_isco(M, a);

    double pi = std::numbers::pi_v<double>;

    double x1 = 2 * cos(acos(a_star)/3 - pi/3);
    double x2 = 2 * cos(acos(a_star)/3 + pi/3);
    double x3 = -2 * cos(acos(a_star)/3);

    double r = 0;

    double x = sqrt(r/M);

    double x0 = sqrt(isco/M);
    double F0 = 1 - 2 * a_star * pow(x0, -3) + assq * pow(x0, -4);
    double G0 = 1 - 2 * pow(x0, -2) + a_star * pow(x0, -3);

    double A = 1 + assq * pow(x, -4) + 2 * assq * pow(x, -6);
    double B = 1 + a_star * pow(x, -3);
    double C = 1 - 3 * pow(x, -2) + 2 * assq * pow(x, -3);
    double D = 1 - 2 * pow(x, -2) + assq * pow(a_star, -4);
    double E = 1 + 4 * assq * pow(x, -4) - 4 * assq * pow(x,-6) + 3 * pow(a_star, 4) * pow(x, -8);
    double F = 1 - 2 * a_star * pow(x, -3) + assq * pow(x, -4);
    double G = 1 - 2 * pow(x, -2) + a_star * pow(x, -4);
    double H = 1 - 2 * pow(x, -2) + 2 * a_star * pow(x, -2) * pow(x0, -1) * pow(F0, -1) * G0;
    double I = A - 2 * a_star * pow(x, -6) * x0 * F0 * pow(G0, -1);
    double O = H * pow(I, -1);
    double J = O - pow(x, -2) * pow(I, -1) * (1 - a_star * pow(x0, -1) * pow(F0, -1) * G0 + assq * pow(x, -2) * H * pow(J, -1) * (1 + 3 * pow(x, -2) - 3 * pow(a_star, -1) * pow(x, -2) * x0 * F0 * pow(G0, -1)));
    double K = fabs(pow(A * J * (1 - pow(x, -4) * pow(A, 2) * pow(D, -1) * pow(x0 * F0 * pow(G0, -1) * O - 2 * a_star * pow(x, -2) * pow(A, -1), 2)), -1));
    double Q = B * pow(C, -1/2.) * (1/x) * (x - x0 - 3/2. * a_star * log(x/x0)
                                            - (3 * pow(x1 - a_star, 2) / (x1 * (x1 - x2) * (x1 - x3))) * log((x - x1) / (x0 - x1))
                                            - (3 * pow(x2 - a_star, 2) / (x2 * (x2 - x1) * (x2 - x3))) * log((x - x2) / (x0 - x2))
                                            - (3 * pow(x3 - a_star, 2) / (x3 * (x3 - x1) * (x3 - x2))) * log((x - x3) / (x0 - x3))
                                            );
    double R = pow(F, 2) * pow(C, -1) - assq * pow(x, -2) * (G * pow(C, -1./2.) - 1);
    double S = pow(A, 2) * pow(B, -2) * C * pow(D, -1) * R;
    double V = pow(D, -1) * (1 + pow(x, -4) * (assq - pow(x0, 2) * pow(F0, 2) * pow(G0, -2)) + 2 * pow(x, -6) * (a_star - x0 * F0 * pow(G0, -1)));

    accretion_disk disk;

    ///we use units of c=g=1

    return disk;
}

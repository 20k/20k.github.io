#include "tov.hpp"
#include <cmath>
#define M_PI 3.14159265358979323846
#include <vec/vec.hpp>
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include "laplace.hpp"

using derivative_t = value<float16>;
using valuef = value<float>;
using valued = value<double>;
using valuei = value<int>;
using valueh = value<float16>;

using v2f = tensor<valuef, 2>;
using v3f = tensor<valuef, 3>;
using v4f = tensor<valuef, 4>;
using v2i = tensor<valuei, 2>;
using v3i = tensor<valuei, 3>;
using v4i = tensor<valuei, 4>;
using m44f = metric<valuef, 4, 4>;
using v3h = tensor<valueh, 3>;

using mut_v4f = tensor<mut<valuef>, 4>;
using mut_v3f = tensor<mut<valuef>, 3>;

///https://www.seas.upenn.edu/~amyers/NaturalUnits.pdf
//https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
double geometric_to_msol(double meters, double m_exponent)
{
    double m_to_kg = 1.3466 * std::pow(10., 27.);
    double msol_kg = 1.988416 * std::pow(10., 30.);
    double msol_meters = msol_kg / m_to_kg;

    return meters / pow(msol_meters, m_exponent);
}

double msol_to_geometric(double distance, double m_exponent)
{
    return geometric_to_msol(distance, -m_exponent);
}

double si_to_geometric(double quantity, double kg_exponent, double s_exponent)
{
    double G = 6.6743015 * pow(10., -11.);
    double C = 299792458;

    double factor = std::pow(G, -kg_exponent) * std::pow(C, 2 * kg_exponent - s_exponent);

    return quantity / factor;
}

double geometric_to_si(double quantity, double kg_exponent, double s_exponent)
{
    return si_to_geometric(quantity, -kg_exponent, -s_exponent);
}


double tov::parameters::rest_mass_density_to_pressure(double rest_mass_density) const
{
    return K * pow(rest_mass_density, Gamma);
}

double tov::parameters::rest_mass_density_to_energy_density(double rest_mass_density) const
{
    double p = rest_mass_density_to_pressure(rest_mass_density);

    double p0 = rest_mass_density;

    return p0 + p/(Gamma-1);
}

///inverse equation of state
///p -> p0
double tov::parameters::pressure_to_rest_mass_density(double p) const
{
    return std::pow(p/K, 1/Gamma);
}

///e = p0 + P/(Gamma-1)
double tov::parameters::pressure_to_energy_density(double p) const
{
    return pressure_to_rest_mass_density(p) + p / (Gamma - 1);
}

double tov::parameters::energy_density_to_pressure(double mu) const
{
    auto func = [&](double arg)
    {
        return rest_mass_density_to_energy_density(arg);
    };

    ///lets solve this numerically
    ///mu = p0 + P/(Gamma-1)
    double lower = 0;
    double upper = 1;

    while(func(upper) < mu)
        upper *= 2;

    for(int i=0; i < 1024; i++)
    {
        double lower_mu = func(lower);
        double upper_mu = func(upper);

        double next = (lower + upper)/2.;

        double next_mu = func(next);

        if(next_mu >= mu)
        {
            upper = next;
        }
        ///next_mu < mu
        else
        {
            lower = next;
        }
    }

    return (lower + upper)/2;
}

tov::integration_state tov::make_integration_state(double p0, double rmin, const parameters& param)
{
    double e = param.rest_mass_density_to_energy_density(p0);
    double m = (4./3.) * M_PI * e * std::pow(rmin, 3.);

    integration_state out;
    out.p = param.rest_mass_density_to_pressure(p0);
    out.m = m;
    return out;
}

//p0 in si units
tov::integration_state tov::make_integration_state_si(double p0, double rmin, const parameters& param)
{
    //kg/m^3 -> m/m^3 -> 1/m^2
    double p0_geom = si_to_geometric(p0, 1, 0);
    //m^-2 -> msol^-2
    double p0_msol = geometric_to_msol(p0_geom, -2);

    //std::cout << "density " << p0_msol << std::endl;

    return make_integration_state(p0_msol, rmin, param);
}

double tov::integration_solution::M_geom() const
{
    return msol_to_geometric(M_msol, 1);
}

double tov::integration_solution::R_geom() const
{
    return msol_to_geometric(R_msol, 1);
}

int tov::integration_solution::radius_to_index(double r) const
{
    assert(radius.size() > 0);

    if(r < radius.front())
        return 0;

    for(int i=1; i < radius.size(); i++)
    {
        if(r < radius[i])
            return i;
    }

    return radius.size() - 1;
}

struct integration_dr
{
    double dm = 0;
    double dp = 0;
};

integration_dr get_derivs(double r, const tov::integration_state& st, const tov::parameters& param)
{
    double e = param.pressure_to_energy_density(st.p);

    double p = st.p;
    double m = st.m;

    integration_dr out;

    out.dm = 4 * M_PI * e * std::pow(r, 2.);
    out.dp = -(e + p) * (m + 4 * M_PI * r*r*r * p) / (r * (r - 2 * m));
    return out;
}

///units are c=g=msol
tov::integration_solution tov::solve_tov(const integration_state& start, const parameters& param, double min_radius, double min_pressure)
{
    integration_state st = start;

    double current_r = min_radius;
    double dr = 1. / 1024.;

    integration_solution sol;

    double last_r = 0;
    double last_m = 0;

    while(1)
    {
        sol.energy_density.push_back(param.pressure_to_energy_density(st.p));
        sol.pressure.push_back(st.p);
        sol.cumulative_mass.push_back(st.m);

        double r = current_r;

        sol.radius.push_back(r);

        last_r = r;
        last_m = st.m;

        integration_dr data = get_derivs(r, st, param);

        st.m += data.dm * dr;
        st.p += data.dp * dr;
        current_r += dr;

        if(!std::isfinite(st.m) || !std::isfinite(st.p))
            break;

        if(st.p <= min_pressure)
            break;
    }

    sol.R_msol = last_r;
    sol.M_msol = last_m;

    return sol;
}

//personally i liked the voyage home better
std::vector<double> tov::search_for_rest_mass(double adm_mass, const parameters& param)
{
    double r_approx = adm_mass / 0.06;

    double start_E = adm_mass / ((4./3.) * M_PI * r_approx*r_approx*r_approx);
    double start_P = param.energy_density_to_pressure(start_E);
    double start_density = param.pressure_to_rest_mass_density(start_P);

    double rmin = 1e-6;

    std::vector<double> densities;
    std::vector<double> masses;

    int to_check = 2000;
    densities.resize(to_check);
    masses.resize(to_check);

    double min_density = start_density / 100;
    double max_density = start_density * 200;

    for(int i=0; i < to_check; i++)
    {
        double frac = (double)i / to_check;

        double test_density = mix(min_density, max_density, frac);

        integration_state next_st = make_integration_state(test_density, rmin, param);
        integration_solution next_sol = solve_tov(next_st, param, rmin, 0.);

        densities[i] = test_density;
        masses[i] = next_sol.M_msol;
    }

    std::vector<double> out;

    for(int i=0; i < to_check - 1; i++)
    {
        double current = masses[i];
        double next = masses[i+1];

        double min_mass = std::min(current, next);
        double max_mass = std::max(current, next);

        if(adm_mass >= min_mass && adm_mass < max_mass)
        {
            double frac = (adm_mass - min_mass) / (max_mass - min_mass);

            out.push_back(mix(densities[i], densities[i+1], frac));
        }
    }

    return out;
}

std::vector<double> initial::calculate_isotropic_r(const tov::integration_solution& sol)
{
    std::vector<double> dlog_dr;
    dlog_dr.reserve(sol.cumulative_mass.size());

    for(int i=0; i < (int)sol.cumulative_mass.size(); i++)
    {
        double r = sol.radius[i];
        double m = sol.cumulative_mass[i];

        double rhs = (1 - sqrtf(1 - 2 * m/r)) / (r * sqrt(1 - 2 * m/r));

        //double rhs = (pow(r, 0.5) - pow(r - 2 * m, 0.5)) / (r * pow(r - 2 * m, 0.5));
        dlog_dr.push_back(rhs);
    }

    std::vector<double> r_hat;
    double last_r = 0;
    double log_rhat_r = 0;

    for(int i=0; i < (int)sol.radius.size(); i++)
    {
        double r = sol.radius[i];

        double dr = r - last_r;

        log_rhat_r += dr * dlog_dr[i];

        //std::cout << "step size " << dr * dlog_dr[i] << std::endl;

        double lr_hat = exp(log_rhat_r);

        r_hat.push_back(lr_hat);

        last_r = r;
    }

    {
        double final_r = r_hat.back();

        double R = sol.radius.back();
        double M = sol.cumulative_mass.back();

        double scale = (1/(2*R)) * (sqrt(R*R - 2 * M * R) + R - M) / final_r;

        for(int i=0; i < (int)sol.radius.size(); i++)
        {
            r_hat[i] *= sol.radius[i] * scale;
        }
    }

    return r_hat;
}

std::vector<double> initial::calculate_tov_phi(const tov::integration_solution& sol)
{
    auto isotropic_r = calculate_isotropic_r(sol);

    auto isotropic_to_schwarzschild = [&](auto isotropic_in)
    {
        return tov::interpolate_by_radius(isotropic_r, sol.radius, isotropic_in);
    };

    int samples = sol.radius.size();

    std::vector<double> phi;
    phi.resize(samples);

    for(int i=0; i < (int)sol.radius.size(); i++)
    {
        phi[i] = pow(sol.radius[i] / isotropic_r[i], 1./2.);
    }

    #if 0
    double check_mass = 0;
    double last_r_bar = 0;

    for(int i=0; i < (int)samples; i++)
    {
        double r_bar = isotropic_r[i];

        double r = isotropic_to_schwarzschild(r_bar);
        double e = tov::interpolate_by_radius(sol.radius, sol.energy_density, r);

        check_mass += 4 * M_PI * r_bar*r_bar * pow(phi[i], 5.) * e * (r_bar - last_r_bar);

        last_r_bar = r_bar;
    }

    std::cout << "Check Mass " << check_mass << std::endl;
    #endif

    return phi;
}

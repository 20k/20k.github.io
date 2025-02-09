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

double tov::parameters::energy_density_to_pressure(double e) const
{
    ///e = p0 + P/(Gamma-1)
    ///e = (P/K)^(1/Gamma) + P/(Gamma-1)
    ///P = (1-1/g)th root of ((1-G)K^(-1/G)
    return std::pow((1-Gamma) * std::pow(K, -1/Gamma), 1/(1-(1/Gamma)));
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

    sol.R = msol_to_geometric(last_r, 1);
    sol.M = last_m;

    sol.R_msol = last_r;
    sol.M_msol = last_m;

    sol.R_geometric = msol_to_geometric(last_r, 1);
    sol.M_geometric = msol_to_geometric(last_m, 1);

    return sol;
}

//personally i liked the voyage home better
std::vector<double> tov::search_for_adm_mass(double adm_mass, const parameters& param)
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
    double max_density = start_density * 5;

    for(int i=0; i < to_check; i++)
    {
        double frac = (double)i / to_check;

        double test_density = mix(min_density, max_density, frac);

        integration_state next_st = make_integration_state(test_density, rmin, param);
        integration_solution next_sol = solve_tov(next_st, param, rmin, 0.);

        densities[i] = test_density;
        masses[i] = next_sol.M;
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

struct tov_data : value_impl::single_source::argument_pack
{
    //linear in terms of radius, padded so that at cell max, our radius is max rad
    buffer<valuef> epsilon;
    literal<valuef> max_rad;

    void build(auto& in)
    {
        using namespace value_impl::builder;

        add(epsilon, in);
        add(max_rad, in);
    }
};

template<typename T>
T sdiff(const std::vector<T>& F, int x, T scale)
{
    if(x == 0)
    {
        return (-F[0] + F[1]) / scale;
    }

    if(x == F.size() - 1)
    {
        return (-F[x - 1] + F[x]) / scale;
    }

    return (F[x + 1] - F[x-1]) / (2 * scale);
}

template<typename T>
T next_guess(const std::vector<T>& F, int x, T A, T B, T C, T h)
{
    if(x == 0)
    {
        return -(A * F[x + 2] - 2 * A * F[x + 1] + B * h * F[x + 1]) / (A - B * h + C*h*h);
    }

    if(x == (int)F.size() - 1)
    {
        return (2 * A * F[x - 1] - A * F[x - 2] + B * h * F[x-1]) / (A + B * h + C * h * h);
    }

    return (2 * A * (F[x - 1] + F[x + 1]) + B * h * (F[x + 1] - F[x - 1])) / (4 * A - 2 * C * h * h);
}

template<typename T>
T interpolate_by_radius(const std::vector<double>& radius, const std::vector<T>& quantity, double r)
{
    assert(radius.size() >= 2);

    if(r <= radius.front())
        return quantity.front();

    if(r >= radius.back())
        return quantity.back();

    for(int i=0; i < (int)radius.size() - 1; i++)
    {
        double r1 = radius[i];
        double r2 = radius[i + 1];

        if(r > r1 && r <= r2)
        {
            double frac = (r - r1) / (r2 - r1);

            return mix(quantity[i], quantity[i + 1], frac);
        }
    }

    return quantity.back();
}

std::vector<double> initial::calculate_tov_phi(const tov::integration_solution& sol)
{
    std::vector<double> dlog_dr;
    dlog_dr.reserve(sol.cumulative_mass.size());

    for(int i=0; i < (int)sol.cumulative_mass.size(); i++)
    {
        double r = sol.radius[i];
        double m = sol.cumulative_mass[i];

        double rhs = (pow(r, 0.5) - pow(r - 2 * m, 0.5)) / (r * pow(r - 2 * m, 0.5));
        dlog_dr.push_back(rhs);
    }

    std::vector<double> r_hat;
    double r_dot_root = 0;
    double last_r = 0;
    double log_rhat_r = 0;

    for(int i=0; i < (int)sol.radius.size(); i++)
    {
        double r = sol.radius[i];

        double dr = r - last_r;

        log_rhat_r += dr * dlog_dr[i];

        double lr_hat = exp(log_rhat_r);

        r_hat.push_back(lr_hat * r);

        last_r = r;
    }

    auto isotropic_to_schwarzschild = [&](float isotropic_in)
    {
        return interpolate_by_radius(r_hat, sol.radius, isotropic_in);
    };

    /*for(auto& i : sol.radius)
    {
        std::cout << "hello "  << i << std::endl;
    }

    for(auto& i : r_hat)
    {
        std::cout << i << " real " << isotropic_to_schwarzschild(i) << std::endl;
    }*/

    /*for(auto& i : r_hat)
    {
        std::cout << "r_hat " << i << std::endl;
    }*/

    std::vector<double> ret;

    #if 0
    auto diff = [](const std::vector<double>& buf, const std::vector<double>& radius, int idx)
    {
        if(idx == buf.size() - 1)
        {
            double h = radius[idx] - radius[idx - 1];
            return (buf[idx] - buf[idx - 1]) / h;
        }

        double h = radius[idx + 1] - radius[idx];
        return (buf[idx + 1] - buf[idx]) / h;
    };

    ret.reserve(sol.energy_density.size());

    for(int i=0; i < (int)sol.mass.size(); i++)
    {
        double dm_dr = diff(sol.mass, sol.radius, i);
        double r = sol.radius[i];
        double e = sol.energy_density[i];

        std::cout << "dm_dr " << dm_dr << std::endl;

        std::cout << "e? " << e << std::endl;;

        double phi_5 = dm_dr / (2 * M_PI * r*r * e);

        double phi = pow(phi_5, 1/5.);

        std::cout << "phi " << phi << std::endl;

        ret[i] = phi;
    }
    ///make sure i handle the integration constants
    #endif // 0

    ///todo: it cannot possibly be right that the answer is phi = 2^(1/5)

    #if 0
    for(int i=0; i < (int)sol.mass.size(); i++)
    {
        ret.push_back(pow(2., 1/5.));
    }

    //ok this is officially insane and i'm very confused
    //is the distribution of phi wrong here?
    double check_mass = 0;

    for(int i=0; i < (int)sol.mass.size() - 1; i++)
    {
        double r = sol.radius[i];
        double e = sol.energy_density[i];

        check_mass += 2 * M_PI * r*r * pow(ret[i], 5.) * e * (sol.radius[i + 1] - sol.radius[i]);
    }

    std::cout << "Check Mass " << check_mass << std::endl;
    #endif

    assert(sol.cumulative_mass.size() == sol.energy_density.size());

    std::vector<float> linearised_epsilon;
    int samples = sol.energy_density.size();

    float min_r = 0;
    float max_r = sol.R_msol;

    for(int i=0; i < samples; i++)
    {
        float dr = (max_r - min_r) / samples;
        double r = i * dr;

        int e1 = sol.radius_to_index(r);
        int e2 = sol.radius_to_index(r + dr);

        float en1 = sol.energy_density.at(e1);
        float en2 = sol.energy_density.at(e2);

        float r1 = sol.radius.at(e1);
        float r2 = sol.radius.at(e2);

        float frac = 0;

        if(fabs(r1 - r2) > 0.0001f)
            frac = (r - r1) / (r2 - r1);

        linearised_epsilon.push_back(mix(en1, en2, frac));
    }

    //linearised_epsilon.push_back(0);

    double scale = max_r / samples;

    std::vector<double> phi_current;
    std::vector<double> phi_next;

    phi_current.resize(linearised_epsilon.size());
    phi_next.resize(linearised_epsilon.size());

    //for(auto& i : phi_current)
    //    i = 1;

    //phi_current.push_back(1.f);
    //phi_next.push_back(1.f);

    for(int i=0; i < (int)phi_current.size(); i++)
    {
        phi_current[i] = 1;

        //phi_current[i] = (1 - (float)i / phi_current.size()) + 0.1f;
    }

    for(int i=0; i < 10240; i++)
    {
        for(int kk=0; kk < samples; kk++)
        {
            double phi = phi_current[kk];
            double e = linearised_epsilon[kk];
            double r = ((double)kk / samples) * sol.R_msol;

            //std::cout << "e " << e << std::endl;

            //std::cout << "r " << r << std::endl;

            //std::cout << "Pcurrent " << phi_current[kk] << " r " << r << " rhs " << 2 * M_PI * r * std::pow(phi, 4.) * e << " scale " << scale << std::endl;
            //double next_phi = next_guess<double>(phi_current, kk, r, 2, 2 * M_PI * r * std::pow(phi, 4.) * e, scale);

            ///(1, -2, 1) phi = h^2 RHS
            ///-2 phi = h^2 RHS - (phi[-1] + phi[1])
            ///phi = (h^2 RHS - (phi[-1] + phi[1])) / -2

            double rhs = -((sdiff(phi_current, kk, scale) * 2/std::max(r, 0.001)) + 2 * M_PI * pow(phi, 5.) * e);

            double d = 0;

            if(kk == 0)
                d = phi_current[2] + phi_current[0];

            else if(kk == samples - 1)
                d = phi_current[kk] + phi_current[kk - 2];

            else
                d = phi_current[kk + 1] + phi_current[kk - 1];

            double next_phi = (scale*scale * rhs - d) / -2;

            //if(kk == 50)
            //std::cout << "Nphi " << next_phi << std::endl;

            //if(kk == samples - 2 && (i % 1000) == 0)
            //s    std::cout << "Nphi " << next_phi << std::endl;

            phi_next[kk] = mix(phi, next_phi, 0.99);
        }

        std::swap(phi_current, phi_next);
    }

    /*for(auto& i : phi_current)
    {
        std::cout << "Phi " << i << std::endl;
    }*/

    double check_mass = 0;

    for(int i=0; i < (int)samples - 1; i++)
    {
        double r = ((double)i / samples) * sol.R_msol;
        double e = linearised_epsilon[i];

        check_mass += 2 * M_PI * r*r * pow(phi_current[i], 5.) * e * (sol.radius[i + 1] - sol.radius[i]);
    }

    double phi_at_radius = phi_current.back();

    double new_phi = 1 + check_mass / (2 * sol.R_msol);

    double diff = (new_phi / phi_at_radius);

    std::cout << "diff " << diff << std::endl;

    for(auto& i : phi_current)
        i *= diff;

    check_mass = 0;

    for(int i=0; i < (int)samples - 1; i++)
    {
        double r = ((double)i / samples) * sol.R_msol;
        double e = linearised_epsilon[i];

        check_mass += 2 * M_PI * r*r * pow(phi_current[i], 5.) * e * (sol.radius[i + 1] - sol.radius[i]);
    }

    std::cout << "Check Mass " << check_mass << std::endl;

    //T next_phi = next_guess<T>(phi, kk, r, 2, 2 * M_PI * r * std::pow(phi[kk], T{4.f}) * rho, scale);

    return phi_current;
}

#if 0
struct tov_pack
{
    cl::buffer epsilon;
    float max_rad = 0;

    tov_pack(cl::context ctx) : epsilon(ctx){}

    void push(cl::args& args)
    {
        args.push_back(epsilon);
        args.push_back(max_rad);
    }
};

cl::buffer initial::tov_solve_full_grid(cl::context ctx, cl::command_queue cqueue, float scale, t3i dim, const initial::neutron_star& star)
{
    std::vector<float> linearised_epsilon;
    int samples = 100;

    float min_r = 0;
    float max_r = star.sol.R_msol;

    for(int i=0; i < samples; i++)
    {
        float dr = (max_r - min_r) / samples;
        double r = i * dr;

        int e1 = star.sol.radius_to_index(r);
        int e2 = star.sol.radius_to_index(r + dr);

        float en1 = star.sol.energy_density.at(e1);
        float en2 = star.sol.energy_density.at(e2);

        float r1 = star.sol.radius.at(e1);
        float r2 = star.sol.radius.at(e2);

        float frac = 0;

        if(fabs(r1 - r2) > 0.0001f)
            frac = (r - r1) / (r2 - r1);

        linearised_epsilon.push_back(mix(en1, en2, frac));
    }

    cl::buffer epsilon(ctx);
    epsilon.alloc(sizeof(cl_float) * samples);
    epsilon.write(cqueue, linearised_epsilon);

    auto get_data = [&](t3i idim, float iscale) {
        tov_pack pack(ctx);
        pack.epsilon = epsilon;
        pack.max_rad = max_r;

        return pack;
    };

    auto get_rhs = [&](laplace_params params, tov_data data)
    {
        v3i centre = (params.dim-1)/2;
        auto u = params.u;
        v3i pos = params.pos;
        v3i dim = params.dim;

        using namespace single_source;

        valuef phi = u[pos, dim];
        pin(phi);

        //return -2 * M_PI * pow(phi, valuef{5}) *
    };
}
#endif

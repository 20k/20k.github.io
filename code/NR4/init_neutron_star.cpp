#include "init_neutron_star.hpp"
#include "tov.hpp"

template<typename T, typename U>
inline
auto integrate_1d(const T& func, int n, const U& upper, const U& lower)
{
    using variable_type = decltype(func(0.f));

    variable_type sum = 0;

    for(int k=1; k < n; k++)
    {
        auto coordinate = lower + k * (upper - lower) / n;

        auto val = func(coordinate);

        sum += val;
    }

    return ((upper - lower) / n) * (func(lower)/2.f + sum + func(upper)/2.f);
}

neutron_star::solution neutron_star::solve(const tov::integration_solution& sol)
{
    std::vector<double> radius_iso = initial::calculate_isotropic_r(sol);
    ///hang on. can i literally just treat the schwarzschild data like its in isotropic?
    ///I think because of the way its laid out: yes

    std::vector<double> tov_phi_iso = initial::calculate_tov_phi(sol);

    int samples = sol.energy_density.size();

    std::vector<double> mu_cfl;
    std::vector<double> pressure_cfl;

    mu_cfl.reserve(samples);
    pressure_cfl.reserve(samples);

    for(int i=0; i < samples; i++)
    {
        double tov_8 = pow(tov_phi_iso[i], 8.);

        mu_cfl.push_back(tov_8 * sol.energy_density[i]);
        pressure_cfl.push_back(tov_8 * sol.pressure[i]);
    }

    ///integrates in the half open range [0, index)
    auto integrate_to_index = [&](auto&& func, int index)
    {
        assert(index >= 0 && index < samples);

        double last_r = 0;
        std::vector<double> out;
        out.reserve(index);

        double current = 0;

        for(int i=0; i < index; i++)
        {
            double r = radius_iso[i];
            double dr = (r - last_r);

            current += func(i) * dr;

            out.push_back(current);

            last_r = r;
        }

        assert(out.size() == index);

        return out;
    };

    double M = 4 * M_PI * integrate_to_index([&](int idx)
    {
        return mu_cfl[idx] * pressure_cfl[idx] * pow(radius_iso[idx], 2.);
    }, samples).back();

    std::vector<double> sigma;
    sigma.reserve(samples);

    for(int i=0; i < samples; i++)
    {
        sigma.push_back((mu_cfl[i] + pressure_cfl[i]) / M);
    }

    std::vector<double> Q = integrate_to_index([&](int idx)
    {
        double r = radius_iso[idx];

        return 4 * M_PI * sigma[idx] * r*r;
    }, samples);

    std::vector<double> C = integrate_to_index([&](int idx)
    {
        double r = radius_iso[idx];

        return (2./3.) * M_PI * sigma[idx] * pow(r, 4.);
    }, samples);
}

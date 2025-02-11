#include "init_neutron_star.hpp"
#include "tov.hpp"
#include "../common/vec/tensor.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include <toolkit/opencl.hpp>

using valuef = value<float>;
using valuei = value<int>;

using v2f = tensor<valuef, 2>;
using v3f = tensor<valuef, 3>;
using v4f = tensor<valuef, 4>;
using v2i = tensor<valuei, 2>;
using v3i = tensor<valuei, 3>;
using v4i = tensor<valuei, 4>;
using m44f = metric<valuef, 4, 4>;

using mut_v4f = tensor<mut<valuef>, 4>;
using mut_v3f = tensor<mut<valuef>, 3>;

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

void neutron_star::add_to_solution(cl::context& ctx, cl::command_queue& cqueue,
                                   discretised_solution& dsol, const params& phys_params, const tov::integration_solution& sol,
                                   tensor<int, 3> dim, float scale)
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
        double r = radius_iso[idx];

        return (mu_cfl[idx] + pressure_cfl[idx]) * pow(r, 2.);
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

    double squiggly_N = (8*M_PI/3.) * integrate_to_index([&](int idx)
    {
        double r = radius_iso[idx];

        return (mu_cfl[idx] + pressure_cfl[idx]) * pow(r, 4.);
    }, samples).back();

    std::vector<double> kappa;
    kappa.reserve(samples);

    for(int i=0; i < samples; i++)
    {
        kappa.push_back((mu_cfl[i] + pressure_cfl[i]) / squiggly_N);
    }

    std::vector<double> unsquiggly_N = integrate_to_index([&](int idx)
    {
        double r = radius_iso[idx];

        return (8*M_PI/3.) * kappa[idx] * pow(r, 4.);
    }, samples);

    auto d2f = [&](const std::vector<double>& in)
    {
        std::vector<float> f;
        f.reserve(in.size());

        for(auto& i : in)
            f.push_back(i);

        cl::buffer buf(ctx);
        buf.alloc(sizeof(cl_float) * in.size());
        buf.write(cqueue, f);
        return buf;
    };

    cl::buffer cl_Q = d2f(Q);
    cl::buffer cl_C = d2f(C);
    cl::buffer cl_uN = d2f(unsquiggly_N);

    cl::buffer cl_sigma = d2f(sigma);
    cl::buffer cl_kappa = d2f(kappa);

    cl::buffer cl_mu_cfl = d2f(mu_cfl);
    cl::buffer cl_pressure_cfl = d2f(pressure_cfl);
    cl::buffer cl_radius = d2f(radius_iso);

    auto accum = [](execution_context& ctx, buffer<valuef> Q, buffer<valuef> C, buffer<valuef> uN,
                    buffer<valuef> sigma, buffer<valuef> kappa,
                    buffer<valuef> mu_cfl, buffer<valuef> pressure_cfl, buffer<valuef> radius,
                    std::array<buffer_mut<valuef>, 6> AIJ_out, buffer_mut<valuef> mu_cfl_out, buffer_mut<valuef> mu_h_out, buffer_mut<valuef> pressure_cfl_out)
    {

    };


    /*neutron_star::solution ret;
    ret.N = unsquiggly_N;
    ret.Q = Q;
    ret.C = C;
    ret.mu_cfl = mu_cfl;
    ret.pressure_cfl = pressure_cfl;

    return ret;*/
}

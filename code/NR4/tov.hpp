#ifndef TOV_HPP_INCLUDED
#define TOV_HPP_INCLUDED

#include <vector>
#include <vec/tensor.hpp>
#include <toolkit/opencl.hpp>

using t3i = tensor<int, 3>;
using t3f = tensor<float, 3>;

double geometric_to_msol(double meters, double m_exponent);
double msol_to_geometric(double distance, double m_exponent);
double si_to_geometric(double quantity, double kg_exponent, double s_exponent);
double geometric_to_si(double quantity, double kg_exponent, double s_exponent);

namespace tov
{
    ///https://colab.research.google.com/drive/1yMD2j3Y6TcsykCI59YWiW9WAMW-SPf12#scrollTo=6vWjt7CWaVyV
    ///https://www.as.utexas.edu/astronomy/education/spring13/bromm/secure/TOC_Supplement.pdf
    ///https://arxiv.org/pdf/gr-qc/0403029
    struct parameters
    {
        double K = 0;
        double Gamma = 0;

        ///p0 -> p
        ///equation of state
        double rest_mass_density_to_pressure(double rest_mass_density) const;

        ///p0 -> e
        double rest_mass_density_to_energy_density(double rest_mass_density) const;

        ///inverse equation of state
        ///p -> p0
        double pressure_to_rest_mass_density(double p) const;

        ///e = p0 + P/(Gamma-1)
        double pressure_to_energy_density(double p) const;

        double energy_density_to_pressure(double e) const;
    };

    struct integration_state
    {
        double m = 0;
        double p = 0;
    };

    integration_state make_integration_state(double p0, double min_radius, const parameters& param);
    integration_state make_integration_state_si(double p0, double min_radius, const parameters& param);

    struct integration_solution
    {
        //in msols
        double M = 0;
        //in meters
        double R = 0;

        ///keeping these as stable api endpoints
        ///todo: fixme, this code is crap
        double M_msol = 0;
        double R_msol = 0;
        double M_geometric = 0;
        double R_geometric = 0;

        std::vector<double> energy_density;
        std::vector<double> pressure;
        std::vector<double> mass;
        std::vector<double> radius;

        int radius_to_index(double r) const;
    };

    integration_solution solve_tov(const integration_state& start, const parameters& param, double min_radius, double min_pressure);
    std::vector<double> search_for_adm_mass(double adm_mass, const parameters& param);
}

namespace initial
{
    std::vector<double> calculate_tov_phi(const tov::integration_solution& sol);

    struct neutron_star
    {
        t3f world_pos;

        tov::integration_solution sol;
    };

    cl::buffer tov_solve_full_grid(cl::context ctx, cl::command_queue cqueue, float scale, t3i dim, const neutron_star& star);
}

#endif // TOV_HPP_INCLUDED

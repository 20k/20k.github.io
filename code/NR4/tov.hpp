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
    template<typename T>
    inline
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
        double M_msol = 0;
        double R_msol = 0;

        std::vector<double> energy_density;
        std::vector<double> pressure;
        std::vector<double> cumulative_mass;
        std::vector<double> radius; //in schwarzschild coordinates, in units of c=G=mSol = 1

        int radius_to_index(double r) const;

        double M_geom();
        double R_geom();
    };

    integration_solution solve_tov(const integration_state& start, const parameters& param, double min_radius, double min_pressure);
    std::vector<double> search_for_adm_mass(double adm_mass, const parameters& param);
}

namespace initial
{
    std::vector<double> calculate_isotropic_r(const tov::integration_solution& sol);
    std::vector<double> calculate_tov_phi(const tov::integration_solution& sol);
}

#endif // TOV_HPP_INCLUDED

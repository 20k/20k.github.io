#include "galaxy_model.hpp"
#include <cmath>
#include "random.hpp"
#include <libtov/tov.hpp>

std::optional<double> select_from_cdf(double frac, auto cdf)
{
    if(frac > cdf(1.))
        return std::nullopt;

    double next_upper = 1;
    double next_lower = 0;

    for(int i=0; i < 500; i++)
    {
        double test_val = (next_upper + next_lower)/2.f;

        double found_val = cdf(test_val);

        if(found_val < frac)
        {
            next_lower = test_val;
        }
        else if(found_val > frac)
        {
            next_upper = test_val;
        }
        else
        {
            return test_val;
        }
    }

    return (next_upper + next_lower)/2.f;
}

double get_c()
{
    return 299792458;
}

///m3 kg-1 s-2
double get_G()
{
    return 6.67430 * std::pow(10., -11.);
}

struct disk_distribution
{
    double M0 = 0;
    double a_frac = 0.01;
    double max_R = 0;

    double normalised_cdf(double fraction)
    {
        double a = a_frac * max_R;

        double r = fraction * max_R;

        return 1 - a / sqrt(r*r + a*a);
    }

    //radius -> velocity
    double fraction_to_velocity(double fraction)
    {
        double R = fraction * max_R;
        double a = a_frac * max_R;

        double G = get_G();

        return std::sqrt((G * M0 * R*R) / pow(R*R + a*a, 3./2.));
    }

    std::optional<double> select_radial_frac(xoshiro256ss_state& rng)
    {
        auto lambda_cdf = [&](double r)
        {
            return normalised_cdf(r);
        };

        double random = uint64_to_double(xoshiro256ss(rng));

        //printf("Random %f\n", random);

        auto result = select_from_cdf(random, lambda_cdf);

        //if(result)
        //    printf("Got res %f\n", result.value());

        return result;
    }
};

double get_solar_mass_kg()
{
    return 1.988416 * std::pow(10., 30.);
}

galaxy_data build_galaxy(float fill_width)
{
    double fill_radius = fill_width/2;

    galaxy_data dat;
    xoshiro256ss_state rng = xoshiro256ss_init(2345);

    int try_num = 1000 * 1000;
    int real_num = 0;

    double milky_way_mass_kg = 2 * pow(10, 12) * get_solar_mass_kg();
    //double milky_way_mass_kg = 4 * pow(10, 11) * get_solar_mass_kg();
    double milky_way_radius_m = 0.5 * 8 * pow(10., 20.);

    disk_distribution disk;
    disk.M0 = milky_way_mass_kg;
    disk.max_R = milky_way_radius_m;

    for(int i=0; i < try_num; i++)
    {
        auto radius_opt = disk.select_radial_frac(rng);

        if(!radius_opt)
            continue;

        double radius = radius_opt.value() * disk.max_R;

        if(radius > disk.max_R)
            continue;

        real_num++;

        double velocity = disk.fraction_to_velocity(radius / disk.max_R);

        double velocity_geom = velocity / get_c();

        double angle = uint64_to_double(xoshiro256ss(rng)) * 2 * std::numbers::pi_v<float>;

        double radius_real = (radius / disk.max_R) * fill_radius;

        t3f vel = {velocity_geom * cos(angle), velocity_geom * sin(angle), 0.f};
        t3f pos = {radius_real * cos(angle), radius_real * sin(angle), 0.f};

        dat.positions.push_back(pos);
        dat.velocities.push_back(vel);
    }

    double mass_m = si_to_geometric(milky_way_mass_kg, 1, 0);
    double mass_real = (mass_m / disk.max_R) * fill_radius;

    printf("Found mass %f\n", mass_real);

    double mass_per_particle = mass_real / real_num;

    for(auto& i : dat.positions)
        dat.masses.push_back(mass_per_particle);

    return dat;
}

#if 0
///https://arxiv.org/pdf/1705.04131.pdf 28
/*float matter_cdf(float m0, float r0, float rc, float r, float B = 1)
{
    return m0 * pow(sqrt(r0/rc) * r/(r + rc), 3 * B);
}*/
float select_from_cdf(float value_mx, float max_radius, auto cdf)
{
    float value_at_max = cdf(max_radius);

    if(value_mx >= value_at_max)
        return max_radius * 1000;

    float next_upper = max_radius;
    float next_lower = 0;

    for(int i=0; i < 50; i++)
    {
        float test_val = (next_upper + next_lower)/2.f;

        float found_val = cdf(test_val);

        if(found_val < value_mx)
        {
            next_lower = test_val;
        }
        else if(found_val > value_mx)
        {
            next_upper = test_val;
        }
        else
        {
            return test_val;
        }
    }

    //printf("Looking for %.14f found %.14f with y = %.14f\n", scaled, (next_upper + next_lower)/2.f, cdf((next_upper + next_lower)/2.f));

    return (next_upper + next_lower)/2.f;
}

double get_solar_mass_kg()
{
    return 1.988416 * std::pow(10., 30.);
}

///m/s
double get_c()
{
    return 299792458;
}

///m3 kg-1 s-2
double get_G()
{
    return 6.67430 * std::pow(10., -11.);
}

struct galaxy_params
{
    double mass_kg = 0;
    double radius_m = 0;
};

struct disk_distribution
{
    bool is_disk = true;

    double a = 3;

    double cdf(double M0, double G, double r)
    {
        /*auto surface_density = [this, M0](double r)
        {
            return (M0 * a / (2 * M_PI)) * 1./std::pow(r*r + a*a, 3./2.);

            //return (M0 / (2 * M_PI * a * a)) / pow(1 + r*r/a*a, 3./2.);
        };

        auto integral = [&surface_density](double r)
        {
            return 2 * M_PI * r * surface_density(r);
        };

        return integrate_1d(integral, 64, r, 0.);*/

        return M0 * (1 - a / sqrt(r*r + a*a));
    }

    ///https://galaxiesbook.org/chapters/II-01.-Flattened-Mass-Distributions.html 8.17 implies that kuzmin uses M0, not the CDF
    double get_velocity_at(double M0, double G, double r)
    {
        return std::sqrt(G * M0 * r * r * pow(r * r + a * a, -3./2.));
        //return std::sqrt(G * cdf(M0, G, r) * r * r * pow(r * r + a * a, -3./2.));
    }

    ///https://adsabs.harvard.edu/full/1995MNRAS.276..453B
    double get_mond_velocity_at(double M0, double G, double r, double a0)
    {
        ///they use h for a
        double u = r / a;

        double v_inf = pow(M0 * G * a0, 1./4.);

        ///I've seen this before but I can't remember what it is
        double squiggle = M0 * G / (a*a * a0);

        double p1 = v_inf * v_inf;

        double p2 = u*u / (1 + u*u);

        double divisor_part = 1 + u*u;

        double interior_1 = sqrt(1 + squiggle*squiggle / (4 * pow(divisor_part, 2)));

        double interior_2 = squiggle / (2 * divisor_part);

        return sqrt(p1 * p2 * sqrt(interior_1 + interior_2));
    }
};

/*struct spherical_distribution
{
    bool is_disk = false;

    double cdf(double M0, double G, double r)
    {
        auto density = [M0](double r)
        {
            double r0 = 1;
            double rc = 1;
            double B = 1;

            return
        };
    }
};*/

///Todo: I've totally messed up the units as I expected

///This is not geometric units, this is scale independent
template<typename T>
struct galaxy_distribution
{
    double mass = 0;
    double max_radius = 0;

    double local_G = 0;
    double meters_to_local = 0;
    double kg_to_local = 0;

    T distribution;

    double local_distance_to_meters(double r)
    {
        return r / meters_to_local;
    }

    double local_mass_to_kg(double m)
    {
        return m / kg_to_local;
    }

    ///M(r)
    //problem 2: we're cutting off the CDF quite early
    double cdf(double r)
    {
        ///correct for cumulative sphere model
        /*auto p2 = [&](float r)
        {
            return 4 * M_PI * r * r * surface_density(r);
        };*/

        ///correct for surface density
        /*auto p3 = [&](double r)
        {
            return 2 * M_PI * r * surface_density(r);
        };

        return integrate_1d(p3, 64, r, 0.);*/

        return distribution.cdf(mass, local_G, r);
    };

    double get_velocity_at(double r)
    {
        /*double a0_ms2 = 1.2 * pow(10., -10.);

        double a0 = a0_ms2 * meters_to_local;

        double p1 = local_G * cdf(r)/r;

        double p2 = (1/sqrt(2.f));

        double frac = 2 * a0 / (local_G * cdf(r));

        double p_inner = 1 + sqrt(1 + pow(r, 4.f) * pow(frac, 2.f));

        double p3 = sqrt(p_inner);

        return std::sqrt(p1 * p2 * p3);*/

        //return std::sqrt(local_G * cdf(r) / r);

        ///https://galaxiesbook.org/chapters/II-01.-Flattened-Mass-Distributions.html 8.16
        //return std::sqrt(local_G * cdf(r) * r * r * pow(r * r + 1 * 1, -3.f/2.f));

        //return distribution.get_mond_velocity_at(mass, local_G, r, a0);
        return distribution.get_velocity_at(mass, local_G, r);
    }

    galaxy_distribution(const galaxy_params& params)
    {
        mass = params.mass_kg / get_solar_mass_kg();
        max_radius = 100.4;

        double to_local_distance = max_radius / params.radius_m;
        double to_local_mass = mass / params.mass_kg;

        local_G = get_G() * pow(to_local_distance, 3) / to_local_mass;
        meters_to_local = to_local_distance;
        kg_to_local = to_local_mass;
    }

    double select_radius(xoshiro256ss_state& rng)
    {
        auto lambda_cdf = [&](double r)
        {
            return cdf(r);
        };

        double random = uint64_to_double(xoshiro256ss(rng));

        double random_mass = random * mass;

        return select_from_cdf(random_mass, max_radius, lambda_cdf);
    }
};

struct numerical_params
{
    double mass = 0;
    double radius = 0;

    double mass_to_m = 0;
    double m_to_scale = 0;

    numerical_params(const galaxy_params& params, float simulation_width)
    {
        double C = 299792458.;
        double G = 6.67430 * pow(10., -11.);

        double mass_in_m = params.mass_kg * G / (C*C);
        double radius_in_m = params.radius_m;

        double max_scale_radius = simulation_width * 0.5f;
        double meters_to_scale = max_scale_radius / radius_in_m;

        mass = mass_in_m * meters_to_scale;
        radius = radius_in_m * meters_to_scale;

        printf("My mass %f\n", mass);

        mass_to_m = G / (C*C);
        m_to_scale = meters_to_scale;
    }

    double convert_mass_to_scale(double mass_kg)
    {
        return mass_kg * mass_to_m * m_to_scale;
    }

    double convert_distance_to_scale(double dist_m)
    {
        return dist_m * m_to_scale;
    }
};


galaxy_data build_galaxy(float simulation_width)
{
    ///https://arxiv.org/abs/1607.08364
    //double milky_way_mass_kg = 6.43 * pow(10., 10.) * 1.16 * get_solar_mass_kg();
    double milky_way_mass_kg = 2 * pow(10, 12) * get_solar_mass_kg();
    //double milky_way_mass_kg = 4 * pow(10, 11) * get_solar_mass_kg();
    //double milky_way_radius_m = 0.5f * pow(10., 21.);
    double milky_way_radius_m = 0.5 * 8 * pow(10., 20.);

    galaxy_params params;
    params.mass_kg = milky_way_mass_kg;
    params.radius_m = milky_way_radius_m;

    galaxy_distribution<disk_distribution> dist(params);

    numerical_params num_params(params, simulation_width);

    std::vector<t3f> positions;
    std::vector<t3f> directions;
    std::vector<float> masses;
    std::vector<float> analytic_cumulative_mass;

    xoshiro256ss_state rng = xoshiro256ss_init(2345);

    int test_particle_count = 1000 * 500;

    //check the actual cumulative mass that we generate
    for(int i=0; i < test_particle_count; i++)
    {
        double radius = dist.select_radius(rng);

        if(radius >= dist.max_radius)
            continue;

        double velocity = dist.get_velocity_at(radius);

        //printf("Local Velocity %f\n", velocity);
        //printf("Local To Meters", 1/dist.meters_to_local);

        double angle = uint64_to_double(xoshiro256ss(rng)) * 2 * std::numbers::pi_v<float>;

        double z = 0;

        double radius_m = dist.local_distance_to_meters(radius);
        double radius_scale = num_params.convert_distance_to_scale(radius_m);

        //double scale_radius = num_params.convert_distance_to_scale(radius);

        t3f pos = {cos(angle) * radius_scale, sin(angle) * radius_scale, z};

        positions.push_back(pos);

        ///velocity is distance/s so should be fine
        double speed_in_ms = dist.local_distance_to_meters(velocity);
        double speed_in_c = speed_in_ms / get_c();

        auto rot = [](t2f in, float rot_angle)
        {
            float len = in.length();

            if(len < 0.00001f)
                return in;

            float cur_angle = atan2(in.y(), in.x());

            float new_angle = cur_angle + rot_angle;

            float nx = len * cos(new_angle);
            float ny = len * sin(new_angle);

            return (t2f){nx, ny};
        };

        t2f velocity_direction = rot({1, 0}, angle + std::numbers::pi_v<float>/2);

        t2f velocity_2d = speed_in_c * velocity_direction;

        t3f velocity_fin = {velocity_2d.x(), velocity_2d.y(), 0.f};

        directions.push_back(velocity_fin);

        double local_analytic_mass = dist.cdf(radius);
        double analytic_mass_kg = dist.local_mass_to_kg(local_analytic_mass);
        double analytic_mass_scale = num_params.convert_mass_to_scale(analytic_mass_kg);

        analytic_cumulative_mass.push_back(analytic_mass_scale);

        assert(speed_in_c < 1);

        //printf("Velocity %f\n", speed_in_c);
        //printf("Position %f %f %f\n", pos.x(), pos.y(), pos.z());
    }

    int real_count = positions.size();

    float max_mass = dist.cdf(dist.max_radius);

    float init_mass = num_params.mass / real_count;

    printf("Max %f %f %f dist mass %f\n", max_mass, init_mass, dist.max_radius, dist.mass);

    ///https://www.mdpi.com/2075-4434/6/3/70/htm mond galaxy info

    for(uint64_t i=0; i < (uint64_t)real_count; i++)
    {
        masses.push_back(init_mass);
    }

    std::vector<float> debug_velocities;
    std::vector<float> debug_analytic_mass;
    std::vector<float> debug_real_mass;

    {
        std::vector<std::tuple<t3f, t3f, float>> pos_vel;
        pos_vel.reserve(real_count);

        for(int i=0; i < real_count; i++)
        {
            pos_vel.push_back({positions[i], directions[i], analytic_cumulative_mass[i]});
        }

        std::sort(pos_vel.begin(), pos_vel.end(), [](auto& i1, auto& i2)
        {
            return std::get<0>(i1).squared_length() < std::get<0>(i2).squared_length();
        });

        float selection_radius = 0;

        float real_mass = 0;

        for(auto& [p, v, m] : pos_vel)
        {
            float p_len = p.length();
            //float v_len = v.length();

            if(p_len >= selection_radius)
            {
                printf("Velocity %f real mass %f radius %f amass %f\n", v.length(), real_mass, p_len, m);

                selection_radius += 0.25f;
                debug_velocities.push_back(v.length());

                debug_real_mass.push_back(real_mass);
                debug_analytic_mass.push_back(m);
            }

            real_mass += init_mass;
        }
    }

    galaxy_data ret;
    ret.positions = std::move(positions);
    ret.velocities = std::move(directions);
    ret.masses = std::move(masses);
    ret.debug_velocities = std::move(debug_velocities);
    ret.debug_analytic_mass = std::move(debug_analytic_mass);
    ret.debug_real_mass = std::move(debug_real_mass);
    //ret.particle_brightness = 0.01;

    return ret;
}
#endif

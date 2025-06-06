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

double density_impl(double M, double R, double z, double a, double b)
{
    double pi = std::numbers::pi_v<double>;

    double left = (b*b * M) / (4 * pi);

    double top = a * R * R + (3 * sqrt(z*z + b*b) + a) * pow(sqrt(z*z + b*b) + a, 2.);
    double bot = pow(R*R + pow(sqrt(z*z + b*b) + a, 2.), 5./2.) * pow(z*z + b*b, 3./2.);

    return left * (top / bot);
}

double poisson_impl(double M, double R, double z, double a, double b)
{
    return -get_G() * M / sqrt(R*R + pow(sqrt(z*z + b*b) + a, 2.));
}

double get_solar_mass_kg()
{
    return 1.988416 * std::pow(10., 30.);
}

double parsec_to_m(double pc)
{
    return pc * 3.086 * pow(10., 16.);
}

///todo: do my shitty density idea
///ie place particles in a regular grid pattern, assign them mass based on density, then scale the overall mass
struct disk_distribution
{
    //double M = 0;
    //double a_frac = 0.05;
    //double b_frac = 0.01;
    //double max_R = 0;
    //double a = 0;
    //double b = 0;

    std::vector<double> masses = {2.05 * pow(10., 10.) * get_solar_mass_kg(), 2.547 * pow(10, 11.) * get_solar_mass_kg()};
    std::vector<double> as = {0, parsec_to_m(7.258)};
    std::vector<double> bs = {parsec_to_m(0.495), parsec_to_m(0.520)};

    double get_mass()
    {
        double d = 0;

        for(auto& i : masses)
            d += i;

        return d;
    }

    double get_max_radius()
    {
        return as[1] * 10;
    }

    double get_density(double x_phys, double y_phys, double z_phys)
    {
        double d = 0;

        double R = (tensor<double, 2>){x_phys, y_phys}.length();

        for(int i=0; i < (int)masses.size(); i++)
            d += density_impl(masses[i], R, z_phys, as[i], bs[i]);

        return d;
    }

    double get_potential(double x_phys, double y_phys, double z_phys)
    {
        double d = 0;

        double R = (tensor<double, 2>){x_phys, y_phys}.length();

        for(int i=0; i < (int)masses.size(); i++)
            d += poisson_impl(masses[i], R, z_phys, as[i], bs[i]);

        return d;
    }

    double get_velocity(double x_phys, double y_phys, double z_phys)
    {
        double max_radius = get_max_radius();

        tensor<double, 3> pos = {x_phys, y_phys, z_phys};

        double R = sqrt(x_phys * x_phys + y_phys * y_phys + z_phys*z_phys);

        double h = max_radius * 0.0001;

        double a_x = (get_potential(x_phys + h, y_phys, z_phys) - get_potential(x_phys - h, y_phys, z_phys)) / (2 * h);
        double a_y = (get_potential(x_phys, y_phys + h, z_phys) - get_potential(x_phys, y_phys - h, z_phys)) / (2 * h);
        double a_z = (get_potential(x_phys, y_phys, z_phys + h) - get_potential(x_phys, y_phys, z_phys - h)) / (2 * h);

        tensor<double, 3> acc = {a_x, a_y, a_z};

        double div = acc.length();

        ///centripetal force = m v^2/r
        //=force due to gravity = F(x) = -m div X
        //set to opposite. m v^2/r = m div X
        //v^2/r = div X

        //double div = acc.length();

        ///so, we have force due to gravity -m Di X
        //centripetal force = m r_i W^2
        ///r_i W^2 = Di X

        //double div = acc.x() + acc.y() + acc.z();

        return sqrt(R * div);
        ///centripetal acceleration = v^2 / R
        ///centripetal acceleration = -F =

        ///F = -m delta phi(x)?
        ///a = diff phi

        //double R = (tensor<double, 2>){x_phys, y_phys}.length();
    }

    /*double normalised_cdf(double fraction)
    {
        double a = a_frac * max_R;

        double r = fraction * max_R;

        return 1 - a / sqrt(r*r + a*a);
    }*/

    //radius -> velocity
    /*double fraction_to_velocity(double fraction, float z_frac)
    {
        double R = fraction * max_R;

        double h = 0.01 * max_R;
        //i don't think this is correct
        return sqrt(R * (potential((R + h) / max_R, z_frac) - potential((R - h) / max_R, z_frac)) / (2 * h));
    }*/

    /*std::optional<double> select_radial_frac(xoshiro256ss_state& rng)
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
    }*/

    ///https://ned.ipac.caltech.edu/level5/Sept16/Sofue/Sofue4.html some explicit values here

    #if 0
    double density(double x_frac, double y_frac, double z_frac)
    {
        double pi = std::numbers::pi_v<double>;

        tensor<float, 3> pos = {x_frac, y_frac, z_frac};

        double R = pos.xy().length() * max_R;
        double z = pos.z() * max_R;

        double a = a_frac * max_R;
        double b = b_frac * max_R;

        double left = (b*b * M0) / (4 * pi);

        double top = a * R * R + (3 * sqrt(z*z + b*b) + a) * pow(sqrt(z*z + b*b) + a, 2.);
        double bot = pow(R*R + pow(sqrt(z*z + b*b) + a, 2.), 5./2.) * pow(z*z + b*b, 3./2.);

        return left * (top / bot);
    }

    double potential(double radius_frac, double z_frac)
    {
        double R = radius_frac * max_R;
        double z = z_frac * max_R;
        double a = a_frac * max_R;
        double b = b_frac * max_R;

        return -get_G() * M0 / sqrt(R*R + pow(sqrt(z*z + b*b) + a, 2.));
    }
    #endif

    #if 0
    double density(double x_frac, double y_frac, double z_frac)
    {
        if(z_frac != 0)
            return 0.f;

        double a = a_frac * max_R;

        tensor<float, 3> pos = {x_frac, y_frac, z_frac};

        double R = pos.length() * max_R;

        double pi = std::numbers::pi_v<double>;

        return (M0 * a / (2 * pi)) * pow(R*R + a*a, -3.f/2.f);
    }

    double potential(double radius_frac, double z_frac)
    {
        double R = radius_frac * max_R;
        double z = z_frac * max_R;
        double a = a_frac * max_R;

        return -get_G() * M0 / sqrt(R*R + pow(fabs(z) + a, 2.));
    }
    #endif
};

galaxy_data build_galaxy(float fill_width)
{
    double fill_radius = fill_width/2;

    galaxy_data dat;
    xoshiro256ss_state rng = xoshiro256ss_init(2345);

    int try_num = 1000 * 1000;
    int real_num = 0;

    //double milky_way_mass_kg = 2 * pow(10, 12) * get_solar_mass_kg();
    //double milky_way_mass_kg = 4 * pow(10, 11) * get_solar_mass_kg();
    //double milky_way_radius_m = 0.5 * 8 * pow(10., 20.);

    //double milky_way_mass_kg = pow(10., 10.) * get_solar_mass_kg();
    //double milky_way_radius_m = 3 * pow(10., 19.);

    disk_distribution disk;
    //disk.M0 = milky_way_mass_kg;
    //disk.max_R = milky_way_radius_m;

    //double encapsulated_mass_kg = milky_way_mass_kg * disk.normalised_cdf(1.f);

    double total_mass = disk.get_mass();
    double max_world_radius = disk.get_max_radius();

    double meters_to_real = fill_radius / max_world_radius;

    double encapsulated_mass_kg = total_mass;
    double mass_m = si_to_geometric(encapsulated_mass_kg, 1, 0);
    double mass_real = mass_m * meters_to_real;

    std::vector<double> analytic_cumulative_mass;

    /*for(int i=0; i < try_num; i++)
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

        double pi = std::numbers::pi_v<double>;

        double angle = uint64_to_double(xoshiro256ss(rng)) * 2 * pi;

        double radius_real = (radius / disk.max_R) * fill_radius;

        t3f vel = {velocity_geom * cos(angle + pi/2), velocity_geom * sin(angle + pi/2), 0.f};
        t3f pos = {radius_real * cos(angle), radius_real * sin(angle), 0.f};

        dat.positions.push_back(pos);
        dat.velocities.push_back(vel);

        double local_analytic_mass = milky_way_mass_kg * disk.normalised_cdf(radius / disk.max_R);
        double analytic_mass_scale = si_to_geometric(local_analytic_mass, 1, 0) * (fill_radius / disk.max_R);
        analytic_cumulative_mass.push_back(analytic_mass_scale);
    }*/

    /*for(int x=0; x < 100; x++)
    {
        double den = disk.density(x / 100., 0, 0);

        printf("Den %.32f\n", den);
    }*/

    //assert(false);

    double total_den = 0;
    std::vector<double> n_densities;

    int max_dim = 70;

    for(int z=-max_dim; z <= max_dim; z++)
    {
        for(int y=-max_dim; y <= max_dim; y++)
        {
            for(int x=-max_dim; x <= max_dim; x++)
            {
                float fx = (float)x / max_dim;
                float fy = (float)y / max_dim;
                float fz = (float)z / max_dim;

                double frac_radius = sqrt(fx * fx + fy * fy + fz * fz);

                if(frac_radius > 1)
                    continue;

                /*double R_frac = sqrt(fx*fx + fy*fy);

                double den = disk.density(fx, fy, fz);

                if((float)den == 0)
                    continue;

                double velocity = disk.fraction_to_velocity(R_frac, fz);*/

                double world_x = fx * disk.get_max_radius();
                double world_y = fy * disk.get_max_radius();
                double world_z = fz * disk.get_max_radius();

                double density = disk.get_density(world_x, world_y, world_z);

                double velocity = disk.get_velocity(world_x, world_y, world_z);

                //velocity = 0;

                double velocity_geom = velocity / get_c();

                //double angle = atan2(fy, fx);

                double pi = std::numbers::pi_v<double>;

                t3f up_axis = {0, 0, 1};

                if(fx == 0 && fy == 0 && fz == 0)
                    continue;

                if(fx == 0 && fy == 0 && fz > 0)
                    up_axis = {1, 0, 0};
                if(fx == 0 && fy == 0 && fz < 0)
                    up_axis = {-1, 0, 0};

                t3f rotation_axis = -cross((t3f){fx, fy, fz}.norm(), up_axis.norm()).norm();

                t3f vel = rotation_axis * velocity_geom;

                //t3f vel = {(float)velocity_geom.x(), (float)velocity_geom.y(), (float)velocity_geom.z()};

                //vel.x() += vel.x() * (uint64_to_double(xoshiro256ss(rng)) - 0.5) * 0.2;
                //vel.y() += vel.y() * (uint64_to_double(xoshiro256ss(rng)) - 0.5) * 0.2;
                //vel.z() += vel.z() * (uint64_to_double(xoshiro256ss(rng)) - 0.5) * 0.2;

                //t3f vel = {velocity_geom * cos(angle + pi/2), velocity_geom * sin(angle + pi/2), 0.f};
                t3f pos = {fx * fill_radius, fy * fill_radius, fz * fill_radius};

                dat.positions.push_back(pos);
                dat.velocities.push_back(vel);

                n_densities.push_back(density);

                total_den += density;

                /*double local_analytic_mass = milky_way_mass_kg * disk.normalised_cdf(frac_radius);
                double analytic_mass_scale = si_to_geometric(local_analytic_mass, 1, 0) * (fill_radius / disk.max_R);
                analytic_cumulative_mass.push_back(analytic_mass_scale);*/
            }
        }
    }

    real_num = dat.positions.size();

    /*for(auto& pos : dat.positions)
    {
        //double den = disk.density(pos.x() / fill_radius, pos.y() / fill_radius, pos.z() / fill_radius);

        double mass = den * (mass_real / total_den);

        dat.masses.push_back(mass);
    }*/

    for(auto den : n_densities)
    {
        double mass = den * (mass_real / total_den);

        dat.masses.push_back(mass);
    }

    /*std::vector<float> debug_velocities;
    std::vector<float> debug_analytic_mass;
    std::vector<float> debug_real_mass;

    {
        std::vector<std::tuple<t3f, t3f, float>> pos_vel;

        for(int i=0; i < real_num; i++)
            pos_vel.push_back({dat.positions[i], dat.velocities[i], dat.masses[i]});

        std::sort(pos_vel.begin(), pos_vel.end(), [](auto& i1, auto& i2)
        {
            return std::get<0>(i1).squared_length() < std::get<0>(i2).squared_length();
        });

        float selection_radius = 0;

        double real_mass = 0;
        int particles = 0;

        for(auto& [p, v, m] : pos_vel)
        {
            float p_len = p.length();
            //float v_len = v.length();

            if(p_len >= selection_radius)
            {
                printf("Velocity %f real mass %f radius %f rmass %.23f particles %i density %.23f\n", v.length(), real_mass, p_len, m, particles, disk.density(p.x() / fill_radius, p.y() / fill_radius, 0));

                selection_radius += 0.25f;
                debug_velocities.push_back(v.length());

                debug_real_mass.push_back(real_mass);
                //debug_analytic_mass.push_back(am);
            }

            real_mass += m;
            particles++;
        }
    }*/

    //assert(false);

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

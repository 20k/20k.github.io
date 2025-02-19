#include "init_neutron_star.hpp"
#include "tov.hpp"
#include "../common/vec/tensor.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include <toolkit/opencl.hpp>
#include "bssn.hpp"
#include "tensor_algebra.hpp"
#include "init_general.hpp"

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

void neutron_star::all_numerical_eos_gpu::init(cl::command_queue cqueue, const std::vector<neutron_star::numerical_eos>& eos)
{
    if(eos.size() == 0)
        return;

    int root_size = eos[0].pressure.size();

    for(auto& i : eos)
        assert(i.pressure.size() == root_size);

    stride = root_size;
    count = eos.size();

    pressures.alloc(sizeof(cl_float) * stride * count);
    max_densities.alloc(sizeof(cl_float) * count);

    std::vector<float> all_pressures;
    std::vector<float> all_densities;

    for(int i=0; i < (int)eos.size(); i++)
    {
        for(auto& j : eos[i].pressure)
            all_pressures.push_back(j);

        all_densities.push_back(eos[i].max_density);
    }

    pressures.write(cqueue, all_pressures);
    max_densities.write(cqueue, all_densities);
}

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

void matter_accum(execution_context& ctx, buffer<valuef> Q_b, buffer<valuef> C_b, buffer<valuef> uN_b,
                buffer<valuef> sigma_b, buffer<valuef> kappa_b,
                buffer<valuef> mu_cfl_b, buffer<valuef> pressure_cfl_b, buffer<valuef> radius_b,
                literal<valuei> lsamples, literal<valuef> lM, literal<valuef> l_sN,
                literal<v3i> ldim, literal<valuef> lscale,
                literal<v3f> lbody_pos, literal<v3f> linear_momentum, literal<v3f> angular_momentum,
                std::array<buffer_mut<valuef>, 6> AIJ_out, buffer_mut<valuef> mu_h_cfl_out,
                std::array<buffer_mut<valuef>, 3> Si_out, buffer_mut<valuei> star_indices, literal<valuei> index)
{
    using namespace single_source;

    valuei x = get_global_id(0);
    valuei y = get_global_id(1);
    valuei z = get_global_id(2);

    pin(x);
    pin(y);
    pin(z);

    v3i dim = ldim.get();
    valuef scale = lscale.get();
    valuei samples = lsamples.get();

    if_e(x >= dim.x() || y >= dim.y() || z >= dim.z(), [&]{
        return_e();
    });

    v3i pos = {x, y, z};

    v3f fpos = (v3f)pos;

    v3f body_pos = lbody_pos.get();
    v3f world_pos = grid_to_world(fpos, dim, scale);

    v3f from_body = world_pos - body_pos;

    valuef r = from_body.length();
    pin(r);

    v3f li = from_body / max(r, valuef(0.001f));

    auto get = [&](single_source::buffer<valuef> quantity, valuef upper_boundary)
    {
        mut<valuef> out = declare_mut_e(valuef(0.f));

        if_e(r <= radius_b[0], [&]{
            as_ref(out) = quantity[0];
        });

        if_e(r > radius_b[samples - 1], [&]{
            as_ref(out) = upper_boundary;
        });

        if_e(r > radius_b[0] && r <= radius_b[samples - 1], [&]{
            mut<valuei> index = declare_mut_e(valuei(0));
            mut<valuef> last_r = declare_mut_e(valuef(0.f));

            for_e(index < samples - 1, assign_b(index, index+1), [&]{
                valuef r1 = radius_b[index];
                valuef r2 = radius_b[index + 1];

                if_e(r > r1 && r <= r2, [&]{
                     valuef frac = (r - r1) / (r2 - r1);

                     as_ref(out) = mix(quantity[index], quantity[index + 1], frac);
                     break_e();
                });
            });
        });

        return declare_e(out);
    };

    ///todo: need to define upper boundaries
    valuef Q = get(Q_b, 1.f);
    valuef C = get(C_b, C_b[samples-1]);
    valuef N = get(uN_b, uN_b[samples - 1]);
    valuef sigma = get(sigma_b, 0.f);
    valuef kappa = get(kappa_b, 0.f);

    valuef mu_cfl = get(mu_cfl_b, 0.f);
    valuef pressure_cfl = get(pressure_cfl_b, 0.f);

    unit_metric<valuef, 3, 3> flat;

    for(int i=0; i < 3; i++)
        flat[i, i] = valuef(1);

    ///super duper unnecessary here but i'm being 110% pedantic
    auto iflat = flat.invert();
    pin(iflat);

    v3f li_lower = flat.lower(li);
    v3f P = linear_momentum.get();
    v3f J = angular_momentum.get();

    valuef pk_lk = dot(P, li_lower);

    tensor<valuef, 3, 3> AIJ_p;

    valuef cr = max(r, valuef(0.01f));

    ///hmm. I think the extrinsic curvature may be wrong
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            AIJ_p[i, j] = (3 * Q / (2 * cr*cr)) *   (P[i] * li[j] + P[j] * li[i] - (iflat[i, j] - li[i] * li[j]) * pk_lk)
                         +(3 * C / (cr*cr*cr*cr)) * (P[i] * li[j] + P[j] * li[i] + (iflat[i, j] - 5 * li[i] * li[j]) * pk_lk);
        }
    }

    auto eijk = get_eijk();

    //super unnecessary, being pedantic
    auto eIJK = iflat.raise(iflat.raise(iflat.raise(eijk, 0), 1), 2);

    tensor<valuef, 3, 3> AIJ_j;

    v3f J_lower = flat.lower(J);

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            valuef sum = 0;

            for(int k=0; k < 3; k++)
            {
                for(int l=0; l < 3; l++)
                {
                    sum += (3 / (cr*cr*cr)) * (li[i] * eIJK[j, k, l] + li[j] * eIJK[i, k, l]) * J_lower[k] * li_lower[l] * N;
                }
            }

            AIJ_j[i, j] += sum;
        }
    }


    tensor<valuef, 3, 3> AIJ = AIJ_p + AIJ_j;

    v3f P_lower = flat.lower(P);

    valuef M = lM.get();
    valuef squiggly_N = l_sN.get();

    valuef W2_P = 0.5f * (1 + sqrt(1 + (4 * dot(P_lower, P)) / (M*M)));

    v3f J_norm = J / max(J.length(), valuef(0.0001f));
    v3f li_lower_norm = li_lower / max(li_lower.length(), valuef(0.0001f));

    value sin2 = 1 - pow(dot(J_norm, li_lower_norm), valuef(2.f));

    valuef W2_J = 0.5f * (1 + sqrt(1 + (4 * dot(J_lower, J) * r * r * sin2) / (squiggly_N*squiggly_N)));

    valuef W = cosh(acosh(sqrt(W2_P)) + acosh(sqrt(W2_J)));

    tensor<int, 2> index_table[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    for(int i=0; i < 6; i++)
    {
        tensor<int, 2> idx = index_table[i];
        as_ref(AIJ_out[i][pos, dim]) += AIJ[idx.x(), idx.y()];
    }

    valuef mu_h = (mu_cfl + pressure_cfl) * W*W - pressure_cfl;

    as_ref(mu_h_cfl_out[pos, dim]) += mu_h;

    v3f Si_P = P * sigma;
    v3f S_iJ;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                S_iJ[i] += eijk[i, j, k] * J[j] * from_body[k] * kappa;
            }
        }
    }

    v3f Si = Si_P + iflat.raise(S_iJ);

    for(int i=0; i < 3; i++)
    {
        as_ref(Si_out[i][pos, dim]) += Si[i];
    }

    if_e(r <= radius_b[samples-1], [&]{
        as_ref(star_indices[pos, dim]) = index.get();
    });

    /*if_e(pos.x() == dim.x()/2 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        value_base se;
        se.type = value_impl::op::SIDE_EFFECT;
        se.abstract_value = "printf(\"hello %f\\n\"," + value_to_string(M) + ")";

        value_impl::get_context().add(se);
    });*/
}

void neutron_star::boot_solver(cl::context ctx)
{
    cl::async_build_and_cache(ctx, [=] {
        return value_impl::make_function(matter_accum, "matter_accum");
    }, {"matter_accum"});
}

void neutron_star::data::add_to_solution(cl::context& ctx, cl::command_queue& cqueue,
                                         discretised_initial_data& dsol,
                                         tensor<int, 3> dim, float scale, int index)
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
        assert(index >= 0 && index <= samples);

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

    /*void matter_accum(execution_context& ctx, buffer<valuef> Q_b, buffer<valuef> C_b, buffer<valuef> uN_b,
                buffer<valuef> sigma_b, buffer<valuef> kappa_b,
                buffer<valuef> mu_cfl_b, buffer<valuef> pressure_cfl_b, buffer<valuef> radius_b,
                literal<valuei> lsamples, literal<valuef> lM, literal<valuef> l_sN,
                literal<v3i> ldim, literal<valuef> lscale,
                literal<v3f> lbody_pos, literal<v3f> linear_momentum, literal<v3f> angular_momentum,
                std::array<buffer_mut<valuef>, 6> AIJ_out, buffer_mut<valuef> mu_cfl_out, buffer_mut<valuef> mu_h_cfl_out, buffer_mut<valuef> pressure_cfl_out,
                std::array<buffer_mut<valuef>, 3> Si_out)*/

    t3f angular_momentum;

    if(params.angular_momentum.momentum)
        angular_momentum = params.angular_momentum.momentum.value();

    if(params.angular_momentum.dimensionless)
    {
        dimensionless_angular_momentum dam = params.angular_momentum.dimensionless.value();

        ///x = J/M^2
        ///J = x M^2
        angular_momentum = dam.axis * dam.x * total_mass * total_mass;
    }

    {
        cl_float clM = M;
        cl_float clsN = squiggly_N;

        cl::args args;
        args.push_back(cl_Q);
        args.push_back(cl_C);
        args.push_back(cl_uN);
        args.push_back(cl_sigma);
        args.push_back(cl_kappa);
        args.push_back(cl_mu_cfl);
        args.push_back(cl_pressure_cfl);
        args.push_back(cl_radius);

        args.push_back(samples);
        args.push_back(clM);
        args.push_back(clsN);
        args.push_back(dim);
        args.push_back(scale);
        args.push_back((t3f)params.position);
        args.push_back((t3f)params.linear_momentum);
        args.push_back((t3f)angular_momentum);
        args.push_back(dsol.AIJ_cfl[0], dsol.AIJ_cfl[1], dsol.AIJ_cfl[2], dsol.AIJ_cfl[3], dsol.AIJ_cfl[4], dsol.AIJ_cfl[5]);
        args.push_back(dsol.mu_h_cfl);
        args.push_back(dsol.Si_cfl[0], dsol.Si_cfl[1], dsol.Si_cfl[2]);
        args.push_back(dsol.star_indices);
        args.push_back(index);

        cqueue.exec("matter_accum", args, {dim.x(), dim.y(), dim.z()}, {8,8,1});
    }
}

neutron_star::data::data(const parameters& p) : params(p)
{
    tov::parameters tov_params;
    tov_params.K = params.K.msols.value();
    tov_params.Gamma = params.Gamma;

    /*//kg/m^3 -> m/m^3 -> 1/m^2
    double p0_geom = si_to_geometric(p0, 1, 0);
    //m^-2 -> msol^-2
    double p0_msol = geometric_to_msol(p0_geom, -2);*/

    tov::integration_state start;

    if(params.mass.p0_kg_m3)
    {
        //kg/m^3 -> m/m^3 -> 1/m^2
        double p0_geom = si_to_geometric(params.mass.p0_kg_m3.value(), 1, 0);
        //m^-2 -> msol^-2
        p0_msols = geometric_to_msol(p0_geom, -2);
    }
    else if(params.mass.p0_geometric)
    {
        p0_msols = geometric_to_msol(params.mass.p0_geometric.value(), -2);
    }
    else if(params.mass.p0_msols)
    {
        p0_msols = params.mass.p0_msols.value();
    }
    else if(params.mass.rest_mass)
    {
        param_rest_mass mass = params.mass.rest_mass.value();

        std::vector<double> masses = tov::search_for_adm_mass(mass.mass, tov_params);

        assert(masses.size() > 0 && mass.result_index >= 0 && mass.result_index < (int)masses.size());

        p0_msols = masses.at(mass.result_index);
    }
    else
        assert(false);

    start = tov::make_integration_state(p0_msols, 1e-6, tov_params);
    sol = tov::solve_tov(start, tov_params, 1e-6, 0);

    total_mass = sol.M_msol;
}

neutron_star::numerical_eos neutron_star::data::get_eos()
{
    numerical_eos ret;

    float max_density = p0_msols * 4;

    ret.max_density = max_density;

    int max_samples = 400;

    for(int i=0; i < max_samples; i++)
    {
        double density = ((double)i / max_samples) * max_density;

        double pressure = params.K.msols.value() * pow(density, params.Gamma);

        ret.pressure.push_back(pressure);
    }

    return ret;
}

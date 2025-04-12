#include "init_neutron_star.hpp"
#include "../common/vec/tensor.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include <toolkit/opencl.hpp>
#include "bssn.hpp"
#include "tensor_algebra.hpp"
#include "init_general.hpp"
#include "value_alias.hpp"

void neutron_star::all_numerical_eos_gpu::init(cl::command_queue cqueue, const std::vector<neutron_star::numerical_eos>& eos)
{
    if(eos.size() == 0)
        return;

    int root_size = eos[0].pressure.size();

    for(auto& i : eos)
        assert(i.pressure.size() == root_size);

    for(auto& i : eos)
        assert(i.mu_to_p0.size() == root_size);

    stride = root_size;
    count = eos.size();

    pressures.alloc(sizeof(cl_float) * stride * count);
    mu_to_p0.alloc(sizeof(cl_float) * stride * count);

    max_densities.alloc(sizeof(cl_float) * count);
    max_mus.alloc(sizeof(cl_float) * count);

    {
        std::vector<float> all_pressures;
        std::vector<float> max_densities_vec;

        for(int i=0; i < (int)eos.size(); i++)
        {
            for(auto& j : eos[i].pressure)
                all_pressures.push_back(j);

            max_densities_vec.push_back(eos[i].max_density);
        }

        pressures.write(cqueue, all_pressures);
        max_densities.write(cqueue, max_densities_vec);
    }

    {
        std::vector<float> all_p0s;
        std::vector<float> max_mus_vec;

        for(int i=0; i < (int)eos.size(); i++)
        {
            for(auto& j : eos[i].mu_to_p0)
                all_p0s.push_back(j);

            max_mus_vec.push_back(eos[i].max_mu);
        }

        mu_to_p0.write(cqueue, all_p0s);
        max_mus.write(cqueue, max_mus_vec);
    }
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
                std::array<buffer_mut<valuef>, 6> AIJ_accumulate, std::array<buffer_mut<valuef>, 6> AIJ_this_star, buffer_mut<valuef> mu_h_cfl_out,
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

    auto get = [&](single_source::buffer<valuef> quantity, valuef upper_boundary, valuef r)
    {
        mut<valuef> out = declare_mut_e(valuef(-1.f));
        mut<valuei> found = declare_mut_e(valuei(0));

        if_e(r <= radius_b[0], [&]{
            as_ref(out) = quantity[0];
            as_ref(found) = valuei(1);
        });

        if_e(r > radius_b[samples - 1], [&]{
            as_ref(out) = upper_boundary;
            as_ref(found) = valuei(1);
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
                     as_ref(found) = valuei(1);
                     break_e();
                });
            });
        });

        if_e(found == 0, [&]{
            print("Borked get\n");
        });

        return declare_e(out);
    };

    auto get_ext = [&](v3f fpos)
    {
        v3f body_pos = lbody_pos.get();
        v3f world_pos = grid_to_world(fpos, dim, scale);

        v3f from_body = world_pos - body_pos;

        valuef r = from_body.length();
        pin(r);

        v3f li = from_body / max(r, valuef(0.0001f));

        ///todo: need to define upper boundaries
        valuef Q = get(Q_b, 1.f, r);
        valuef C = get(C_b, C_b[samples-1], r);
        valuef N = get(uN_b, uN_b[samples - 1], r);
        valuef sigma = get(sigma_b, 0.f, r);
        valuef kappa = get(kappa_b, 0.f, r);

        printf("Upper N %.32f\n", uN_b[samples - 1]);

        valuef mu_cfl = get(mu_cfl_b, 0.f, r);
        valuef pressure_cfl = get(pressure_cfl_b, 0.f, r);

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
        return AIJ;
    };

    tensor<valuef, 3> Si;

    for(int i=0; i < 3; i++)
    {
        valuef sum = 0;

        for(int j=0; j < 3; j++)
        {
            v3f offset;
            offset[j] = 0.1f;

            v3f fpos = (v3f)pos;

            tensor<valuef, 3, 3> right = get_ext(fpos + offset);
            tensor<valuef, 3, 3> left = get_ext(fpos - offset);

            auto diff = (right - left) / (2 * scale);

            sum += diff[i, j] / (8 * M_PI);
        }

        Si[i] = sum;
    }

    v3f cSi = declare_e(Si);

    /*
    valuef mu_h = (mu_cfl + pressure_cfl) * W2 - pressure_cfl;

    as_ref(mu_h_cfl_out[pos, dim]) += mu_h;

    for(int i=0; i < 3; i++)
    {
        as_ref(Si_out[i][pos, dim]) += cSi[i];
    }*/


    v3f fpos = (v3f)pos;

    v3f body_pos = lbody_pos.get();
    v3f world_pos = grid_to_world(fpos, dim, scale);

    v3f from_body = world_pos - body_pos;

    valuef r = from_body.length();
    pin(r);

    valuef mu_cfl = get(mu_cfl_b, 0.f, r);
    valuef pressure_cfl = get(pressure_cfl_b, 0.f, r);

    unit_metric<valuef, 3, 3> flat;

    for(int i=0; i < 3; i++)
        flat[i, i] = valuef(1);

    valuef W2 = 0.5f * (1 + sqrt(1 + (4 * flat.dot(cSi, cSi)) / max(pow(mu_cfl + pressure_cfl, 2.f), valuef(1e-12f))));

    valuef mu_h = (mu_cfl + pressure_cfl) * W2 - pressure_cfl;

    as_ref(mu_h_cfl_out[pos, dim]) += mu_h;

    for(int i=0; i < 3; i++)
    {
        as_ref(Si_out[i][pos, dim]) += cSi[i];
    }

    tensor<valuef, 3, 3> AIJ = get_ext(fpos);

    tensor<int, 2> index_table[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    for(int i=0; i < 6; i++)
    {
        tensor<int, 2> idx = index_table[i];
        as_ref(AIJ_accumulate[i][pos, dim]) += AIJ[idx.x(), idx.y()];
        as_ref(AIJ_this_star[i][pos, dim]) = AIJ[idx.x(), idx.y()];
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

void matter_p2(execution_context& ctx, buffer<valuef> Q_b, buffer<valuef> C_b, buffer<valuef> uN_b,
                buffer<valuef> sigma_b, buffer<valuef> kappa_b,
                buffer<valuef> mu_cfl_b, buffer<valuef> pressure_cfl_b, buffer<valuef> radius_b,
                literal<valuei> lsamples, literal<valuef> lM, literal<valuef> l_sN,
                literal<v3i> ldim, literal<valuef> lscale,
                literal<v3f> lbody_pos, literal<v3f> linear_momentum, literal<v3f> angular_momentum,
                std::array<buffer_mut<valuef>, 6> AIJ_accumulate, std::array<buffer_mut<valuef>, 6> AIJ_this_star, buffer_mut<valuef> mu_h_cfl_out,
                std::array<buffer_mut<valuef>, 3> Si_out, buffer_mut<valuei> star_indices, literal<valuei> index)
{
    return;

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

    valuef sigma = get(sigma_b, 0.f);
    valuef kappa = get(kappa_b, 0.f);

    tensor<valuef, 3, 3> diffable_AIJ;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            int table[3][3] = {{0, 1, 2},
                               {1, 3, 4},
                               {2, 4, 5}};

            diffable_AIJ[i, j] = AIJ_this_star[table[i][j]][pos, dim];
        }
    }

    mut_v3f Si = declare_mut_e(v3f());

    derivative_data d;
    d.pos = pos;
    d.dim = dim;
    d.scale = scale;

    if_e(distance_to_boundary(pos, dim) >= 2, [&]{
        for(int i=0; i < 3; i++)
        {
            valuef sum = 0;

            for(int j=0; j < 3; j++)
            {
                sum += diff1(diffable_AIJ[i, j], j, d);
            }

            as_ref(Si[i]) = sum / (8 * M_PI);
        }
    });

    unit_metric<valuef, 3, 3> flat;

    for(int i=0; i < 3; i++)
        flat[i, i] = valuef(1);

    valuef mu_cfl = get(mu_cfl_b, 0.f);
    valuef pressure_cfl = get(pressure_cfl_b, 0.f);

    v3f cSi = declare_e(Si);

    valuef W2 = 0.5f * (1 + sqrt(1 + (4 * flat.dot(cSi, cSi)) / max(pow(mu_cfl + pressure_cfl, 2.f), valuef(1e-12f))));

    {
        auto iflat = flat.invert();

        v3f li_lower = flat.lower(li);
        v3f P = linear_momentum.get();
        v3f J = angular_momentum.get();
        v3f J_lower = flat.lower(J);

        v3f P_lower = flat.lower(P);

        valuef M = lM.get();
        valuef squiggly_N = l_sN.get();

        valuef W2_P = 0.5f * (1 + sqrt(1 + (4 * dot(P_lower, P)) / (M*M)));

        v3f J_norm = J / max(J.length(), valuef(0.0001f));
        v3f li_lower_norm = li_lower / max(li_lower.length(), valuef(0.0001f));
        value sin2 = 1 - pow(dot(J_norm, li_lower_norm), valuef(2.f));

        valuef W2_J = 0.5f * (1 + sqrt(1 + (4 * dot(J_lower, J) * r * r * sin2) / (squiggly_N*squiggly_N)));

        auto eijk = get_eijk();

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

        v3f tSi = Si_P + iflat.raise(S_iJ);

        valuef cW = sqrt(0.5f * (1 + sqrt(1 + (4 * flat.dot(tSi, tSi)) / max(pow(mu_cfl + pressure_cfl, 2.f), valuef(1e-12f)))));

        /*cSi[0] = tSi[0];
        cSi[1] = tSi[1];
        cSi[2] = tSi[2];

        W2 = cW * cW;*/

        /*if_e(mu_cfl > 0 && W2 > 0, [&]{
            valuef m1 = (mu_cfl + pressure_cfl) * W2 - pressure_cfl;
            valuef m2 = (mu_cfl + pressure_cfl) * cW*cW - pressure_cfl;

            valuef d0 = diff1(diffable_AIJ[0, 0], 0, d);
            valuef d1 = diff1(diffable_AIJ[1, 1], 1, d);
            valuef d2 = diff1(diffable_AIJ[2, 2], 2, d);

            //print("hi1 %f %f %f hi2 %f %f %f\n", cSi[0], cSi[1], cSi[2], tSi[0], tSi[1], tSi[2]);

            print("%.24f %.24f r_err %.24f %f %f calc W2 %f paper W2 %f\n", m1, m2, (m1 - m2) / m1, W2_J, W2_P, W2, cW*cW);
        });*/
    }

    valuef mu_h = (mu_cfl + pressure_cfl) * W2 - pressure_cfl;

    as_ref(mu_h_cfl_out[pos, dim]) += mu_h;

    for(int i=0; i < 3; i++)
    {
        as_ref(Si_out[i][pos, dim]) += cSi[i];
    }
}

void neutron_star::boot_solver(cl::context ctx)
{
    cl::async_build_and_cache(ctx, [=] {
        return value_impl::make_function(matter_accum, "matter_accum");
    }, {"matter_accum"});

    cl::async_build_and_cache(ctx, [=] {
        return value_impl::make_function(matter_p2, "matter_p2");
    }, {"matter_p2"});
}

void neutron_star::data::add_to_solution(cl::context& ctx, cl::command_queue& cqueue,
                                         discretised_initial_data& dsol,
                                         tensor<int, 3> dim, float scale, int index)
{
    std::vector<cl::buffer> AIJ_temp;

    for(int i=0; i < 6; i++)
    {
        AIJ_temp.emplace_back(ctx);

        AIJ_temp.back().alloc(sizeof(cl_float) * dim.x() * dim.y() * dim.z());
    }

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

    t3f linear_momentum;

    if(params.linear_momentum.momentum)
        linear_momentum = params.linear_momentum.momentum.value();

    if(params.linear_momentum.dimensionless)
    {
        dimensionless_linear_momentum dam = params.linear_momentum.dimensionless.value();

        linear_momentum = dam.axis * dam.x * total_mass;
    }

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
        for(int i=0; i < 6; i++)
            AIJ_temp[i].set_to_zero(cqueue);

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
        args.push_back((t3f)linear_momentum);
        args.push_back((t3f)angular_momentum);
        args.push_back(dsol.AIJ_cfl[0], dsol.AIJ_cfl[1], dsol.AIJ_cfl[2], dsol.AIJ_cfl[3], dsol.AIJ_cfl[4], dsol.AIJ_cfl[5]);
        args.push_back(AIJ_temp[0], AIJ_temp[1], AIJ_temp[2], AIJ_temp[3], AIJ_temp[4], AIJ_temp[5]);
        args.push_back(dsol.mu_h_cfl);
        args.push_back(dsol.Si_cfl[0], dsol.Si_cfl[1], dsol.Si_cfl[2]);
        args.push_back(dsol.star_indices);
        args.push_back(index);

        cqueue.exec("matter_accum", args, {dim.x(), dim.y(), dim.z()}, {8,8,1});
        cqueue.exec("matter_p2", args, {dim.x(), dim.y(), dim.z()}, {8,8,1});
    }
}

neutron_star::data::data(const parameters& p) : params(p)
{
    tov::eos::polytrope* tov_params = new tov::eos::polytrope(params.Gamma, params.K.msols.value());
    eos = tov_params;

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

        std::vector<double> masses = tov::search_for_rest_mass(mass.mass, *tov_params);

        assert(masses.size() > 0 && mass.result_index >= 0 && mass.result_index < (int)masses.size());

        p0_msols = masses.at(mass.result_index);
    }
    else
        assert(false);

    start = tov::make_integration_state(p0_msols, 1e-6, *tov_params);
    sol = tov::solve_tov(start, *tov_params, 1e-6, 0).value();

    total_mass = sol.M_msol;
    stored = get_eos();
}

neutron_star::numerical_eos neutron_star::data::get_eos()
{
    numerical_eos ret;

    float max_density = p0_msols * 4;

    ret.max_density = max_density;

    ret.max_mu = eos->p0_to_mu(max_density);

    int max_samples = 400;

    for(int i=0; i < max_samples; i++)
    {
        double density = ((double)i / max_samples) * max_density;

        double pressure = eos->p0_to_P(density);

        ret.pressure.push_back(pressure);
    }

    for(int i=0; i < max_samples; i++)
    {
        double mu = ((double)i / max_samples) * ret.max_mu;

        double p0 = eos->mu_to_p0(mu);

        ret.mu_to_p0.push_back(p0);
    }

    return ret;
}

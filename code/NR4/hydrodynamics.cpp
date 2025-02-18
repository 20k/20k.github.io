#include "hydrodynamics.hpp"
#include "init_general.hpp"

#define DIVISION_TOL 0.0001f

///so like. What if I did the projective real strategy?

template<typename T, typename U>
inline
T divide_with_limit(const T& top, const T& bottom, const U& limit, float tol = DIVISION_TOL)
{
    return ternary(bottom >= tol, top / bottom, T{limit});
}

valuef get_Gamma()
{
    return 2;
}

valuef get_h_with_gamma_eos(valuef e)
{
    return 1 + get_Gamma() * e;
}

valuef e_star_to_epsilon(valuef p_star, valuef e_star, valuef W, valuef w)
{
    valuef e_m6phi = W*W*W;
    valuef Gamma = get_Gamma();

    return pow(divide_with_limit(e_m6phi, w, 0.f, 0.00001f), Gamma - 1) * pow(e_star, Gamma) * pow(p_star, Gamma - 2);

    //return pow(divide_with_limit(e_m6phi * pow(e_star, Gamma) * pow(p_star, Gamma - 2), w, 0.f, 0.000001f), Gamma-1);
    //return pow(divide_with_limit(e_m6phi * pow(e_star, Gamma) * pow(p_star, Gamma - 2), w, 0.f, 0.000001f), Gamma-1);

    //return pow(e_m6phi / max(w, 0.001f), Gamma-1) * pow(e_star, Gamma) * pow(p_star, Gamma - 2);
}

valuef calculate_p0e(valuef Gamma, valuef W, valuef w, valuef p_star, valuef e_star)
{
    //valuef iv_au0 = p_star / max(w, 0.001f);

    valuef iv_au0 = divide_with_limit(p_star, w, 0.f, 0.00001f);

    valuef e_m6phi = W*W*W;

    return pow(max(e_star * e_m6phi * iv_au0, 0.f), Gamma);
}

valuef gamma_eos(valuef Gamma, valuef W, valuef w, valuef p_star, valuef e_star)
{
    return calculate_p0e(Gamma, W, w, p_star, e_star) * (Gamma - 1);
}

template<typename T>
valuef full_hydrodynamic_args<T>::adm_p(bssn_args& args, const derivative_data& d)
{
    valuef lw = w[d.pos, d.dim];
    //valuef lP = P[d.pos, d.dim];
    valuef es = max(e_star[d.pos, d.dim], 0.f);
    valuef ps = max(p_star[d.pos, d.dim], 0.f);

    valuef epsilon = e_star_to_epsilon(ps, es, args.W, lw);

    valuef h = get_h_with_gamma_eos(epsilon);

    return h * lw * (args.W * args.W * args.W) - gamma_eos(get_Gamma(), args.W, lw, ps, es);
}

template<typename T>
tensor<valuef, 3> full_hydrodynamic_args<T>::adm_Si(bssn_args& args, const derivative_data& d)
{
    v3f cSi = {Si[0][d.pos, d.dim], Si[1][d.pos, d.dim], Si[2][d.pos, d.dim]};

    return pow(args.W, 3.f) * cSi;
}

template<typename T>
tensor<valuef, 3, 3> full_hydrodynamic_args<T>::adm_W2_Sij(bssn_args& args, const derivative_data& d)
{
    valuef ps =  max(p_star[d.pos, d.dim], 0.f);
    valuef es =  max(e_star[d.pos, d.dim], 0.f);
    v3f cSi = {Si[0][d.pos, d.dim], Si[1][d.pos, d.dim], Si[2][d.pos, d.dim]};
    valuef lw = w[d.pos, d.dim];
    //valuef lP = P[d.pos, d.dim];

    //valuef lP = calculate_p0e(get_Gamma(), args.W, lw, ps, es) * (get_Gamma() - 1);

    valuef lP = gamma_eos(get_Gamma(), args.W, lw, ps, es);

    valuef epsilon = e_star_to_epsilon(ps, es, args.W, lw);

    valuef h = get_h_with_gamma_eos(epsilon);

    tensor<valuef, 3, 3> W2_Sij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            W2_Sij[i, j] = divide_with_limit(pow(args.W, 5.f) * cSi[i] * cSi[j], lw * h, 0.f, 0.000001f);
            //W2_Sij[i, j] = (pow(args.W, 5.f) / max(lw * h, 0.001f)) * cSi[i] * cSi[j];
        }
    }

    return (W2_Sij + lP * args.cY.to_tensor());
}

//there's *some* interaction between the division constants here, that I need to pin down
v3f calculate_vi(valuef gA, v3f gB, valuef W, valuef w, valuef epsilon, v3f Si, const unit_metric<valuef, 3, 3>& cY)
{
    valuef h = get_h_with_gamma_eos(epsilon);

    //note to self, actually hand derived this and am sure its correct
    return -gB + (divide_with_limit(W*W * gA, w*h, 0.f)) * cY.invert().raise(Si);
    //return -gB + ((W*W * gA * cY.invert().raise(Si)) / max(w*h, 0.001f));
}

template<typename T>
valuef full_hydrodynamic_args<T>::dbg(bssn_args& args, const derivative_data& d)
{
    //return fabs(p_star[d.pos, d.dim]) * 10;
    //return sqrt(pow(Si[0][d.pos, d.dim], 2.f) + pow(Si[1][d.pos, d.dim], 2.f)) * 10;
    //return sqrt(pow(Si[0][d.pos, d.dim], 2.f) + pow(Si[2][d.pos, d.dim], 2.f)) * 100;
    //return e_star[d.pos, d.dim] * 0.5;
    return fabs(Si[0][d.pos, d.dim]) * 100 * 100;
}

template struct full_hydrodynamic_args<buffer<valuef>>;
template struct full_hydrodynamic_args<buffer_mut<valuef>>;

std::vector<buffer_descriptor> hydrodynamic_buffers::get_description()
{
    buffer_descriptor p;
    p.name = "p*";
    p.dissipation_coeff = 0.05;

    buffer_descriptor e;
    e.name = "e*";
    e.dissipation_coeff = 0.05;

    buffer_descriptor s0;
    s0.name = "cs0";
    s0.dissipation_coeff = 0.05;

    buffer_descriptor s1;
    s1.name = "cs1";
    s1.dissipation_coeff = 0.05;

    buffer_descriptor s2;
    s2.name = "cs2";
    s2.dissipation_coeff = 0.05;

    return {p, e, s0, s1, s2};
}

std::vector<cl::buffer> hydrodynamic_buffers::get_buffers()
{
    return {p_star, e_star, Si[0], Si[1], Si[2]};
}

void hydrodynamic_buffers::allocate(cl::context ctx, cl::command_queue cqueue, t3i size)
{
    int64_t cells = int64_t{size.x()} * size.y() * size.z();

    p_star.alloc(sizeof(cl_float) * cells);
    e_star.alloc(sizeof(cl_float) * cells);

    p_star.set_to_zero(cqueue);
    e_star.set_to_zero(cqueue);

    for(auto& i : Si)
    {
        i.alloc(sizeof(cl_float) * cells);
        i.set_to_zero(cqueue);
    }
}

std::vector<buffer_descriptor> hydrodynamic_utility_buffers::get_description()
{
    buffer_descriptor P;
    P.name = "P";

    buffer_descriptor w;
    w.name = "w";

    buffer_descriptor vi0;
    vi0.name = "vi0";
    buffer_descriptor vi1;
    vi1.name = "vi1";
    buffer_descriptor vi2;
    vi2.name = "vi2";

    return {w, P, vi0, vi1, vi2};
}

std::vector<cl::buffer> hydrodynamic_utility_buffers::get_buffers()
{
    return {w, P, vi[0], vi[1], vi[2]};
}

void hydrodynamic_utility_buffers::allocate(cl::context ctx, cl::command_queue cqueue, t3i size)
{
    int64_t cells = int64_t{size.x()} * size.y() * size.z();

    P.alloc(sizeof(cl_float) * cells);
    w.alloc(sizeof(cl_float) * cells);

    P.set_to_zero(cqueue);
    w.set_to_zero(cqueue);

    for(auto& i : vi)
    {
        i.alloc(sizeof(cl_float) * cells);
        i.set_to_zero(cqueue);
    }
}

struct eos_gpu : value_impl::single_source::argument_pack
{
    buffer<valuef> pressures;
    buffer<valuef> densities;
    literal<valuei> pressure_stride;
    literal<valuei> eos_count;

    void build(auto& in)
    {
        using namespace value_impl::builder;

        add(pressures, in);
        add(densities, in);
        add(pressure_stride, in);
        add(eos_count, in);
    }
};

valuef calculate_w(valuef p_star, valuef e_star, valuef W, valuef Gamma, inverse_metric<valuef, 3, 3> icY, v3f Si);

void init_hydro(execution_context& ectx, bssn_args_mem<buffer<valuef>> in, full_hydrodynamic_args<buffer_mut<valuef>> hydro, literal<v3i> ldim, literal<valuef> scale,
                buffer<valuef> mu_cfl_b, buffer<valuef> mu_h_cfl_b, buffer<valuef> pressure_cfl_b, buffer<valuef> cfl_b, buffer<valuef> u_correction_b, std::array<buffer<valuef>, 3> Si_cfl_b,
                buffer<valuei> indices, eos_gpu eos_data)
{
    using namespace single_source;

    valuei x = value_impl::get_global_id(0);
    valuei y = value_impl::get_global_id(1);
    valuei z = value_impl::get_global_id(2);

    pin(x);
    pin(y);
    pin(z);

    v3i dim = ldim.get();

    if_e(x >= dim.x() || y >= dim.y() || z >= dim.z(), []{
        return_e();
    });

    v3i pos = {x, y, z};
    pin(pos);

    valuei index = indices[pos, dim];

    if_e(index == -1, [&]{
        return_e();
    });

    valuef max_density = eos_data.densities[index];

    bssn_args args(pos, dim, in);

    auto pressure_to_p0 = [&](valuef P)
    {
        valuei offset = index * eos_data.pressure_stride.get();

        mut<valuei> i = declare_mut_e(valuei(0));
        mut<valuef> out = declare_mut_e(valuef(0));

        for_e(i < eos_data.pressure_stride.get() - 1, assign_b(i, i+1), [&]{
            valuef p1 = eos_data.pressures[offset + i];
            valuef p2 = eos_data.pressures[offset + i + 1];

            if_e(P >= p1 && P <= p2, [&]{
                valuef val = (P - p1) / (p2 - p1);

                as_ref(out) = (((valuef)i + val) / (valuef)eos_data.pressure_stride.get()) * max_density;

                break_e();
            });
        });

        return declare_e(out);
    };

    auto p0_to_pressure = [&](valuef p0)
    {
        valuei offset = index * eos_data.pressure_stride.get();

        valuef idx = clamp((p0 / max_density) * (valuef)eos_data.pressure_stride.get(), valuef(0), (valuef)eos_data.pressure_stride.get() - 2);

        valuei fidx = (valuei)idx;

        return mix(eos_data.pressures[offset + fidx], eos_data.pressures[offset + fidx + 1], idx - floor(idx));
    };

    //valuef mu_cfl = mu_cfl_b[pos, dim];
    valuef mu_h_cfl = mu_h_cfl_b[pos, dim];
    valuef pressure_cfl = pressure_cfl_b[pos, dim];
    valuef phi = cfl_b[pos, dim] + u_correction_b[pos, dim];
    v3f Si_cfl = {Si_cfl_b[0][pos, dim], Si_cfl_b[1][pos, dim], Si_cfl_b[2][pos, dim]};

    //valuef mu = mu_cfl * pow(phi, -8);
    valuef mu_h = mu_h_cfl * pow(phi, -8);
    valuef pressure_from_cfl = pressure_cfl * pow(phi, -8);
    v3f Si = Si_cfl * pow(phi, -10);

    //valuef u0 = sqrt((mu_h + pressure) / max(mu + pressure, 0.001f));

    valuef Gamma = get_Gamma();

    ///ok, lets operate under the assumption that W = 1, and the pressure is wrong
    ///that means that mu = ph, so we can assume that mu is correct
    auto mu_to_p0 = [&](valuef mu)
    {
        ///mu = p0 + f(p0) / (Gamma-1)
        mut<valuei> i = declare_mut_e(valuei(0));
        mut<valuef> out = declare_mut_e(valuef(0));

        int steps = 400;

        for_e(i < steps, assign_b(i, i+1), [&]{
            valuef frac = (valuef)i / steps;
            valuef frac2 = (valuef)(i + 1) / steps;

            valuef d0 = frac * max_density;
            valuef d1 = frac2 * max_density;

            pin(d0);
            pin(d1);

            valuef mu0 = d0 + p0_to_pressure(d0) / (Gamma - 1);
            valuef mu1 = d1 + p0_to_pressure(d1) / (Gamma - 1);

            pin(mu0);
            pin(mu1);

            if_e(mu >= mu0 && mu <= mu1, [&]{
                //print("hello %f\n", mu);

                valuef frac = (mu - mu0) / (mu1 - mu0);

                as_ref(out) = mix(d0, d1, frac);
                break_e();
            });
        });

        return declare_e(out);
    };

    auto get_mu_for = [&](valuef muh, valuef W)
    {
        ///solve the equation muh = (mu + f(mu)) W^2 - mu
        ///W >= 1
        ///f(mu) >= 0
        ///mu >= 0, obviously
        ///(mu + f(mu)) W^2 > mu. I don't use this, but you might be able to do something sane
        ///you may be able to solve this with fixed point iteration by solving for mu on the rhs trivially
        ///we'll also have a good guess for mu, which seems a shame to waste
        mut<valuei> i = declare_mut_e(valuei(0));
        mut<valuef> out = declare_mut_e(valuef(0));

        int steps = 400;

        valuef max_mu = muh * 10;

        for_e(i < steps, assign_b(i, i+1), [&]{
            valuef f1 = (valuef)i / steps;
            valuef f2 = (valuef)(i + 1) / steps;

            valuef test_mu1 = f1 * max_mu;
            valuef test_mu2 = f2 * max_mu;

            pin(test_mu1);
            pin(test_mu2);

            valuef p0_1 = mu_to_p0(test_mu1);
            valuef p0_2 = mu_to_p0(test_mu2);

            pin(p0_1);
            pin(p0_2);

            valuef p_1 = p0_to_pressure(p0_1);
            valuef p_2 = p0_to_pressure(p0_2);

            pin(p_1);
            pin(p_2);

            valuef test_muh1 = (test_mu1 + p_1) * W*W - p_1;
            valuef test_muh2 = (test_mu2 + p_2) * W*W - p_2;

            pin(test_muh1);
            pin(test_muh2);

            /*if_e(pos.x() == dim.x()/2 + 2 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
                print("hello searching %f lower %f upper %f test1 %f test2 %f\n", muh, test_muh1, test_muh2, test_mu1, test_mu2);
            });*/

            if_e(muh >= test_muh1 && muh <= test_muh2, [&]{

                valuef frac = (muh - test_muh1) / (test_muh2 - test_muh1);

                as_ref(out) = mix(test_mu1, test_mu2, frac);
                break_e();
            });
        });

        return declare_e(out);
    };

    valuef cW = max(args.W, 0.0001f);
    metric<valuef, 3, 3> Yij = args.cY / (cW*cW);

    valuef ysj = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            ysj += Yij[i, j] * Si[i] * Si[j];
        }
    }

    pin(ysj);

    valuef u0 = 1;
    valuef mu = mu_h;

    for(int i=0; i < 100; i++)
    {
        //yijsisj = (mu + p)^2 W^2 (W^2 - 1)
        //(yijsisj) / (mu+p)^2 = W^4 - W^2
        //C = W^4 - W^2
        //W = sqrt(1 + sqrt(4C + 1)) / sqrt(2)

        valuef p0 = mu_to_p0(mu);
        valuef pressure = p0_to_pressure(mu);

        pin(p0);
        pin(pressure);

        valuef C = ysj / pow(mu + pressure, 2.f);
        valuef next_W = sqrt(1 + sqrt(4 * C + 1)) / sqrtf(2.f);

        /*if_e(pos.x() == dim.x()/2 + 2 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
            print("u0 %f\n", u0);
            print("mu %f\n", mu);

            print("Err %f %f %f", ysj - pow(mu + pressure, 2) * u0 * u0 * (u0 * u0 - 1), mu_h - ((mu + pressure) * u0 * u0 - pressure), next_W);
        });*/

        u0 = next_W;
        mu = get_mu_for(mu_h, u0);
        pin(u0);
        pin(mu);

        //we have u0, now lets solve for mu
        //mu_h = (mu + f(mu)) * W^2 - f(mu)
    }

    valuef p0 = mu_to_p0(mu);

    valuef pressure = p0_to_pressure(p0);

    /*if_e(pos.x() == dim.x()/2 + 2 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        print("Pressure 1 %f %f\n", pressure, pressure_from_cfl);
    });*/

    //valuef pressure = pressure_from_cfl;

    valuef p0_e = pressure / (Gamma - 1);

    //print("Test pressure %f cfl pressure %f\n", p0_e, pressure_from_cfl);

    ///mu = p0 + P/(Gamma-1)
    ///mu = p0 + f(p0) / (Gamma-1)
    //valuef p0 = pressure_to_p0(pressure);
    ///P = p0(1 + eps)

    ///(mu + P) W^2 - P
    ///W = 1
    ///mu = ph


    //valuef p0 = pow(pressure / 123.741, 2.f);


    value gA = args.gA;

    //fluid dynamics cannot have a singular initial slice, so setting the clamping pretty high here because its irrelevant
    //thing is we have 0 quantities at the singularity, so as long as you don't generate a literal NaN here, you're 100% fine

    valuef p_star = p0 * gA * u0 * pow(cW, -3);
    valuef e_star = pow(p0_e, (1/Gamma)) * gA * u0 * pow(cW, -3);

    /*valuef yijSiSj = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            yijSiSj += Yij[i, j] * Si[i] * Si[j];
        }
    }

    valuef w = u0;
    valuef calc = pow(mu + pressure, 2.f) * w*w * (w*w - 1);

    if_e(pos.x() == dim.x()/2 + 15 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        value_base se;
        se.type = value_impl::op::SIDE_EFFECT;
        se.abstract_value = "printf(\"left %f, right %f\\n\"," + value_to_string(yijSiSj) + "," + value_to_string(calc) + ")";

        value_impl::get_context().add(se);
    });*/

    //valuef w = u0;
    //valuef calc_ph = (mu + pressure) * w*w - pressure;

    /*if_e(pos.x() == dim.x()/2 + 1 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        value_base se;
        se.type = value_impl::op::SIDE_EFFECT;
        se.abstract_value = "printf(\"left %f, right %f\\n\"," + value_to_string(mu) + "," + value_to_string(mu_h) + ")";

        value_impl::get_context().add(se);
    });*/

    v3f Si_lo_cfl = pow(cW, -3) * Yij.lower(Si);

    if_e(pos.x() == dim.x()/2 + 2 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        //print("Muh %f si %f %f %f\n", mu_h, Si_lo_cfl[0], Si_lo_cfl[1], Si_lo_cfl[2]);
        //print("Cfl? %f\n", phi);
    });

    as_ref(hydro.p_star[pos, dim]) = p_star;
    as_ref(hydro.e_star[pos, dim]) = e_star;
    as_ref(hydro.Si[0][pos, dim]) = Si_lo_cfl[0];
    as_ref(hydro.Si[1][pos, dim]) = Si_lo_cfl[1];
    as_ref(hydro.Si[2][pos, dim]) = Si_lo_cfl[2];

    valuef real_w = p_star * gA * u0;

    ///ok so. I'm trying to answer the quesiton of why w = p* gA u0
    ///is not the same answer as calculate_w
    valuef w = calculate_w(p_star, e_star, args.W, Gamma, args.cY.invert(), Si_lo_cfl);

    as_ref(hydro.w[pos, dim]) = w;

    as_ref(hydro.P[pos, dim]) = gamma_eos(Gamma, args.W, w, p_star, e_star);
    //as_ref(hydro.P[pos, dim]) = (Gamma - 1) * p0_e;


    ///ok: What've I've discovered
    ///Si isn't well defined when gA != 1 in our init conditions
    /*if_e(pos.x() == dim.x()/2 + 10 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        //print("First %f", p_star * gA * u0);
        print("P %.23f p_star %f e_star %f calc w %f numerical w %f W %f\n", hydro.P[pos, dim], p_star, e_star, real_w, w, args.W);
        //print("Si %f %f %f\n", Si_lo_cfl[0], Si_lo_cfl[1], Si_lo_cfl[2]);
    });*/

}

struct hydrodynamic_concrete
{
    valuef p_star;
    valuef e_star;
    v3f Si;

    valuef w;
    valuef P;

    template<typename T>
    hydrodynamic_concrete(v3i pos, v3i dim, full_hydrodynamic_args<T> args)
    {
        p_star = max(args.p_star[pos, dim], 0.f);
        e_star = max(args.e_star[pos, dim], 0.f);
        Si = {args.Si[0][pos, dim], args.Si[1][pos, dim], args.Si[2][pos, dim]};
        w = args.w[pos, dim];
        P = args.P[pos, dim];
    }

    template<typename T, typename U>
    hydrodynamic_concrete(v3i pos, v3i dim, hydrodynamic_base_args<T> bargs, hydrodynamic_utility_args<U> uargs)
    {
        p_star = max(bargs.p_star[pos, dim], 0.f);
        e_star = max(bargs.e_star[pos, dim], 0.f);
        Si = {bargs.Si[0][pos, dim], bargs.Si[1][pos, dim], bargs.Si[2][pos, dim]};
        w = uargs.w[pos, dim];
        P = uargs.P[pos, dim];
    }
};

valuef w_next_interior(valuef p_star, valuef e_star, valuef W, valuef w_prev, valuef Gamma)
{
    valuef chi = W*W;
    valuef em_6_phi_G = pow(chi, Gamma - 1);

    valuef geg = Gamma * pow(e_star, Gamma);

    //valuef A = pow(max(W, 0.00001f), 3.f * Gamma - 3.f);
    valuef wG = pow(w_prev, Gamma - 1);

    return divide_with_limit(wG, wG + em_6_phi_G * geg * pow(p_star, Gamma - 2), 0.f, 0.000001f);

    //return wG / max(wG + A * Gamma * pow(e_star, Gamma) * pow(max(p_star, 0.00001f), Gamma - 2), 0.0001f);
}

///i'm getting slightyl different results between this method, and the analytic methd for the initial conditions
///checkme thoroughly
///have a feeling it might be pure numerical problems
valuef calculate_w(valuef p_star, valuef e_star, valuef W, valuef Gamma, inverse_metric<valuef, 3, 3> icY, v3f Si)
{
    using namespace single_source;

    valuef w = 0.5f;

    valuef p_sq = p_star * p_star;
    valuef cst = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cst += icY[i, j] * Si[i] * Si[j];
        }
    }

    cst = W*W * cst;

    //pin(p_sq);
    pin(cst);

    for(int i=0; i < 140; i++)
    {
        valuef D = w_next_interior(p_star, e_star, W, w, Gamma);

        valuef w_next = sqrt(max(p_sq + cst * D*D, 0.f));

        pin(w_next);

        w = w_next;
    }

    return w;
}

valuef w2_m_p2(valuef p_star, valuef e_star, valuef W, valuef Gamma, inverse_metric<valuef, 3, 3> icY, v3f Si, valuef w)
{
    valuef p_sq = p_star * p_star;
    valuef cst = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cst += icY[i, j] * Si[i] * Si[j];
        }
    }

    cst = W*W * cst;

    valuef D = w_next_interior(p_star, e_star, W, w, Gamma);

    return max(cst * D*D, 0.f);

}

constexpr float min_p_star = 1e-8f;

///todo: i need to de-mutify hydro
void calculate_hydro_intermediates(execution_context& ectx, bssn_args_mem<buffer<valuef>> in, full_hydrodynamic_args<buffer_mut<valuef>> hydro,
                                   literal<v3i> idim, literal<valuef> scale,
                                   buffer<tensor<value<short>, 3>> positions, literal<valuei> positions_length)
{
    using namespace single_source;

    valuei lid = value_impl::get_global_id(0);

    pin(lid);

    v3i dim = idim.get();

    if_e(lid >= positions_length.get(), []{
        return_e();
    });

    v3i pos = (v3i)positions[lid];
    pin(pos);

    hydrodynamic_concrete hydro_args(pos, dim, hydro);

    if_e(hydro_args.p_star <= min_p_star, [&]{
        as_ref(hydro.P[pos, dim]) = valuef(0);
        as_ref(hydro.w[pos, dim]) = valuef(0);
        as_ref(hydro.vi[0][pos, dim]) = valuef(0);
        as_ref(hydro.vi[1][pos, dim]) = valuef(0);
        as_ref(hydro.vi[2][pos, dim]) = valuef(0);

        return_e();
    });

    bssn_args args(pos, dim, in);

    valuef w = calculate_w(hydro_args.p_star, hydro_args.e_star, args.W, get_Gamma(), args.cY.invert(), hydro_args.Si);
    w = max(w, hydro_args.p_star * args.gA * 1);

    valuef P = gamma_eos(get_Gamma(), args.W, w, hydro_args.p_star, hydro_args.e_star);

    ///hmmmm.. si?
    /*if_e(pos.x() == dim.x()/2 + 10 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{

        print("P %.23f p_star %f e_star %f w %f W %f Si %f %f %f\n", hydro.P[pos, dim], hydro_args.p_star, hydro_args.e_star, w, args.W, hydro_args.Si[0], hydro_args.Si[1], hydro_args.Si[2]);
        //print("First %f", p_star * gA * u0);
        //print("P %.23f\n", hydro.P[pos, dim]);
        //print("Si %f %f %f\n", Si_lo_cfl[0], Si_lo_cfl[1], Si_lo_cfl[2]);
    });*/

    /*if_e(pos.x() == dim.x()/2 + 20 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        value_base se;
        se.type = value_impl::op::SIDE_EFFECT;
        se.abstract_value = "printf(\"p %f pstar %f\\n\"," + value_to_string(P) + "," + value_to_string(hydro_args.p_star) + ")";

        value_impl::get_context().add(se);
    });*/

    valuef epsilon = e_star_to_epsilon(hydro_args.p_star, hydro_args.e_star, args.W, w);

    as_ref(hydro.w[pos, dim]) = w;
    as_ref(hydro.P[pos, dim]) = P;

    v3f vi = calculate_vi(args.gA, args.gB, args.W, w, epsilon, hydro_args.Si, args.cY);

    //vi = clamp(vi, -0.05f, 0.05f);

    /*if_e(pos.x() == dim.x()/2 + 20 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        print("Vi %f %f %f p* %.23f e* %.23f\n", vi[0], vi[1], vi[2], hydro_args.p_star, hydro_args.e_star);
    });*/

    as_ref(hydro.vi[0][pos, dim]) = vi[0];
    as_ref(hydro.vi[1][pos, dim]) = vi[1];
    as_ref(hydro.vi[2][pos, dim]) = vi[2];
}

void evolve_hydro(execution_context& ectx, bssn_args_mem<buffer<valuef>> in,
                  hydrodynamic_base_args<buffer<valuef>> h_base, hydrodynamic_base_args<buffer<valuef>> h_in, hydrodynamic_base_args<buffer_mut<valuef>> h_out,
                  hydrodynamic_utility_args<buffer<valuef>> util,
                  literal<v3i> idim, literal<valuef> scale, literal<valuef> timestep,
                  buffer<tensor<value<short>, 3>> positions, literal<valuei> positions_length)
{
    using namespace single_source;

    valuei lid = value_impl::get_global_id(0);

    pin(lid);

    v3i dim = idim.get();

    if_e(lid >= positions_length.get(), []{
        return_e();
    });

    v3i pos = (v3i)positions[lid];
    pin(pos);

    bssn_args args(pos, dim, in);
    hydrodynamic_concrete hydro_args(pos, dim, h_in, util);

    derivative_data d;
    d.pos = pos;
    d.dim = idim.get();
    d.scale = scale.get();

    valuef p_star = hydro_args.p_star;
    valuef e_star = hydro_args.e_star;
    v3f Si = hydro_args.Si;
    valuef w = hydro_args.w;

    valuef epsilon = e_star_to_epsilon(p_star, e_star, args.W, w);

    //valuef p0e = calculate_p0e(get_Gamma(), args.W, w, p_star, e_star);
    //valuef p0 = p0e / epsilon;

    /*if_e((pos.x() == dim.x()/2 + 2) && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        print("epsilon %f e_star %f Pressure %f p0 %f p_star %f w %f\n", epsilon, e_star, hydro_args.P, p0, p_star, w);
    });*/

    //v3f vi = calculate_vi(args.gA, args.gB, args.W, w, epsilon, Si, args.cY);

    v3f vi = {util.vi[0][pos, dim], util.vi[1][pos, dim], util.vi[2][pos, dim]};

    auto leib = [&](valuef v1, valuef v2, int i)
    {
        //return diff1(v1 * v2, i, d);

        return diff1(v1, i, d) * v2 + diff1(v2, i, d) * v1;
    };

    ///things that are odd
    ///u_k is very high, but vi is very low. Indicates an extreme metric tensor that feels unjustified
    ///if u_k is so high, why doesn't the stuff drift away?
    ///dgA is the primary driver of the Sk increase

    valuef dp_star = 0;
    valuef de_star = 0;
    v3f dSi_p1;

    for(int i=0; i < 3; i++)
    {
        //dp_star += diff1(p_star * vi[i], i, d);
        //de_star += diff1(e_star * vi[i], i, d);

        dp_star += leib(p_star, vi[i], i);
        de_star += leib(e_star, vi[i], i);

        for(int k=0; k < 3; k++)
        {
            dSi_p1[k] += leib(Si[k], vi[i], i);
            //dSi_p1[k] += diff1(Si[k] * vi[i], i, d);
        }
    }

    #if 0
    {
        tensor<valuef, 3, 3> SkVi;

        for(int i=0; i < 3; i++)
        {
            for(int k=0; k < 3; k++)
            {
                SkVi[k, i] = Si[k] * vi[i];
            }
        }

        for(int k=0; k < 3; k++)
        {
            valuef sum = 0;

            for(int i=0; i < 3; i++)
            {
                sum += diff1(SkVi[k, i], i, d);
            }

            dSi_p1[k] = sum;

            /*if(k == 0)
            {
                if_e((pos.x() == dim.x()/2 + 19) && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
                    print("dSi %.10f\n", dSi_p1[k]);
                });
            }*/
        }
    }
    #endif

    dp_star = -dp_star;
    de_star = -de_star;
    dSi_p1 = -dSi_p1;

    #if 0
    auto calculate_PQvis = [&](const valuef& gA, const v3f& gB, const inverse_metric<valuef, 3, 3>& icY, const valuef& W, const valuef& w)
    {
        valuef e_m6phi = pow(W, 3.f);
        valuef e_6phi = pow(max(W, 0.001f), -3.f);

        valuef dkvk = 0;

        for(int k=0; k < 3; k++)
        {
            dkvk += 2 * diff1(vi[k], k, d);
        }

        valuef littledv = dkvk * scale.get();
        valuef Gamma = get_Gamma();

        valuef A = divide_with_limit(pow(e_star, Gamma) * pow(p_star, Gamma - 1) * pow(e_m6phi, Gamma - 1), pow(w, Gamma - 1), 0.f, 0.000001f);

        //ctx.add("DBG_A", A);

        ///[0.1, 1.0}
        valuef CQvis = 1.f;

        valuef PQvis = ternary(littledv < 0, CQvis * A * pow(littledv, 2), valuef{0.f});

        return PQvis;
    };

    auto e_star_rhs = [&]() -> valuef
    {
        valuef e_m6phi = pow(args.W, 3.f);

        valuef PQvis = calculate_PQvis(args.gA, args.gB, args.cY.invert(), args.W, w);

        valuef sum_interior_rhs = 0;

        for(int k=0; k < 3; k++)
        {
            value to_diff = divide_with_limit(w * vi[k], p_star * e_m6phi, 0.f, 0.000001f);

            sum_interior_rhs += diff1(to_diff, k, d);
        }

        valuef Gamma = get_Gamma();

        valuef p0e = calculate_p0e(Gamma, args.W, w, p_star, e_star);

        valuef degenerate = divide_with_limit(valuef{1}, pow(p0e, 1 - 1/Gamma), 0.f, 0.000001f);

        return -degenerate * (PQvis / Gamma) * sum_interior_rhs;
    };


    de_star += e_star_rhs() * 0;
    #endif

    valuef h = get_h_with_gamma_eos(epsilon);

    //so: best guess
    ///u_i is becoming extremely high for some reason
    ///then again, does it have any meaningful value vs v^i? I guess yes in the sense that it
    ///contributes to the stress energy tensor
    /*if_e((pos.x() == dim.x()/2 + 19) && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        valuef u0 = w / (p_star * args.gA);

        valuef u1 = Si[0] / (h * p_star);

        valuef dvi0 = diff1(vi[0], 0, d);
        valuef dvi1 = diff1(vi[1], 1, d);
        valuef dvi2 = diff1(vi[2], 2, d);

        print("Si %.10f p* %f e* %f u1 %f u0 %f h %f vi %.12f %.12f %.12f W %f gA %f Display value %f\n", Si[0], p_star, e_star, u1, u0, h, dvi0, dvi1, dvi2, args.W, args.gA, Si[0] * 100 * 100);
        //print("Si %.10f %f %f p* %f e* %f u1 %f u0 %f h %f epsilon %.16f Display value %f\n", Si[0], Si[1], Si[2], p_star, e_star, u1, u0, h, epsilon, Si[0] * 100 * 100);
    });*/

    /*if_e((pos.x() == dim.x()/2 + 19 || pos.x() == dim.x()/2 + 18 || pos.x() == dim.x()/2 + 20) && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        print("gA %f Display value %f\n",args.gA, Si[0] * 100 * 100);
        //print("Si %.10f %f %f p* %f e* %f u1 %f u0 %f h %f epsilon %.16f Display value %f\n", Si[0], Si[1], Si[2], p_star, e_star, u1, u0, h, epsilon, Si[0] * 100 * 100);
    });*/

    ///todo: think I've figured it out. epsilon is huge, p0 is zero effectively
    ///i think I'm clamping e*, but i also need to recalculate and clamp sk
    ///hmm. there's a bit of a discontinuity. Maybe it is enough?
    ///hmmmm. Clamping e* isn't the problem. An interesting observation is that
    ///clamping e* would cause u_k to be proportionally higher, as we don't clamp Sk
    #if 0
    if_e((pos.x() == dim.x()/2 + 18) && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        valuef u0 = w / (p_star * args.gA);

        valuef u1 = Si[0] / (h * p_star);

        valuef dvi0 = diff1(vi[0], 0, d);
        valuef dvi1 = diff1(vi[1], 1, d);
        valuef dvi2 = diff1(vi[2], 2, d);

        valuef p0e = calculate_p0e(get_Gamma(), args.W, w, p_star, e_star);
        valuef p0 = p0e / epsilon;

        print("vi %f %f %f Si %f %f %f u1 %f display %f\n", vi[0], vi[1], vi[2], Si[0], Si[1], Si[2], u1, Si[0] * 100 * 100);

        //print("P0 %f epsilon %f p* %.12f e* %.12f e*/p* %.12f h %f display value %f\n", p0, epsilon, p_star, e_star, e_star/p_star, h, Si[0] * 100 * 100);

        //print("Si %.10f %f %f p* %f e* %f u1 %f u0 %f h %f epsilon %.16f Display value %f\n", Si[0], Si[1], Si[2], p_star, e_star, u1, u0, h, epsilon, Si[0] * 100 * 100);
    });
    #endif

    ///SO
    ///Si[0] is becoming degenerate at the boundary of the neutron star for.... some ungodly reason
    ///i wonder if systematic errors with the calculation of w, ala p*^2 could do it
    for(int k=0; k < 3; k++)
    {
        valuef p1 = (-args.gA * pow(max(args.W, 0.001f), -3.f)) * diff1(hydro_args.P, k, d);
        valuef p2 = -w * h * diff1(args.gA, k, d);

        valuef p3;

        for(int j=0; j < 3; j++)
        {
            p3 += -Si[j] * diff1(args.gB[j], k, d);
        }

        valuef p4;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                valuef deriv = diff1(args.cY.invert()[i,j], k, d);

                /*if(i == 0 && j == 0 && k == 0)
                {
                    if_e((pos.x() == dim.x()/2 + 2) && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
                        print("Metd %f %.16f %.16f\n", deriv, Si[0], Si[0] * Si[0]);
                    });
                }*/

                valuef l1 = Si[i] / h;
                valuef l2 = divide_with_limit(Si[j], w, 0.f, 0.000001f);

                p4 += 0.5f * args.gA * args.W * args.W * l1 * l2 * deriv;

                //p4 += divide_with_limit(0.5f * args.gA * args.W * args.W * Si[i] * Si[j] * deriv, w * h, 0.f, 0.000001f);
            }
        }

        valuef w2_m_p2_calc = w2_m_p2(p_star, e_star, args.W, get_Gamma(), args.cY.invert(), Si, w);

        valuef p5 = divide_with_limit(args.gA * h * w2_m_p2_calc, w, 0.f, 0.000001f) * (diff1(args.W, k, d) / max(args.W, 0.001f));

        dSi_p1[k] += p1 + p2 + p3 + p4 + p5;

        /*if_e((pos.x() == dim.x()/2 + 2) && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
            print("p4 %.16f\n", p4);
        });*/

        /*if(k == 0)
        {
            if_e((pos.x() == dim.x()/2 + 19) && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
                print("Components %.10f %.10f %.10f %.10f %.10f\n", p1, p2, p3, p4, p5);
            });
        }*/
    }

    valuef fin_p_star = max(h_base.p_star[pos, dim] + timestep.get() * dp_star, 0.f);
    valuef fin_e_star = max(h_base.e_star[pos, dim] + timestep.get() * de_star, 0.f);

    valuef max_p = 1;

    fin_e_star = ternary(fin_p_star < (1e-6f * max_p), min(fin_e_star, 10 * fin_p_star), fin_e_star);

    mut_v3f fin_Si = declare_mut_e((v3f){});

    for(int i=0; i < 3; i++)
    {
        as_ref(fin_Si[i]) = h_base.Si[i][pos, dim] + timestep.get() * dSi_p1[i];
    }

    /*if_e(p_star >= min_p_star, [&]{
        v3f u_k = declare_e(fin_Si) / (h * p_star);

        u_k = clamp(u_k, -0.2f, 0.2f);

        as_ref(fin_Si) = u_k * h * p_star;
    });*/

    //fin_Si = clamp(fin_Si, -0.005f, 0.005f);

    if_e((pos.x() == dim.x()/2 + 2) && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
         ///so: the higher Si (and vi or ui), the more p_star is smanged away from the core
         ///why is Si increasing?
        //print("dVi %f %f %f\n", diff1(vi[0], 0, d), diff1(vi[1], 1, d), diff1(vi[2], 2, d));

        //print("vi %f %f %f dp_star %f p_star %f timestep %f\n", vi[0], vi[1], vi[2], dp_star, p_star, timestep.get());
         //print("p* %f\n", p_star);
         //print("w %f\n", w / p_star);
        //print("si %f %f %f\n", fin_Si[0] / (p_star * h), fin_Si[1] / (p_star * h), fin_Si[2] / (p_star * h));
    });

    ///something wonky is going on, this makes everything go brighter, weirdly. The utility buffer still has stuff in it
    ///but it shouldn't be used (?)
    ///for some reason, having this enabled makes things brighter
    ///tomorow: todo, check denegerate initial conditions
    if_e(fin_p_star <= min_p_star, [&]{
        as_ref(h_out.p_star[pos, dim]) = valuef(0);
        as_ref(h_out.e_star[pos, dim]) = valuef(0);

        for(int i=0; i < 3; i++)
        {
            as_ref(h_out.Si[i][pos, dim]) = valuef(0);
        }

        return_e();
    });

    //if_e(fin_p_star > min_p_star, [&]{
        /*as_ref(h_out.p_star[pos, dim]) = h_base.p_star[pos, dim];
        as_ref(h_out.e_star[pos, dim]) = h_base.e_star[pos, dim];

        for(int i=0; i < 3; i++)
        {
            as_ref(h_out.Si[i][pos, dim]) = h_base.Si[i][pos, dim];
        }*/

        as_ref(h_out.p_star[pos, dim]) = fin_p_star;
        as_ref(h_out.e_star[pos, dim]) = fin_e_star;

        for(int i=0; i < 3; i++)
        {
            as_ref(h_out.Si[i][pos, dim]) = as_constant(fin_Si[i]);
        }
    //});
}

hydrodynamic_plugin::hydrodynamic_plugin(cl::context ctx)
{
    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(init_hydro, "init_hydro");
    }, {"init_hydro"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(calculate_hydro_intermediates, "calculate_hydro_intermediates");
    }, {"calculate_hydro_intermediates"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(evolve_hydro, "evolve_hydro");
    }, {"evolve_hydro"});
}

buffer_provider* hydrodynamic_plugin::get_buffer_factory(cl::context ctx)
{
    return new hydrodynamic_buffers(ctx);
}

buffer_provider* hydrodynamic_plugin::get_utility_buffer_factory(cl::context ctx)
{
    return new hydrodynamic_utility_buffers(ctx);
}

void hydrodynamic_plugin::init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, cl::buffer u_buf, buffer_provider* to_init, buffer_provider* to_init_utility)
{
    neutron_star::all_numerical_eos_gpu neos(ctx);
    neos.init(cqueue, pack.stored_eos);

    hydrodynamic_buffers& bufs = *dynamic_cast<hydrodynamic_buffers*>(to_init);
    hydrodynamic_utility_buffers& ubufs = *dynamic_cast<hydrodynamic_utility_buffers*>(to_init_utility);

    {
        t3i dim = pack.dim;

        ///39
        cl::args args;
        in.append_to(args);
        args.push_back(bufs.p_star);
        args.push_back(bufs.e_star);
        args.push_back(bufs.Si[0]);
        args.push_back(bufs.Si[1]);
        args.push_back(bufs.Si[2]);
        args.push_back(ubufs.w);
        args.push_back(ubufs.P);
        args.push_back(ubufs.vi[0]);
        args.push_back(ubufs.vi[1]);
        args.push_back(ubufs.vi[2]);
        args.push_back(pack.dim);
        args.push_back(pack.scale);
        args.push_back(pack.disc.mu_cfl);
        args.push_back(pack.disc.mu_h_cfl);
        args.push_back(pack.disc.pressure_cfl);
        args.push_back(pack.disc.cfl);
        args.push_back(u_buf);
        args.push_back(pack.disc.Si_cfl[0]);
        args.push_back(pack.disc.Si_cfl[1]);
        args.push_back(pack.disc.Si_cfl[2]);
        args.push_back(pack.disc.star_indices);
        args.push_back(neos.pressures, neos.max_densities, neos.stride, neos.count);

        cqueue.exec("init_hydro", args, {dim.x(), dim.y(), dim.z()}, {8,8,1});
    }
}

void hydrodynamic_plugin::step(cl::context ctx, cl::command_queue cqueue, const plugin_step_data& sdata)
{
    /*for(auto& i : sdata.bssn_buffers)
        printf("Len %i\n", i.alloc_size);

    for(auto& i : sdata.buffers[sdata.base_idx])
        printf("Len %i\n", i.alloc_size);

    for(auto& i : sdata.buffers[sdata.in_idx])
        printf("Len %i\n", i.alloc_size);

    for(auto& i : sdata.buffers[sdata.out_idx])
        printf("Len %i\n", i.alloc_size);

    for(auto& i : sdata.utility_buffers)
        printf("Len %i\n", i.alloc_size);*/

    //return;

    //printf("Indices %i %i %i\n", sdata.base_idx, sdata.in_idx, sdata.out_idx);

    //printf("Mem ptr %p %p %p\n", sdata.buffers[sdata.base_idx][0].native_mem_object.data, sdata.buffers[sdata.in_idx][0].native_mem_object.data, sdata.buffers[sdata.out_idx][0].native_mem_object.data);

    {
        cl::args args;

        for(auto& i : sdata.bssn_buffers)
            args.push_back(i);

        for(auto& i : sdata.buffers[sdata.in_idx])
            args.push_back(i);

        for(auto& i : sdata.utility_buffers)
            args.push_back(i);

        args.push_back(sdata.dim);
        args.push_back(sdata.scale);
        args.push_back(sdata.evolve_points);
        args.push_back(sdata.evolve_length);

        cqueue.exec("calculate_hydro_intermediates", args, {sdata.evolve_length}, {128});
    }

    //assert(false);

    {
        //printf("Base in out %i %i %i\n", sdata.base_idx, sdata.in_idx, sdata.out_idx);

        cl::args args;

        for(auto& i : sdata.bssn_buffers)
            args.push_back(i);

        for(auto& i : sdata.buffers[sdata.base_idx])
            args.push_back(i);

        for(auto& i : sdata.buffers[sdata.in_idx])
            args.push_back(i);

        for(auto& i : sdata.buffers[sdata.out_idx])
            args.push_back(i);

        for(auto& i : sdata.utility_buffers)
            args.push_back(i);

        args.push_back(sdata.dim);
        args.push_back(sdata.scale);
        args.push_back(sdata.timestep);
        args.push_back(sdata.evolve_points);
        args.push_back(sdata.evolve_length);

        cqueue.exec("evolve_hydro", args, {sdata.evolve_length}, {128});
    }

    /*{
        cl::args args;

        for(auto& i : sdata.bssn_buffers)
            args.push_back(i);

        for(auto& i : sdata.buffers[sdata.out_idx])
            args.push_back(i);

        for(auto& i : sdata.utility_buffers)
            args.push_back(i);

        args.push_back(sdata.dim);
        args.push_back(sdata.scale);
        args.push_back(sdata.evolve_points);
        args.push_back(sdata.evolve_length);

        cqueue.exec("calculate_hydro_intermediates", args, {sdata.evolve_length}, {128});
    }*/
}

void hydrodynamic_plugin::add_args_provider(all_adm_args_mem& mem)
{
    mem.add(full_hydrodynamic_args<buffer<valuef>>());
}

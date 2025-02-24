#include "hydrodynamics.hpp"
#include "init_general.hpp"

///so like. What if I did the projective real strategy?

//stable with 1e-6, but the neutron star dissipates
constexpr float min_p_star = 1e-7f;

template<typename T>
inline
T safe_divide(const T& top, const T& bottom, float tol = 1e-7)
{
    return top / max(bottom, T{tol});
}

valuef get_Gamma()
{
    return 2;
}

valuef calculate_h_from_epsilon(valuef epsilon)
{
    return 1 + get_Gamma() * epsilon;
}

valuef calculate_epsilon(valuef p_star, valuef e_star, valuef W, valuef w)
{
    valuef e_m6phi = W*W*W;
    valuef Gamma = get_Gamma();

    return pow(safe_divide(e_m6phi, w), Gamma - 1) * pow(e_star, Gamma) * pow(p_star, Gamma - 2);
}

valuef calculate_p0e(valuef W, valuef w, valuef p_star, valuef e_star)
{
    valuef e_m6phi = W*W*W;
    valuef Gamma = get_Gamma();
    valuef iv_au0 = safe_divide(p_star, w);

    return pow(max(e_star * e_m6phi * iv_au0, 0.f), Gamma);
}

valuef eos(valuef W, valuef w, valuef p_star, valuef e_star)
{
    valuef Gamma = get_Gamma();
    return calculate_p0e(W, w, p_star, e_star) * (Gamma - 1);
}

//todo: I may need to set vi to 0 here manually
//or, I may need to remove the leibnitz that I'm doing
///todo: try setting this to zero where appropriate
v3f calculate_vi(valuef gA, v3f gB, valuef W, valuef w, valuef epsilon, v3f Si, const unit_metric<valuef, 3, 3>& cY, valuef p_star)
{
    valuef h = calculate_h_from_epsilon(epsilon);

    //note to self, actually hand derived this and am sure its correct
    //tol is very intentionally set to 1e-6, breaks if lower than this
    v3f real_value = -gB + safe_divide(W*W * gA, w*h, 1e-6) * cY.invert().raise(Si);

    return ternary(p_star <= min_p_star, (v3f){}, real_value);
}

valuef calculate_Pvis(valuef W, v3f vi, valuef p_star, valuef e_star, valuef w, const derivative_data& d, valuef total_elapsed, valuef linear_damping_timescale)
{
    valuef e_m6phi = pow(W, 3.f);

    valuef dkvk = 0;

    for(int k=0; k < 3; k++)
    {
        dkvk += 2 * diff1(vi[k], k, d);
    }

    valuef littledv = dkvk * d.scale;
    valuef Gamma = get_Gamma();

    valuef A = safe_divide(pow(e_star, Gamma) * pow(p_star, Gamma - 1) * pow(e_m6phi, Gamma - 1), pow(w, Gamma - 1), 1e-6f);

    //ctx.add("DBG_A", A);

    ///[0.1, 1.0}
    valuef CQvis = 1.f;

    ///it looks like the littledv is to only turn on viscosity when the flow is compressive
    valuef PQvis = ternary(littledv < 0, CQvis * A * pow(littledv, 2), valuef{0.f});

    return PQvis;

    valuef linear_damping = exp(-(total_elapsed * total_elapsed) / (2 * linear_damping_timescale * linear_damping_timescale));

    ///paper i'm looking at only turns on viscosity inside a star, ie p > pcrit. We could calculate a crit value
    ///or, we could simply make this time variable, though that's kind of annoying
    valuef CLvis = 1.f * linear_damping;
    valuef n = 1;

    valuef PLvis = ternary(littledv < 0, -CLvis * sqrt((get_Gamma()/n) * p_star * A) * littledv, valuef(0.f));

    return PQvis + PLvis;
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

    hydrodynamic_concrete(v3i pos, v3i dim, hydrodynamic_base_args<buffer<valuef>> bargs, hydrodynamic_utility_args<buffer<valuef>> uargs)
    {
        p_star = max(bargs.p_star[pos, dim], 0.f);
        e_star = max(bargs.e_star[pos, dim], 0.f);
        Si = {bargs.Si[0][pos, dim], bargs.Si[1][pos, dim], bargs.Si[2][pos, dim]};
        w = uargs.w[pos, dim];
        P = uargs.P[pos, dim];
    }

    valuef calculate_h_with_eos(valuef W)
    {
        valuef epsilon = calculate_epsilon(W);

        return ::calculate_h_from_epsilon(epsilon);
    }

    valuef calculate_epsilon(valuef W)
    {
        return ::calculate_epsilon(p_star, e_star, W, w);
    }

    valuef calculate_p0e(valuef W)
    {
        return ::calculate_p0e(W, w, p_star, e_star);
    }

    valuef eos(valuef W)
    {
        return ::eos(W, w, p_star, e_star);
    }

    v3f calculate_vi(valuef gA, v3f gB, valuef W, const unit_metric<valuef, 3, 3>& cY)
    {
        valuef epsilon = calculate_epsilon(W);

        return ::calculate_vi(gA, gB, W, w, epsilon, Si, cY, p_star);
    }

    ///rhs here to specifically indicate that we're returning -(di Vec v^i), ie the negative
    valuef advect_rhs(valuef in, v3f vi, const derivative_data& d)
    {
        auto leib = [&](valuef v1, valuef v2, int i)
        {
            return diff1(v1 * v2, i, d);
            //return diff1(v1, i, d) * v2 + diff1(v2, i, d) * v1;
        };

        valuef sum = 0;

        for(int i=0; i < 3; i++)
        {
            sum += leib(in, vi[i], i);
        }

        return -sum;
    }

    v3f advect_rhs(v3f in, v3f vi, const derivative_data& d)
    {
        v3f ret;

        for(int i=0; i < 3;  i++)
            ret[i] = advect_rhs(in[i], vi, d);

        return ret;
    }

    valuef calculate_Pvis(valuef W, v3f vi, const derivative_data& d, valuef total_elapsed, valuef damping_timescale)
    {
        return ::calculate_Pvis(W, vi, p_star, e_star, w, d, total_elapsed, damping_timescale);
    }

    valuef e_star_rhs(valuef gA, v3f gB, unit_metric<valuef, 3, 3> cY, valuef W, v3f vi, const derivative_data& d, valuef total_elapsed, valuef damping_timescale)
    {
        auto icY = cY.invert();

        valuef e_6phi = pow(max(W, 0.1f), -3.f);

        valuef Pvis = calculate_Pvis(W, vi, d, total_elapsed, damping_timescale);

        valuef sum_interior_rhs = 0;

        for(int k=0; k < 3; k++)
        {
            value to_diff = safe_divide(w * vi[k] * e_6phi, p_star, 1e-6);

            sum_interior_rhs += diff1(to_diff, k, d);
        }

        valuef Gamma = get_Gamma();

        valuef p0e = calculate_p0e(W);

        valuef degenerate = safe_divide(valuef{1}, pow(p0e, 1 - 1/Gamma), 1e-6);

        return -degenerate * (Pvis / Gamma) * sum_interior_rhs;
    }
};

///it might be because i'm using hydro_args.eos instead of P
template<typename T>
valuef full_hydrodynamic_args<T>::adm_p(bssn_args& args, const derivative_data& d)
{
    hydrodynamic_concrete hydro_args(d.pos, d.dim, *this);

    valuef h = hydro_args.calculate_h_with_eos(args.W);

    return ternary(hydro_args.p_star <= 0,
                   valuef(),
                   hydro_args.w * h * pow(args.W, 3.f) - hydro_args.eos(args.W));
}

template<typename T>
tensor<valuef, 3> full_hydrodynamic_args<T>::adm_Si(bssn_args& args, const derivative_data& d)
{
    v3f cSi = {this->Si[0][d.pos, d.dim], this->Si[1][d.pos, d.dim], this->Si[2][d.pos, d.dim]};

    valuef p_star = this->p_star[d.pos, d.dim];

    return ternary(p_star <= 0,
                   v3f(),
                   pow(args.W, 3.f) * cSi);
}

template<typename T>
tensor<valuef, 3, 3> full_hydrodynamic_args<T>::adm_W2_Sij(bssn_args& args, const derivative_data& d)
{
    hydrodynamic_concrete hydro_args(d.pos, d.dim, *this);

    valuef h = hydro_args.calculate_h_with_eos(args.W);

    tensor<valuef, 3, 3> W2_Sij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            W2_Sij[i, j] = safe_divide(pow(args.W, 5.f) * hydro_args.Si[i] * hydro_args.Si[j], hydro_args.w * h);
        }
    }

    return ternary(hydro_args.p_star <= 0,
                   tensor<valuef, 3, 3>(),
                   W2_Sij + hydro_args.eos(args.W) * args.cY.to_tensor());
}

template<typename T>
valuef full_hydrodynamic_args<T>::dbg(bssn_args& args, const derivative_data& d)
{
    return fabs(this->p_star[d.pos, d.dim]) * 500;
    //return sqrt(pow(Si[0][d.pos, d.dim], 2.f) + pow(Si[1][d.pos, d.dim], 2.f)) * 10;
    //return sqrt(pow(Si[0][d.pos, d.dim], 2.f) + pow(Si[2][d.pos, d.dim], 2.f)) * 100;
    //return e_star[d.pos, d.dim] * 0.5;
    //return fabs(this->Si[0][d.pos, d.dim]) * 100 * 100;
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
    s0.dissipation_coeff = 0.2;
    s0.dissipation_order = 4;

    buffer_descriptor s1;
    s1.name = "cs1";
    s1.dissipation_coeff = 0.2;
    s1.dissipation_order = 4;

    buffer_descriptor s2;
    s2.name = "cs2";
    s2.dissipation_coeff = 0.2;
    s2.dissipation_order = 4;

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

    return {w, P};
}

std::vector<cl::buffer> hydrodynamic_utility_buffers::get_buffers()
{
    return {w, P};
}

void hydrodynamic_utility_buffers::allocate(cl::context ctx, cl::command_queue cqueue, t3i size)
{
    int64_t cells = int64_t{size.x()} * size.y() * size.z();

    P.alloc(sizeof(cl_float) * cells);
    w.alloc(sizeof(cl_float) * cells);

    P.set_to_zero(cqueue);
    w.set_to_zero(cqueue);

    for(int i=0; i < 4; i++)
    {
        intermediate.emplace_back(ctx);
        intermediate.back().alloc(sizeof(cl_float) * cells);
        intermediate.back().set_to_zero(cqueue);
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

valuef calculate_w(valuef p_star, valuef e_star, valuef W, inverse_metric<valuef, 3, 3> icY, v3f Si);

void init_hydro(execution_context& ectx, bssn_args_mem<buffer<valuef>> in, full_hydrodynamic_args<buffer_mut<valuef>> hydro, literal<v3i> ldim, literal<valuef> scale,
                buffer<valuef> mu_h_cfl_b, buffer<valuef> cfl_b, buffer<valuef> u_correction_b, std::array<buffer<valuef>, 3> Si_cfl_b,
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

        if_e(i == eos_data.pressure_stride.get(), [&]{
            print("Error, overflowed pressure data\n");
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

    valuef mu_h_cfl = mu_h_cfl_b[pos, dim];
    valuef phi = cfl_b[pos, dim] + u_correction_b[pos, dim];
    v3f Si_cfl = {Si_cfl_b[0][pos, dim], Si_cfl_b[1][pos, dim], Si_cfl_b[2][pos, dim]};

    valuef mu_h = mu_h_cfl * pow(phi, -8);
    v3f Si = Si_cfl * pow(phi, -10);

    valuef Gamma = get_Gamma();

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

        u0 = next_W;
        //we have u0, now lets solve for mu
        //mu_h = (mu + f(mu)) * W^2 - f(mu)
        mu = get_mu_for(mu_h, u0);

        pin(u0);
        pin(mu);
    }

    valuef p0 = mu_to_p0(mu);

    valuef pressure = p0_to_pressure(p0);

    valuef p0_e = pressure / (Gamma - 1);

    value gA = args.gA;

    //fluid dynamics cannot have a singular initial slice, so setting the clamping pretty high here because its irrelevant
    //thing is we have 0 quantities at the singularity, so as long as you don't generate a literal NaN here, you're 100% fine

    valuef p_star = p0 * gA * u0 * pow(cW, -3);
    valuef e_star = pow(p0_e, (1/Gamma)) * gA * u0 * pow(cW, -3);

    ///Si isn't well defined when gA != 1 in our init conditions
    ///oh! Si isn't a hydrodynamic field!

    valuef w = p_star * gA * u0;

    v3f ui = Si / ((mu + pressure) * u0);

    v3f u_i;

    for(int s=0; s < 3; s++)
    {
        valuef sum = 0;

        for(int k=0; k < 3; k++)
        {
            sum += Yij[s, k] * ui[k];
        }

        u_i[s] = args.gB[s] * u0 + sum;
    }

    valuef h = calculate_h_from_epsilon(p0_e / p0);

    v3f Si_lo_cfl = p_star * h * u_i;

    ///next, I need to lower ui with the 4-metric, only spatially though!

    //v3f Si_lo_cfl = pow(cW, -3) * Yij.lower(Si);

    as_ref(hydro.p_star[pos, dim]) = p_star;
    as_ref(hydro.e_star[pos, dim]) = e_star;
    as_ref(hydro.Si[0][pos, dim]) = Si_lo_cfl[0];
    as_ref(hydro.Si[1][pos, dim]) = Si_lo_cfl[1];
    as_ref(hydro.Si[2][pos, dim]) = Si_lo_cfl[2];

    //w also isn't well defined when gA != 1
    //valuef real_w = p_star * gA * u0;

    ///ok so. I'm trying to answer the quesiton of why w = p* gA u0
    ///is not the same answer as calculate_w
    valuef calc_w = calculate_w(p_star, e_star, args.W, args.cY.invert(), Si_lo_cfl);

    as_ref(hydro.w[pos, dim]) = w;
    as_ref(hydro.P[pos, dim]) = eos(args.W, w, p_star, e_star);
}

valuef w_next_interior(valuef p_star, valuef e_star, valuef W, valuef w_prev)
{
    valuef Gamma = get_Gamma();

    valuef A = pow(max(W, 0.001f), 3.f * Gamma - 3.f);
    valuef wG = pow(w_prev, Gamma - 1);

    return safe_divide(wG, wG + A * Gamma * pow(e_star, Gamma) * pow(max(p_star, 1e-7f), Gamma - 2));
}

valuef calculate_w_constant(valuef W, const inverse_metric<valuef, 3, 3>& icY, v3f Si)
{
    valuef cst = 0;

    for(int i=0; i < 3; i++)
    {
        valuef sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum += icY[i, j] * Si[j];
        }

        cst += Si[i] * sum;
    }

    return W*W * cst;
}

valuef calculate_w(valuef p_star, valuef e_star, valuef W, inverse_metric<valuef, 3, 3> icY, v3f Si)
{
    using namespace single_source;

    valuef w = 0.5f;

    valuef p_sq = p_star * p_star;

    valuef cst = calculate_w_constant(W, icY, Si);

    //pin(p_sq);
    pin(cst);

    for(int i=0; i < 140; i++)
    {
        valuef D = w_next_interior(p_star, e_star, W, w);

        valuef w_next = sqrt(max(p_sq + cst * D*D, 0.f));

        pin(w_next);

        w = w_next;
    }

    return w;
}

valuef w2_m_p2(valuef p_star, valuef e_star, valuef W, inverse_metric<valuef, 3, 3> icY, v3f Si, valuef w)
{
    valuef p_sq = p_star * p_star;

    valuef cst = calculate_w_constant(W, icY, Si);

    valuef D = w_next_interior(p_star, e_star, W, w);

    return max(cst * D*D, 0.f);

}
void calculate_w_kern(execution_context& ectx, bssn_args_mem<buffer<valuef>> in, hydrodynamic_base_args<buffer<valuef>> hydro, buffer_mut<valuef> w_out,
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

    valuef p_star = hydro.p_star[pos, dim];
    valuef e_star = hydro.e_star[pos, dim];
    v3f Si = {hydro.Si[0][pos, dim], hydro.Si[1][pos, dim], hydro.Si[2][pos, dim]};

    if_e(p_star <= min_p_star, [&]{
        as_ref(w_out[pos, dim]) = valuef(0);

        return_e();
    });

    bssn_args args(pos, dim, in);

    valuef w = calculate_w(p_star, e_star, args.W, args.cY.invert(), Si);
    w = max(w, p_star * args.gA * 1);

    as_ref(w_out[pos, dim]) = w;
}

#define MIN_LAPSE 0.45f
#define MIN_VISCOSITY_LAPSE 0.4f

void calculate_p_kern(execution_context& ectx, bssn_args_mem<buffer<valuef>> in, hydrodynamic_base_args<buffer<valuef>> hydro, buffer<valuef> w_in, buffer_mut<valuef> P_out,
                      literal<v3i> idim, literal<valuef> scale, literal<valuef> total_elapsed, literal<valuef> damping_timescale,
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

    valuef p_star = hydro.p_star[pos, dim];
    valuef e_star = hydro.e_star[pos, dim];
    v3f Si = {hydro.Si[0][pos, dim], hydro.Si[1][pos, dim], hydro.Si[2][pos, dim]};

    if_e(p_star <= min_p_star, [&]{
        as_ref(P_out[pos, dim]) = valuef(0);

        return_e();
    });

    derivative_data d;
    d.pos = pos;
    d.dim = dim;
    d.scale = scale.get();

    bssn_args args(pos, dim, in);

    valuef w = w_in[pos, dim];

    //this might be a source of instability when p* is low, because of derivatives
    mut<valuef> P = declare_mut_e(max(eos(args.W, w, p_star, e_star), 0.f));

    t3i dbg = {77, 76, 86};

    /*if_e(pos.x() == dbg.x() && pos.y() == dbg.y() && pos.z() == dbg.z(), [&]{
        print("Pressure p1 %f\n", as_constant(P));
    });*/

    if_e(args.gA >= MIN_VISCOSITY_LAPSE, [&]{
        valuef epsilon = calculate_epsilon(p_star, e_star, args.W, w);
        v3f vi = calculate_vi(args.gA, args.gB, args.W, w, epsilon, Si, args.cY, p_star);

        ///derivative in the x direction is broken, time to chase down
        /*if_e(pos.x() == dbg.x() && pos.y() == dbg.y() && pos.z() == dbg.z(), [&]{
            auto vars = get_differentiation_variables<5>(vi.x(), 1);

            print("Cpt %f %f %f %f\n", vars[0], vars[1], vars[3], vars[4]);

            //valuef deriv = (-vars[4] + vars[0]) + 8 * (vars[3] - vars[1]);

            //print("Summmed %f full deriv %f\n", deriv, deriv / (12 * scale.get()));

            print("Dbgvi %f %f %f diff %f %f %f\n", vi.x(), vi.y(), vi.z(), diff1(vi.x(), 0, d), diff1(vi.y(), 1, d), diff1(vi.z(), 2, d));
        });*/

        as_ref(P) += calculate_Pvis(args.W, vi, p_star, e_star, w, d, total_elapsed.get(), damping_timescale.get());
    });

    if_e(pos.x() == dbg.x() && pos.y() == dbg.y() && pos.z() == dbg.z(), [&]{
        print("Pressure p2 %f\n", as_constant(P));
    });

    as_ref(P_out[pos, dim]) = as_constant(P);
}

void advect_all(execution_context& ectx, bssn_args_mem<buffer<valuef>> in,
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

    derivative_data d;
    d.pos = pos;
    d.dim = idim.get();
    d.scale = scale.get();

    bssn_args args(pos, dim, in);
    hydrodynamic_concrete hydro_args(pos, dim, h_in, util);

    if_e(hydro_args.p_star <= min_p_star, [&]{
        as_ref(h_out.p_star[pos, dim]) = h_in.p_star[pos, dim];
        as_ref(h_out.e_star[pos, dim]) = h_in.e_star[pos, dim];
        as_ref(h_out.Si[0][pos, dim]) = h_in.Si[0][pos, dim];
        as_ref(h_out.Si[1][pos, dim]) = h_in.Si[1][pos, dim];
        as_ref(h_out.Si[2][pos, dim]) = h_in.Si[2][pos, dim];
        return_e();
    });

    if_e(args.gA < MIN_LAPSE, [&]{
        valuef damp = 0.1f;

        valuef dt_p_star = damp * (0 - h_in.p_star[pos, dim]);
        valuef dt_e_star = damp * (0 - h_in.e_star[pos, dim]);

        valuef dt_s0 = damp * (0 - h_in.Si[0][pos, dim]);
        valuef dt_s1 = damp * (0 - h_in.Si[1][pos, dim]);
        valuef dt_s2 = damp * (0 - h_in.Si[2][pos, dim]);

        valuef fin_p_star = h_base.p_star[pos, dim] + dt_p_star * timestep.get();
        valuef fin_e_star = h_base.e_star[pos, dim] + dt_e_star * timestep.get();

        valuef fin_s0 = h_base.Si[0][pos, dim] + dt_s0 * timestep.get();
        valuef fin_s1 = h_base.Si[1][pos, dim] + dt_s1 * timestep.get();
        valuef fin_s2 = h_base.Si[2][pos, dim] + dt_s2 * timestep.get();

        as_ref(h_out.p_star[pos, dim]) = max(fin_p_star, 0.f);
        as_ref(h_out.e_star[pos, dim]) = max(fin_e_star, 0.f);
        as_ref(h_out.Si[0][pos, dim]) = fin_s0;
        as_ref(h_out.Si[1][pos, dim]) = fin_s1;
        as_ref(h_out.Si[2][pos, dim]) = fin_s2;
        return_e();
    });

    v3f vi = hydro_args.calculate_vi(args.gA, args.gB, args.W, args.cY);

    valuef dp_star = hydro_args.advect_rhs(hydro_args.p_star, vi, d);
    valuef de_star = hydro_args.advect_rhs(hydro_args.e_star, vi, d);
    v3f dSi = hydro_args.advect_rhs(hydro_args.Si, vi, d);

    valuef fin_e_star = h_base.e_star[pos, dim] + de_star * timestep.get();
    //fin_e_star = ternary(hydro_args.p_star < valuef(1e-6f), min(fin_e_star, 10 * hydro_args.p_star), fin_e_star);

    as_ref(h_out.p_star[pos, dim]) = max(h_base.p_star[pos, dim] + dp_star * timestep.get(), 0.f);
    as_ref(h_out.e_star[pos, dim]) = max(fin_e_star, 0.f);

    as_ref(h_out.Si[0][pos, dim]) = h_base.Si[0][pos, dim] + dSi[0] * timestep.get();
    as_ref(h_out.Si[1][pos, dim]) = h_base.Si[1][pos, dim] + dSi[1] * timestep.get();
    as_ref(h_out.Si[2][pos, dim]) = h_base.Si[2][pos, dim] + dSi[2] * timestep.get();
}

void evolve_si_p2(execution_context& ectx, bssn_args_mem<buffer<valuef>> in,
                  hydrodynamic_base_args<buffer<valuef>> h_base, hydrodynamic_base_args<buffer<valuef>> h_in, std::array<buffer_mut<valuef>, 3> Si_out, buffer_mut<valuef> e_star_out,
                  hydrodynamic_utility_args<buffer<valuef>> util,
                  literal<v3i> idim, literal<valuef> scale, literal<valuef> timestep, literal<valuef> total_elapsed, literal<valuef> damping_timescale,
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

    derivative_data d;
    d.pos = pos;
    d.dim = idim.get();
    d.scale = scale.get();

    bssn_args args(pos, dim, in);
    hydrodynamic_concrete hydro_args(pos, dim, h_in, util);

    if_e(hydro_args.p_star <= min_p_star, [&]{
        as_ref(e_star_out[pos, dim]) = h_in.e_star[pos, dim];
        as_ref(Si_out[0][pos, dim]) = h_in.Si[0][pos, dim];
        as_ref(Si_out[1][pos, dim]) = h_in.Si[1][pos, dim];
        as_ref(Si_out[2][pos, dim]) = h_in.Si[2][pos, dim];
        return_e();
    });

    valuef p_star = hydro_args.p_star;
    valuef e_star = hydro_args.e_star;
    v3f Si = hydro_args.Si;

    valuef h = hydro_args.calculate_h_with_eos(args.W);
    valuef w = hydro_args.w;

    mut<valuef> de_star = declare_mut_e(valuef(0.f));

    if_e(args.gA >= MIN_VISCOSITY_LAPSE, [&]{
        valuef epsilon = hydro_args.calculate_epsilon(args.W);
        v3f vi = calculate_vi(args.gA, args.gB, args.W, w, epsilon, hydro_args.Si, args.cY, p_star);
        as_ref(de_star) += hydro_args.e_star_rhs(args.gA, args.gB, args.cY, args.W, vi, d, total_elapsed.get(), damping_timescale.get());
    });

    v3f dSi_p1;

    for(int k=0; k < 3; k++)
    {
        ///so, it seems like this term is a little unstable
        ///Si advect is bad
        valuef p1 = (-args.gA * pow(max(args.W, 0.1f), -3.f)) * diff1(hydro_args.P, k, d);
        valuef p2 = -w * h * diff1(args.gA, k, d);

        valuef p3;

        for(int j=0; j < 3; j++)
        {
            p3 += -Si[j] * diff1(args.gB[j], k, d) ;
        }

        valuef p4;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                valuef deriv = diff1(args.cY.invert()[i,j], k, d);

                valuef l1 = Si[i] / h;
                valuef l2 = safe_divide(Si[j], w);

                p4 += 0.5f * args.gA * args.W * args.W * l1 * l2 * deriv;
            }
        }

        valuef w2_m_p2_calc = w2_m_p2(p_star, e_star, args.W, args.cY.invert(), Si, w);

        valuef p5 = safe_divide(args.gA * h * w2_m_p2_calc, w) * (diff1(args.W, k, d) / max(args.W, 0.1f));

        dSi_p1[k] += (p1 + p2 + p3 + p4 + p5);
    }

    mut_v3f fin_Si = declare_mut_e((v3f){});

    for(int i=0; i < 3; i++)
    {
        as_ref(fin_Si[i]) = h_in.Si[i][pos, dim] + timestep.get() * dSi_p1[i];
    }

    //#define CLAMP_HIGH_VELOCITY
    #ifdef CLAMP_HIGH_VELOCITY
    //if_e(p_star >= min_p_star && p_star < 1e-5, [&]{
    if_e(p_star >= min_p_star && p_star < min_p_star * 10, [&]{
        v3f dfsi = declare_e(fin_Si);

        v3f u_k;

        for(int i=0; i < 3; i++)
            u_k[i] = safe_divide(dfsi[i], h * p_star, 1e-6);

        u_k = clamp(u_k, -0.2f, 0.2f);

        as_ref(fin_Si) = u_k * h * p_star;
    });
    #endif

    for(int i=0; i < 3; i++)
    {
        as_ref(Si_out[i][pos, dim]) = as_constant(fin_Si[i]);
    }

    //as_ref(e_star_out[pos, dim]) = h_in.e_star[pos, dim];


    valuef fin_e_star = h_in.e_star[pos, dim] + as_constant(de_star) * timestep.get();
    //fin_e_star = ternary(hydro_args.p_star < valuef(1e-6f), min(fin_e_star, 10 * hydro_args.p_star), fin_e_star);

    as_ref(e_star_out[pos, dim]) = max(fin_e_star, 0.f);
}



void evolve_hydro_all(execution_context& ectx, bssn_args_mem<buffer<valuef>> in,
                  hydrodynamic_base_args<buffer<valuef>> h_base, hydrodynamic_base_args<buffer<valuef>> h_in, hydrodynamic_base_args<buffer_mut<valuef>> h_out,
                  hydrodynamic_utility_args<buffer<valuef>> util,
                  literal<v3i> idim, literal<valuef> scale, literal<valuef> timestep, literal<valuef> total_elapsed, literal<valuef> damping_timescale,
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

    derivative_data d;
    d.pos = pos;
    d.dim = idim.get();
    d.scale = scale.get();

    bssn_args args(pos, dim, in);
    hydrodynamic_concrete hydro_args(pos, dim, h_in, util);

    if_e(hydro_args.p_star <= min_p_star, [&]{
        as_ref(h_out.p_star[pos, dim]) = h_in.p_star[pos, dim];
        as_ref(h_out.e_star[pos, dim]) = h_in.e_star[pos, dim];
        as_ref(h_out.Si[0][pos, dim]) = h_in.Si[0][pos, dim];
        as_ref(h_out.Si[1][pos, dim]) = h_in.Si[1][pos, dim];
        as_ref(h_out.Si[2][pos, dim]) = h_in.Si[2][pos, dim];
        return_e();
    });

    if_e(args.gA < MIN_LAPSE, [&]{
        valuef damp = 0.1f;

        valuef dt_p_star = damp * (0 - h_in.p_star[pos, dim]);
        valuef dt_e_star = damp * (0 - h_in.e_star[pos, dim]);

        valuef dt_s0 = damp * (0 - h_in.Si[0][pos, dim]);
        valuef dt_s1 = damp * (0 - h_in.Si[1][pos, dim]);
        valuef dt_s2 = damp * (0 - h_in.Si[2][pos, dim]);

        valuef fin_p_star = h_base.p_star[pos, dim] + dt_p_star * timestep.get();
        valuef fin_e_star = h_base.e_star[pos, dim] + dt_e_star * timestep.get();

        valuef fin_s0 = h_base.Si[0][pos, dim] + dt_s0 * timestep.get();
        valuef fin_s1 = h_base.Si[1][pos, dim] + dt_s1 * timestep.get();
        valuef fin_s2 = h_base.Si[2][pos, dim] + dt_s2 * timestep.get();

        as_ref(h_out.p_star[pos, dim]) = max(fin_p_star, 0.f);
        as_ref(h_out.e_star[pos, dim]) = max(fin_e_star, 0.f);
        as_ref(h_out.Si[0][pos, dim]) = fin_s0;
        as_ref(h_out.Si[1][pos, dim]) = fin_s1;
        as_ref(h_out.Si[2][pos, dim]) = fin_s2;
        return_e();
    });

    v3f vi = hydro_args.calculate_vi(args.gA, args.gB, args.W, args.cY);

    valuef dp_star = hydro_args.advect_rhs(hydro_args.p_star, vi, d);
    mut<valuef> de_star = declare_mut_e(hydro_args.advect_rhs(hydro_args.e_star, vi, d));
    v3f dSi = hydro_args.advect_rhs(hydro_args.Si, vi, d);

    valuef e_advect_only = declare_e(de_star);
    valuef si_advect = dSi[0];

    if_e(args.gA >= MIN_VISCOSITY_LAPSE, [&]{
        as_ref(de_star) += hydro_args.e_star_rhs(args.gA, args.gB, args.cY, args.W, vi, d, total_elapsed.get(), damping_timescale.get());
    });

    valuef w = hydro_args.w;
    valuef h = hydro_args.calculate_h_with_eos(args.W);

    v3f dSi_p1;

    /*Diff infinity P 0.000016 m2 0.000035 m1 0.000045 p1 infinity p2 0.000000
    Components -infinity -0.000051 -0.000004 -0.000005 0.000013
    met? 0.998912 -0.413515 -0.111903 0.863452 -0.048539 1.471631
    imet? 1.268327 0.613974 0.116695 1.457507 0.094760 0.691517
    76 74 83     p* 0.0010295503 e* 0.0107779577 si 0.000432 0.000503 -0.000731 w 0.001272 P 0.000016 vi 0.204688 0.219966 -0.112130 epsilon 0.0422649533 raised 0.000928 top_1 0.000257 uk 0.386495 0.450185 -0.654695 h 1.084530
    76 74 83 out p* 0.0009650248 e* 0.0100246817 si 0.000640 0.000489 -infinity*/

    t3i dbg = {77, 76, 86};

    ///we could use the advanced Si here
    for(int k=0; k < 3; k++)
    {
        if_e(pos.x() == dbg.x() && pos.y() == dbg.y() && pos.z() == dbg.z() && k == 0, [&]{
            print("Diff %f P %f m2 %f m1 %f p1 %f p2 %f\n", diff1(hydro_args.P, k, d), hydro_args.P, util.P[pos + (v3i){0, 0, -2}, dim], util.P[pos + (v3i){0, 0, -1}, dim], util.P[pos + (v3i){0, 0, 1}, dim], util.P[pos + (v3i){0, 0, 2}, dim]);
        });

        ///so, it seems like this term is a little unstable
        ///Si advect is bad
        valuef p1 = (-args.gA * pow(max(args.W, 0.1f), -3.f)) * diff1(hydro_args.P, k, d);
        valuef p2 = -w * h * diff1(args.gA, k, d);

        valuef p3;

        for(int j=0; j < 3; j++)
        {
            p3 += -hydro_args.Si[j] * diff1(args.gB[j], k, d) ;
        }

        valuef p4;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                valuef deriv = diff1(args.cY.invert()[i,j], k, d);

                valuef l1 = hydro_args.Si[i] / h;
                valuef l2 = safe_divide(hydro_args.Si[j], w);

                p4 += 0.5f * args.gA * args.W * args.W * l1 * l2 * deriv;
            }
        }

        valuef w2_m_p2_calc = w2_m_p2(hydro_args.p_star, hydro_args.e_star, args.W, args.cY.invert(), hydro_args.Si, w);

        valuef p5 = safe_divide(args.gA * h * w2_m_p2_calc, w) * (diff1(args.W, k, d) / max(args.W, 0.1f));

        dSi_p1[k] += (p1 + p2 + p3 + p4 + p5);

        if_e(pos.x() == dbg.x() && pos.y() == dbg.y() && pos.z() == dbg.z() && k == 0, [&]{
            print("Components %f %f %f %f %f\n", p1, p2, p3, p4, p5);
        });
    }

    dSi += dSi_p1;

    v3f base_Si = {h_base.Si[0][pos, dim], h_base.Si[1][pos, dim], h_base.Si[2][pos, dim]};

    mut_v3f fin_Si = declare_mut_e(base_Si + timestep.get() * dSi);

    //#define CLAMP_HIGH_VELOCITY
    #ifdef CLAMP_HIGH_VELOCITY
    if_e(hydro_args.p_star >= min_p_star && hydro_args.p_star < min_p_star * 10, [&]{
        v3f dfsi = declare_e(fin_Si);

        v3f u_k;

        for(int i=0; i < 3; i++)
            u_k[i] = safe_divide(dfsi[i], h * hydro_args.p_star, 1e-6);

        u_k = clamp(u_k, -0.2f, 0.2f);

        as_ref(fin_Si) = u_k * h * hydro_args.p_star;
    });
    #endif

    /*if_e(!isfinite(dp_star) || !isfinite(de_star) || !isfinite(dSi[0]) || !isfinite(dSi[1]) || !isfinite(dSi[2]) || !isfinite(hydro_args.p_star) || !isfinite(hydro_args.e_star) || !isfinite(hydro_args.Si[0]) || !isfinite(hydro_args.Si[1]) || !isfinite(hydro_args.Si[2])
         || !isfinite(args.cY[0, 0]) || !isfinite(args.cY[1, 0]) || !isfinite(args.cY[2, 0]) || !isfinite(args.cY[1, 1]) || !isfinite(args.cY[2, 1]) || !isfinite(args.cY[2, 2]), [&]{
        print("Pos p1 %i %i %i\n", pos.x(), pos.y(), pos.z());
    });*/

    ///Pos p1 58 73 72
    ///Pos p1 87 80 69
    ///Pos p1 76 74 83

    #if 1
    if_e(pos.x() == dbg.x() && pos.y() == dbg.y() && pos.z() == dbg.z(), [&]{
    //if_e(pos.x() == 71 && pos.y() == 63 && pos.z() == 53, [&]{
    //if_e(pos.x() == 59 && pos.y() == 81 && pos.z() == 57, [&]{
        valuef epsilon = hydro_args.calculate_epsilon(args.W);

        unit_metric<valuef, 3, 3> cY = args.cY;
        pin(cY);

        auto inverted = cY.invert();

        valuef raised = args.cY.invert().raise(hydro_args.Si)[1];

        v3f dfsi = declare_e(hydro_args.Si);

        v3f u_k;

        for(int i=0; i < 3; i++)
            u_k[i] = safe_divide(dfsi[i], h * hydro_args.p_star);

        valuef t1 = args.W * args.W * args.gA * raised;

        print("met? %f %f %f %f %f %f\n", args.cY[0, 0], args.cY[1, 0], args.cY[2, 0], args.cY[1, 1], args.cY[2, 1], args.cY[2, 2]);
        print("imet? %f %f %f %f %f %f\n", inverted[0, 0], inverted[1, 0], inverted[2, 0], inverted[1, 1], inverted[2, 1], inverted[2, 2]);
        print("%i %i %i     p* %.10f e* %.10f si %f %f %f w %f P %f vi %f %f %f epsilon %.10f raised %f top_1 %f uk %f %f %f h %f\n", pos.x(), pos.y(), pos.z(), hydro_args.p_star, hydro_args.e_star, hydro_args.Si[0], hydro_args.Si[1], hydro_args.Si[2], hydro_args.w, hydro_args.P, vi[0], vi[1], vi[2], epsilon, raised, t1, u_k[0], u_k[1], u_k[2], h);

        auto vars = get_differentiation_variables<5>(vi.z(), 2);

        print("DVars m2 %f m1 %f p1 %f p2 %f\n", vars[0], vars[1], vars[3], vars[4]);

        print("Dbgvi %f %f %f diff %f %f %f\n", vi.x(), vi.y(), vi.z(), diff1(vi.x(), 0, d), diff1(vi.y(), 1, d), diff1(vi.z(), 2, d));

        valuef fin_p_star = max(h_base.p_star[pos, dim] + dp_star * timestep.get(), 0.f);
        valuef fin_e_star = max(h_base.e_star[pos, dim] + de_star * timestep.get(), 0.f);

        print("Si Adjacent m2 %f m1 %f p1 %f p2 %f\n", h_in.Si[0][pos - (v3i){0, 0, 2}, dim], h_in.Si[0][pos - (v3i){0, 0, 1}, dim], h_in.Si[0][pos + (v3i){0, 0, 1}, dim], h_in.Si[0][pos + (v3i){0, 0, 2}, dim]);

        print("%i %i %i out p* %.10f e* %.10f si %f %f %f e*_advect %f Si_advect %f\n", pos.x(), pos.y(), pos.z(), fin_p_star, fin_e_star, as_constant(fin_Si[0]), as_constant(fin_Si[1]), as_constant(fin_Si[2]), e_advect_only, si_advect);
    });
    #endif

    as_ref(h_out.p_star[pos, dim]) = max(h_base.p_star[pos, dim] + dp_star * timestep.get(), 0.f);
    as_ref(h_out.e_star[pos, dim]) = max(h_base.e_star[pos, dim] + de_star * timestep.get(), 0.f);

    as_ref(h_out.Si[0][pos, dim]) = as_constant(fin_Si[0]);
    as_ref(h_out.Si[1][pos, dim]) = as_constant(fin_Si[1]);
    as_ref(h_out.Si[2][pos, dim]) = as_constant(fin_Si[2]);
}

void finalise_hydro(execution_context& ectx, bssn_args_mem<buffer<valuef>> in,
                    hydrodynamic_base_args<buffer_mut<valuef>> hydro,
                    literal<v3i> idim,
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

    if_e(hydro.p_star[pos, dim] <= min_p_star, [&]{
        as_ref(hydro.p_star[pos, dim]) = valuef(0);
        as_ref(hydro.e_star[pos, dim]) = valuef(0);

        as_ref(hydro.Si[0][pos, dim]) = valuef(0);
        as_ref(hydro.Si[1][pos, dim]) = valuef(0);
        as_ref(hydro.Si[2][pos, dim]) = valuef(0);
        return_e();
    });

    if_e(hydro.p_star[pos, dim] < valuef(1e-6f), [&]{
        valuef e_star = declare_e(hydro.e_star[pos, dim]);

        as_ref(hydro.e_star[pos, dim]) = min(e_star, 10 * hydro.p_star[pos, dim]);
    });

    #if 1
    //if_e((hydro.e_star[pos, dim] <= hydro.p_star[pos, dim]) || (hydro.p_star[pos, dim] <= min_p_star * 10), [&]{
        bssn_args args(pos, dim, in);

        valuef p_star = hydro.p_star[pos, dim];
        valuef e_star = hydro.e_star[pos, dim];

        v3f Si = {hydro.Si[0][pos, dim], hydro.Si[1][pos, dim], hydro.Si[2][pos, dim]};

        valuef w = calculate_w(p_star, e_star, args.W, args.cY.invert(), Si);

        valuef epsilon = calculate_epsilon(p_star, e_star, args.W, w);

        valuef h = calculate_h_from_epsilon(epsilon);

        v3f dfsi = {hydro.Si[0][pos, dim], hydro.Si[1][pos, dim], hydro.Si[2][pos, dim]};

        v3f u_k;

        for(int i=0; i < 3; i++)
            u_k[i] = safe_divide(dfsi[i], h * hydro.p_star[pos, dim], 1e-6);

        u_k = clamp(u_k, -1.f, 1.f);

        v3f fin = u_k * h * hydro.p_star[pos, dim];

        as_ref(hydro.Si[0][pos, dim]) = fin[0];
        as_ref(hydro.Si[1][pos, dim]) = fin[1];
        as_ref(hydro.Si[2][pos, dim]) = fin[2];
    //});
    #endif
}

hydrodynamic_plugin::hydrodynamic_plugin(cl::context ctx)
{
    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(init_hydro, "init_hydro");
    }, {"init_hydro"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(calculate_p_kern, "calculate_p");
    }, {"calculate_p"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(calculate_w_kern, "calculate_w");
    }, {"calculate_w"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(advect_all, "advect_all");
    }, {"advect_all"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(evolve_si_p2, "evolve_si_p2");
    }, {"evolve_si_p2"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(evolve_hydro_all, "evolve_hydro_all");
    }, {"evolve_hydro_all"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(finalise_hydro, "finalise_hydro");
    }, {"finalise_hydro"});
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
        args.push_back(pack.dim);
        args.push_back(pack.scale);
        args.push_back(pack.disc.mu_h_cfl);
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

    float damping_timescale = 500;

    hydrodynamic_buffers& bufs_base = *dynamic_cast<hydrodynamic_buffers*>(sdata.buffers[sdata.base_idx]);
    hydrodynamic_buffers& bufs_in = *dynamic_cast<hydrodynamic_buffers*>(sdata.buffers[sdata.in_idx]);
    hydrodynamic_buffers& bufs_out = *dynamic_cast<hydrodynamic_buffers*>(sdata.buffers[sdata.out_idx]);
    hydrodynamic_utility_buffers& ubufs = *dynamic_cast<hydrodynamic_utility_buffers*>(sdata.utility_buffers);

    auto utility_buffers = ubufs.get_buffers();

    auto calc_intermediates = [&](hydrodynamic_buffers& in)
    {
        {
            cl::args args;

            for(auto& i : sdata.bssn_buffers)
                args.push_back(i);

            for(auto i : in.get_buffers())
                args.push_back(i);

            args.push_back(ubufs.w);

            args.push_back(sdata.dim);
            args.push_back(sdata.scale);
            args.push_back(sdata.evolve_points);
            args.push_back(sdata.evolve_length);

            cqueue.exec("calculate_w", args, {sdata.evolve_length}, {128});
        }

        {
            cl::args args;

            for(auto& i : sdata.bssn_buffers)
                args.push_back(i);

            for(auto i : in.get_buffers())
                args.push_back(i);

            args.push_back(ubufs.w);
            args.push_back(ubufs.P);

            args.push_back(sdata.dim);
            args.push_back(sdata.scale);
            args.push_back(sdata.total_elapsed);
            args.push_back(damping_timescale);
            args.push_back(sdata.evolve_points);
            args.push_back(sdata.evolve_length);

            cqueue.exec("calculate_p", args, {sdata.evolve_length}, {128});
        }
    };

    std::vector<cl::buffer> cl_base = bufs_base.get_buffers();

    #if 1
    {
        std::vector<cl::buffer> cl_in = bufs_in.get_buffers();
        std::vector<cl::buffer> cl_out = bufs_out.get_buffers();

        calc_intermediates(bufs_in);

        cl::args args;

        for(auto& i : sdata.bssn_buffers)
            args.push_back(i);

        for(auto& i : cl_base)
            args.push_back(i);

        for(auto& i : cl_in)
            args.push_back(i);

        for(auto& i : cl_out)
            args.push_back(i);

        for(auto& i : utility_buffers)
            args.push_back(i);

        args.push_back(sdata.dim);
        args.push_back(sdata.scale);
        args.push_back(sdata.timestep);
        args.push_back(sdata.total_elapsed);
        args.push_back(damping_timescale);
        args.push_back(sdata.evolve_points);
        args.push_back(sdata.evolve_length);

        cqueue.exec("evolve_hydro_all", args, {sdata.evolve_length}, {128});
    }
    #endif


    #if 0
    {
        std::vector<cl::buffer> cl_in = bufs_in.get_buffers();
        std::vector<cl::buffer> cl_out = bufs_out.get_buffers();

        calc_intermediates(bufs_in);

        cl::args args;

        for(auto& i : sdata.bssn_buffers)
            args.push_back(i);

        for(auto& i : cl_base)
            args.push_back(i);

        for(auto& i : cl_in)
            args.push_back(i);

        for(auto& i : cl_out)
            args.push_back(i);

        for(auto& i : utility_buffers)
            args.push_back(i);

        args.push_back(sdata.dim);
        args.push_back(sdata.scale);
        args.push_back(sdata.timestep);
        args.push_back(sdata.evolve_points);
        args.push_back(sdata.evolve_length);

        cqueue.exec("advect_all", args, {sdata.evolve_length}, {128});
    }


    std::swap(bufs_out.p_star, bufs_in.p_star);
    std::swap(bufs_out.e_star, bufs_in.e_star);
    std::swap(bufs_out.Si[0], bufs_in.Si[0]);
    std::swap(bufs_out.Si[1], bufs_in.Si[1]);
    std::swap(bufs_out.Si[2], bufs_in.Si[2]);

    {
        calc_intermediates(bufs_in);
        std::vector<cl::buffer> cl_in = bufs_in.get_buffers();

        cl::args args;

        for(auto& i : sdata.bssn_buffers)
            args.push_back(i);

        for(auto& i : cl_base)
            args.push_back(i);

        for(auto& i : cl_in)
            args.push_back(i);

        //printf("Buf in e* %p %p %p\n", bufs_base.e_star.native_mem_object.data, bufs_in.e_star.native_mem_object.data, bufs_out.e_star.native_mem_object.data);

        args.push_back(ubufs.intermediate.at(0));
        args.push_back(ubufs.intermediate.at(1));
        args.push_back(ubufs.intermediate.at(2));

        args.push_back(ubufs.intermediate.at(3));

        for(auto& i : utility_buffers)
            args.push_back(i);

        args.push_back(sdata.dim);
        args.push_back(sdata.scale);
        args.push_back(sdata.timestep);
        args.push_back(sdata.total_elapsed);
        args.push_back(damping_timescale);
        args.push_back(sdata.evolve_points);
        args.push_back(sdata.evolve_length);

        cqueue.exec("evolve_si_p2", args, {sdata.evolve_length}, {128});
    }

    std::swap(bufs_in.p_star, bufs_out.p_star);
    std::swap(bufs_in.e_star, bufs_out.e_star);

    std::swap(bufs_out.Si[0], bufs_in.Si[0]);
    std::swap(bufs_out.Si[1], bufs_in.Si[1]);
    std::swap(bufs_out.Si[2], bufs_in.Si[2]);

    std::swap(bufs_out.Si[0], ubufs.intermediate.at(0));
    std::swap(bufs_out.Si[1], ubufs.intermediate.at(1));
    std::swap(bufs_out.Si[2], ubufs.intermediate.at(2));
    std::swap(bufs_out.e_star, ubufs.intermediate.at(3));
    #endif
}

void hydrodynamic_plugin::finalise(cl::context ctx, cl::command_queue cqueue, std::vector<cl::buffer> bssn_buffers, buffer_provider* out, t3i dim, cl::buffer evolve_points, cl_int evolve_length)
{
    hydrodynamic_buffers& bufs = *dynamic_cast<hydrodynamic_buffers*>(out);
    auto all = bufs.get_buffers();

    cl::args args;

    for(auto& i : bssn_buffers)
        args.push_back(i);

    for(auto& i : all)
        args.push_back(i);

    args.push_back(dim);
    args.push_back(evolve_points);
    args.push_back(evolve_length);

    cqueue.exec("finalise_hydro", args, {evolve_length}, {128});
}


void hydrodynamic_plugin::add_args_provider(all_adm_args_mem& mem)
{
    mem.add(full_hydrodynamic_args<buffer<valuef>>());
}

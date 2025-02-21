#include "hydrodynamics.hpp"
#include "init_general.hpp"

///so like. What if I did the projective real strategy?

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
v3f calculate_vi(valuef gA, v3f gB, valuef W, valuef w, valuef epsilon, v3f Si, const unit_metric<valuef, 3, 3>& cY)
{
    valuef h = calculate_h_from_epsilon(epsilon);

    //note to self, actually hand derived this and am sure its correct
    //tol is very intentionally set to 1e-6, breaks if lower than this
    return clamp(-gB + safe_divide(W*W * gA, w*h, 1e-6) * cY.invert().raise(Si), -1.f, 1.f);
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

        return ::calculate_vi(gA, gB, W, w, epsilon, Si, cY);
    }

    ///rhs here to specifically indicate that we're returning -(di Vec v^i), ie the negative
    valuef advect_rhs(valuef in, v3f vi, const derivative_data& d)
    {
        auto leib = [&](valuef v1, valuef v2, int i)
        {
            //return diff1(v1 * v2, i, d);
            return diff1(v1, i, d) * v2 + diff1(v2, i, d) * v1;
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

    valuef e_star_rhs(valuef gA, v3f gB, unit_metric<valuef, 3, 3> cY, valuef W, v3f vi, const derivative_data& d)
    {
        auto icY = cY.invert();

        auto calculate_PQvis = [&]()
        {
            valuef e_m6phi = pow(W, 3.f);

            valuef dkvk = 0;

            for(int k=0; k < 3; k++)
            {
                dkvk += 2 * diff1(vi[k], k, d);
            }

            valuef littledv = dkvk * d.scale;
            valuef Gamma = get_Gamma();

            valuef A = safe_divide(pow(e_star, Gamma) * pow(p_star, Gamma - 1) * pow(e_m6phi, Gamma - 1), pow(w, Gamma - 1), 1e-5f);

            //ctx.add("DBG_A", A);

            ///[0.1, 1.0}
            valuef CQvis = 0.1f;

            valuef PQvis = ternary(littledv < 0, CQvis * A * pow(littledv, 2), valuef{0.f});

            return PQvis;
        };

        valuef e_m6phi = pow(W, 3.f);

        valuef PQvis = calculate_PQvis();

        valuef sum_interior_rhs = 0;

        for(int k=0; k < 3; k++)
        {
            value to_diff = safe_divide(w * vi[k], p_star * e_m6phi, 1e-5);

            sum_interior_rhs += diff1(to_diff, k, d);
        }

        valuef Gamma = get_Gamma();

        valuef p0e = calculate_p0e(W);

        valuef degenerate = safe_divide(valuef{1}, pow(p0e, 1 - 1/Gamma), 1e-5);

        return -degenerate * (PQvis / Gamma) * sum_interior_rhs;
    }
};

///it might be because i'm using hydro_args.eos instead of P
template<typename T>
valuef full_hydrodynamic_args<T>::adm_p(bssn_args& args, const derivative_data& d)
{
    hydrodynamic_concrete hydro_args(d.pos, d.dim, *this);

    valuef epsilon = hydro_args.calculate_epsilon(args.W);

    valuef h = hydro_args.calculate_h_with_eos(args.W);

    return ternary(hydro_args.p_star <= min_p_star * 1,
                   valuef(),
                   h * hydro_args.w * (args.W * args.W * args.W) - hydro_args.eos(args.W));
}

template<typename T>
tensor<valuef, 3> full_hydrodynamic_args<T>::adm_Si(bssn_args& args, const derivative_data& d)
{
    v3f cSi = {this->Si[0][d.pos, d.dim], this->Si[1][d.pos, d.dim], this->Si[2][d.pos, d.dim]};

    valuef p_star = this->p_star[d.pos, d.dim];

    return ternary(p_star <= min_p_star * 1,
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

    return ternary(hydro_args.p_star <= min_p_star * 1,
                   tensor<valuef, 3, 3>(),
                   W2_Sij + hydro_args.eos(args.W) * args.cY.to_tensor());
}

template<typename T>
valuef full_hydrodynamic_args<T>::dbg(bssn_args& args, const derivative_data& d)
{
    //return fabs(p_star[d.pos, d.dim]) * 10;
    //return sqrt(pow(Si[0][d.pos, d.dim], 2.f) + pow(Si[1][d.pos, d.dim], 2.f)) * 10;
    //return sqrt(pow(Si[0][d.pos, d.dim], 2.f) + pow(Si[2][d.pos, d.dim], 2.f)) * 100;
    //return e_star[d.pos, d.dim] * 0.5;
    return fabs(this->Si[0][d.pos, d.dim]) * 100 * 100;
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
    s0.dissipation_coeff = 0.01;
    s0.dissipation_order = 2;

    buffer_descriptor s1;
    s1.name = "cs1";
    s1.dissipation_coeff = 0.01;
    s1.dissipation_order = 2;

    buffer_descriptor s2;
    s2.name = "cs2";
    s2.dissipation_coeff = 0.01;
    s2.dissipation_order = 2;

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
    v3f Si_lo_cfl = pow(cW, -3) * Yij.lower(Si);

    as_ref(hydro.p_star[pos, dim]) = p_star;
    as_ref(hydro.e_star[pos, dim]) = e_star;
    as_ref(hydro.Si[0][pos, dim]) = Si_lo_cfl[0];
    as_ref(hydro.Si[1][pos, dim]) = Si_lo_cfl[1];
    as_ref(hydro.Si[2][pos, dim]) = Si_lo_cfl[2];

    //w also isn't well defined when gA != 1
    //valuef real_w = p_star * gA * u0;

    ///ok so. I'm trying to answer the quesiton of why w = p* gA u0
    ///is not the same answer as calculate_w
    valuef w = calculate_w(p_star, e_star, args.W, args.cY.invert(), Si_lo_cfl);

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

void calculate_hydro_intermediates(execution_context& ectx, bssn_args_mem<buffer<valuef>> in, hydrodynamic_base_args<buffer<valuef>> hydro, hydrodynamic_utility_args<buffer_mut<valuef>> out,
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
        as_ref(out.P[pos, dim]) = valuef(0);
        as_ref(out.w[pos, dim]) = valuef(0);

        return_e();
    });

    bssn_args args(pos, dim, in);

    valuef w = calculate_w(p_star, e_star, args.W, args.cY.invert(), Si);
    w = max(w, p_star * args.gA * 1);

    valuef P = max(eos(args.W, w, p_star, e_star), 0.f);

    as_ref(out.w[pos, dim]) = w;
    as_ref(out.P[pos, dim]) = P;
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

    if_e(hydro_args.p_star <= min_p_star, [&]{
        as_ref(h_out.p_star[pos, dim]) = valuef(min_p_star - 1e-9f);
        as_ref(h_out.e_star[pos, dim]) = hydro_args.e_star;
        as_ref(h_out.Si[0][pos, dim]) = hydro_args.Si[0];
        as_ref(h_out.Si[1][pos, dim]) = hydro_args.Si[1];
        as_ref(h_out.Si[2][pos, dim]) = hydro_args.Si[2];
        return_e();
    });

    derivative_data d;
    d.pos = pos;
    d.dim = idim.get();
    d.scale = scale.get();

    valuef p_star = hydro_args.p_star;
    valuef e_star = hydro_args.e_star;
    v3f Si = hydro_args.Si;
    valuef w = hydro_args.w;

    v3f vi = hydro_args.calculate_vi(args.gA, args.gB, args.W, args.cY);

    valuef dp_star = hydro_args.advect_rhs(p_star, vi, d);
    valuef de_star = hydro_args.advect_rhs(e_star, vi, d);
    v3f dSi_p1 = hydro_args.advect_rhs(Si, vi, d);

    valuef de_star_advect = de_star;

    de_star += hydro_args.e_star_rhs(args.gA, args.gB, args.cY, args.W, vi, d);

    valuef h = hydro_args.calculate_h_with_eos(args.W);

    for(int k=0; k < 3; k++)
    {
        valuef p1 = (-args.gA * pow(max(args.W, 0.001f), -3.f)) * diff1(hydro_args.P, k, d);
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

                #if 0
                if(k == 2)
                {
                    auto dump = [&](v3i offset){
                        unit_metric<valuef, 3, 3> met;
                        met[0, 0] = in.cY[0][pos + offset, dim];
                        met[1, 0] = in.cY[1][pos + offset, dim];
                        met[2, 0] = in.cY[2][pos + offset, dim];
                        met[0, 1] = in.cY[1][pos + offset, dim];
                        met[0, 2] = in.cY[2][pos + offset, dim];

                        met[1, 1] = in.cY[3][pos + offset, dim];
                        met[1, 2] = in.cY[4][pos + offset, dim];
                        met[2, 1] = in.cY[4][pos + offset, dim];
                        met[2, 2] = in.cY[5][pos + offset, dim];

                        //if_e(pos.x() == 59 && pos.y() == 81 && pos.z() == 55, [&]{
                            print("Dump %i %f %f %f %f %f %f\n", offset.z(), in.cY[0][pos + offset, dim],
                                in.cY[1][pos + offset, dim],
                                in.cY[2][pos + offset, dim],
                                in.cY[3][pos + offset, dim],
                                in.cY[4][pos + offset, dim],
                                in.cY[5][pos + offset, dim]);
                        //});
                    };

                    //auto cY_up_0 = in.cY[0][pos - (v3i){0, 0, 1}, dim];
                    //auto cY2 = in.cY[0][pos + (v3i){0, 0, 1}, dim];



                    if_e(pos.x() == 81 && pos.y() == 75 && pos.z() == 59, [&]{
                        dump({0, 0, -1});
                        dump({0, 0, -2});
                        dump({0, 0, 1});
                        dump({0, 0, 2});

                        print("Deriv %f %i %i l1 %f l2 %f\n", deriv, valuei(i), valuei(j), l1, l2);
                    });
                }
                #endif

                p4 += 0.5f * args.gA * args.W * args.W * l1 * l2 * deriv;
            }
        }

        valuef w2_m_p2_calc = w2_m_p2(p_star, e_star, args.W, args.cY.invert(), Si, w);

        valuef p5 = safe_divide(args.gA * h * (w*w - p_star*p_star), w) * (diff1(args.W, k, d) / max(args.W, 0.001f));

        /*if_e(pos.x() == 80 && pos.y() == 76 && pos.z() == 66, [&]{
            print("hi %f %f %f %f %f\n", p1, p2, p3, p4, p5);
        });*/

        dSi_p1[k] += (p1 + p2 + p3 + p4 + p5);

        /*if(k == 2)
        {
            if_e(pos.x() == 81 && pos.y() == 75 && pos.z() == 57, [&]{
                print("Test %f %f %f %f %f\n", p1, p2, p3, p4, p5);
            });
        }*/
    }

    valuef fin_p_star = max(h_base.p_star[pos, dim] + timestep.get() * dp_star, 0.f);
    valuef fin_e_star = max(h_base.e_star[pos, dim] + timestep.get() * de_star, 0.f);

    valuef max_p = 1;

    ///looks like the basic problem is the superheating issue
    fin_e_star = ternary(fin_p_star < (1e-6f * max_p), min(fin_e_star, 10 * fin_p_star), fin_e_star);

    ///temporary
    fin_e_star = min(fin_e_star, 10 * fin_p_star);

    mut_v3f fin_Si = declare_mut_e((v3f){});

    for(int i=0; i < 3; i++)
    {
        as_ref(fin_Si[i]) = h_base.Si[i][pos, dim] + timestep.get() * dSi_p1[i];
    }

    ///p* 0.0025857189 e* 0.0258571822
    ///1e-3 is the current density

    #define CLAMP_HIGH_VELOCITY
    #ifdef CLAMP_HIGH_VELOCITY
    //if_e(p_star >= min_p_star && p_star < 1e-4, [&]{
    //if_e(p_star >= min_p_star && p_star < 1e-5, [&]{
        /*v3f dfsi = declare_e(fin_Si);

        v3f u_k;

        for(int i=0; i < 3; i++)
            u_k[i] = safe_divide(dfsi[i], h * p_star, 1e-6);

        u_k = clamp(u_k, -0.2f, 0.2f);

        as_ref(fin_Si) = u_k * h * p_star;*/
    //});
    #endif

    /*    p* 0.000222 e* 0.002546 si -0.000010 -0.000143 0.000003 w 0.000256 P 0.000002 vi -0.064003 -0.347698 0.014055 epsilon 0.017120 top_1 -0.000097
    p* 0.000107 e* 0.000000 si 0.000171 -0.005759 -0.000041 w 0.005367 P 0.000000 vi -0.044568 -0.736823 0.007491 epsilon 0.000000 top_1 -0.004060
out p* 0.000240 e* 0.030715 si -0.000013 -0.018418 0.000004
out p* 0.000136 e* 0.000000 si -0.000079 0.013146 0.000016
Pos 80 71 65
Pos 80 73 65*/
    ///Pos p1 80 75 57

    /*if_e(!isfinite(fin_p_star) || !isfinite(fin_e_star) || !isfinite(fin_Si[0]) || !isfinite(fin_Si[1]) || !isfinite(fin_Si[2]), [&]{
        print("Pos p1 %i %i %i\n", pos.x(), pos.y(), pos.z());
    });*/

    ///Pos 83 78 63

    /*if_e(pos.x() == 83 && pos.y() == 78 && pos.z() == 63, [&]{
        if_e(!isfinite())

        print("p* %f e* %f si %f %f %f w %f P %f\n", p_star, e_star, Si[0], Si[1], Si[2], w, hydro_args.P);
    });*/

    /*if_e(!isfinite(p_star) || !isfinite(e_star) || !isfinite(Si[0]) || !isfinite(Si[1]) || !isfinite(Si[2]), [&]{
        print("Pos p2 %i %i %i\n", pos.x(), pos.y(), pos.z());
    });*/

    /*de* advect -0.308110 full -0.308110
    de* advect 0.002658 full 0.002658
        p* 0.0000860349 e* 0.0064756656 si -0.000017 0.000672 0.000012 w 0.000582 P 0.000001 vi -0.109995 0.792189 0.008223 epsilon 0.0538912192 raised 0.000732 top_1 0.000508
        p* 0.0000013049 e* 0.0050032577 si -0.000000 0.000000 0.000000 w 0.000001 P 0.000015 vi -0.013618 0.004324 0.001394 epsilon 15.0190372467 raised 0.000000 top_1 0.000000
    out p* 0.0000007338 e* 0.0000073375 si -0.000000 0.000004 0.000004
    out p* 0.0000890945 e* 0.0000000000 si -0.000021 -0.035679 -0.000218
    Pos 73 85 65
    Pos 73 83 65*/

    //if_e(pos.x() == 73 && pos.y() == 85 && pos.z() == 65, [&]{
    //if_e(pos.x() == 80 && pos.y() == 75 && pos.z() == 57, [&]{
    //if_e(fabs(dSi_p1[1]) > 0.1f, [&]{
    //if_e(pos.x() == 75 && pos.y() == 82 && pos.z() == 69, [&]{
    ///58 83 65     p* 0.0000108149 e* 0.0001215594 si -0.000001 -0.000003 nan w 0.000009 P 0.000000 vi -1.000000 -1.000000 -1.000000 epsilon 0.0012506038 raised -nan top_1 -nan
    //if_e(!isfinite(p_star) || !isfinite(e_star) || !isfinite(Si[0]) || !isfinite(Si[1]) || !isfinite(Si[2]), [&]{

    pin(fin_p_star);
    pin(fin_e_star);

    ///Pos p1 59 81 55

    /*Pos p1 81 75 57
      Pos p1 59 81 57
      Pos p1 53 59 58
      Pos p1 81 75 58
      Pos p1 59 81 58*/

    //Pos p1 54 79 63

    /*if_e(!isfinite(fin_p_star) || !isfinite(fin_e_star) || !isfinite(fin_Si[0]) || !isfinite(fin_Si[1]) || !isfinite(fin_Si[2]) || !isfinite(p_star) || !isfinite(e_star) || !isfinite(Si[0]) || !isfinite(Si[1]) || !isfinite(Si[2])
         || !isfinite(args.cY[0, 0]) || !isfinite(args.cY[1, 0]) || !isfinite(args.cY[2, 0]) || !isfinite(args.cY[1, 1]) || !isfinite(args.cY[2, 1]) || !isfinite(args.cY[2, 2]), [&]{
        print("Pos p1 %i %i %i\n", pos.x(), pos.y(), pos.z());
    });*/

    //54 79 63
    //71 63 53
    #if 0
    //if_e(pos.x() == 59 && pos.y() == 81 && pos.z() == 55, [&]{
    if_e(pos.x() == dim.x()/2 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
    //if_e(pos.x() == 71 && pos.y() == 63 && pos.z() == 53, [&]{
    //if_e(pos.x() == 59 && pos.y() == 81 && pos.z() == 57, [&]{
        valuef epsilon = hydro_args.calculate_epsilon(args.W);

        unit_metric<valuef, 3, 3> cY = args.cY;
        pin(cY);

        auto inverted = cY.invert();

        /*valuef l1 = cY[1, 1] * cY[2, 2] - cY[2, 1] * cY[1, 2];
        valuef l2 = cY[0, 2] * cY[2, 1] - cY[0, 1] * cY[2, 2];
        valuef l3 = cY[0, 1] * cY[1, 2] - cY[0, 2] * cY[1, 1];
        valuef l4 = cY[1, 2] * cY[2, 0] - cY[1, 0] * cY[2, 2];
        valuef l5 = cY[0, 0] * cY[2, 2] - cY[0, 2] * cY[2, 0];
        valuef l6 = cY[1, 0] * cY[0, 2] - cY[0, 0] * cY[1, 2];
        valuef l7 = cY[1, 0] * cY[2, 1] - cY[2, 0] * cY[1, 1];
        valuef l8 = cY[2, 0] * cY[0, 1] - cY[0, 0] * cY[2, 1];
        valuef l9 = cY[0, 0] * cY[1, 1] - cY[1, 0] * cY[0, 1];*/

        valuef raised = args.cY.invert().raise(Si)[1];

        v3f dfsi = declare_e(fin_Si);

        v3f u_k;

        for(int i=0; i < 3; i++)
            u_k[i] = safe_divide(dfsi[i], h * p_star);

        //print("what is my purpose dfsi2 %f h %f p* %f h*p* %f dfsi / (h*p) %f\n", dfsi[2], h, p_star, h*p_star, dfsi[2] / (h * p_star));

        valuef t1 = args.W * args.W * args.gA * raised;

        //print("Components %f %f %f %f %f %f %f %f %f\n", l1, l2, l3, l4, l5, l6, l7, l8, l9);

        /*Test -0.000000 0.000000 -0.000000 -0.000000 -0.000000
          met? 1.105540 -0.030693 -0.042335 0.910168 0.035884 0.997689
          imet? 0.906776 0.029102 0.037431 1.101193 -0.038371 1.005285
81 75 59     p* 0.0000000010 e* 0.0000000104 si -0.000000 0.000000 0.000000 w 0.000000 P 0.000000 vi -0.008797 0.007538 0.001144 epsilon 0.0000000008 raised 0.000000 top_1 0.000000 uk -nan -nan -nan
81 75 59 out p* 0.0000000010 e* 0.0000000102 si -nan -nan -nan*/

        //print("de* advect %f full %f mult %f\n", de_star_advect, de_star, de_star * timestep.get());
        print("met? %f %f %f %f %f %f\n", args.cY[0, 0], args.cY[1, 0], args.cY[2, 0], args.cY[1, 1], args.cY[2, 1], args.cY[2, 2]);
        print("imet? %f %f %f %f %f %f\n", inverted[0, 0], inverted[1, 0], inverted[2, 0], inverted[1, 1], inverted[2, 1], inverted[2, 2]);
        print("%i %i %i     p* %.10f e* %.10f si %f %f %f w %f P %f vi %f %f %f epsilon %.10f raised %f top_1 %f uk %f %f %f h %f\n", pos.x(), pos.y(), pos.z(), p_star, e_star, Si[0], Si[1], Si[2], w, hydro_args.P, vi[0], vi[1], vi[2], epsilon, raised, t1, u_k[0], u_k[1], u_k[2], h);
        print("%i %i %i out p* %.10f e* %.10f si %f %f %f\n", pos.x(), pos.y(), pos.z(), fin_p_star, fin_e_star, as_constant(fin_Si[0]), as_constant(fin_Si[1]), as_constant(fin_Si[2]));

        //print("Base e* %.10f\n", h_base.e_star[pos, dim]);
        //print("Pos %i %i %i\n", pos.x(), pos.y(), pos.z());
    });
    #endif

    ///i think this is causing problems
    ///so, backwards euler: we do one step, and flush to 0. That becomes out new output buffer, but base still has stuff in it
    ///because in == 0, we flood shit in, increasing the temperature
    ///backwards euler doesn't work well with discontinuities. Todo: Enforce this constraint somewhere else
    ///if p* in < min_p_star, quietly do nothing
    /*if_e(fin_p_star <= min_p_star, [&]{
        as_ref(h_out.p_star[pos, dim]) = valuef(0);
        as_ref(h_out.e_star[pos, dim]) = valuef(0);

        for(int i=0; i < 3; i++)
        {
            as_ref(h_out.Si[i][pos, dim]) = valuef(0);
        }

        return_e();
    });*/

    as_ref(h_out.p_star[pos, dim]) = fin_p_star;
    as_ref(h_out.e_star[pos, dim]) = fin_e_star;

    for(int i=0; i < 3; i++)
    {
        as_ref(h_out.Si[i][pos, dim]) = as_constant(fin_Si[i]);
    }
}

void finalise_hydro(execution_context& ectx,
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
    });
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

    {
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
}

void hydrodynamic_plugin::finalise(cl::context ctx, cl::command_queue cqueue, buffer_provider* out, t3i dim, cl::buffer evolve_points, cl_int evolve_length)
{
    hydrodynamic_buffers& bufs = *dynamic_cast<hydrodynamic_buffers*>(out);
    auto all = bufs.get_buffers();

    cl::args args;

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

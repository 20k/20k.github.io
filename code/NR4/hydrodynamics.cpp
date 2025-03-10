#include "hydrodynamics.hpp"
#include "init_general.hpp"

///so like. What if I did the projective real strategy?

//stable with 1e-6, but the neutron star dissipates
constexpr float min_p_star = 1e-9f;

template<typename T>
inline
auto safe_divide(const auto& top, const T& bottom, float tol = 1e-8)
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

    ///this probably isn't super regular

    ///(W^3^Gamma-1 / w^(gamma - 1)) * e*^Gamma * p*^(Gamma - 2)
    ///p*^(Gamma - 2) * W^3^Gamma-1 * e*^Gamma / w^(Gamma - 1)
    ///p*^(Gamma - 2) * W^3^Gamma-1 * e* * e*^Gamma-1 / w^(Gamma - 1)

    return pow(p_star, Gamma - 2) * pow(e_m6phi, Gamma - 1) * e_star * safe_divide(pow(e_star, Gamma - 1), pow(w, Gamma - 1));
}

valuef calculate_p0e(valuef p_star, valuef e_star, valuef W, valuef w)
{
    valuef e_m6phi = W*W*W;
    valuef Gamma = get_Gamma();
    valuef iv_au0 = safe_divide(p_star, w);

    return pow(max(e_star * e_m6phi * iv_au0, 0.f), Gamma);
}

valuef calculate_p0(valuef p_star, valuef W, valuef w)
{
    valuef iv_au0 = safe_divide(p_star, w);

    //p* = p0 au0 e^6phi
    //au0 = w / p*
    ///p* / (w/p*) = p0 e^6phi
    ///p* * p*/w = p0 e^6phi
    ///p*^2/w = p0 e^6phi
    ///p*^2/w e^-6phi = p0

    valuef e_m6phi = W*W*W;

    return p_star * e_m6phi * safe_divide(p_star, w);
}

valuef eos(valuef W, valuef w, valuef p_star, valuef e_star)
{
    valuef Gamma = get_Gamma();
    return calculate_p0e(p_star, e_star, W, w) * (Gamma - 1);
}

//todo: I may need to set vi to 0 here manually
//or, I may need to remove the leibnitz that I'm doing
///todo: try setting this to zero where appropriate
v3f calculate_vi(valuef gA, v3f gB, valuef W, valuef w, valuef epsilon, v3f Si, const unit_metric<valuef, 3, 3>& cY, valuef p_star)
{
    valuef h = calculate_h_from_epsilon(epsilon);

    v3f Si_upper = cY.invert().raise(Si);

    //note to self, actually hand derived this and am sure its correct
    //tol is very intentionally set to 1e-6, breaks if lower than this
    v3f real_value = -gB + (W*W * gA / h) * safe_divide(Si_upper, w, 1e-6);

    //produces a lot longer inspirals
    //return real_value;
    return ternary(p_star <= min_p_star, (v3f){}, real_value);
}

v3f calculate_ui(valuef p_star, valuef epsilon, v3f Si, valuef w, valuef gA, v3f gB, const unit_metric<valuef, 3, 3>& cY)
{
    valuef h = calculate_h_from_epsilon(epsilon);

    v3f u_k;

    for(int i=0; i < 3; i++)
        u_k[i] = safe_divide(Si[i], h * p_star, 1e-6);

    valuef u0 = safe_divide(w, p_star * gA);

    return -gB * u0 + cY.invert().raise(u_k);
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

    valuef A = pow(e_star, Gamma) * pow(e_m6phi, Gamma - 1) * safe_divide(pow(p_star, Gamma - 1), pow(w, Gamma - 1), 1e-6f);

    //ctx.add("DBG_A", A);

    ///[0.1, 1.0]
    valuef CQvis = 1.f;

    ///it looks like the littledv ?: is to only turn on viscosity when the flow is compressive
    #define COMPRESSIVE_VISCOSITY
    #ifdef COMPRESSIVE_VISCOSITY
    valuef PQvis = ternary(littledv < 0, CQvis * A * pow(littledv, 2), valuef{0.f});
    #else
    valuef PQvis = CQvis * A * pow(littledv, 2);
    #endif

    valuef linear_damping = exp(-(total_elapsed * total_elapsed) / (2 * linear_damping_timescale * linear_damping_timescale));

    ///paper i'm looking at only turns on viscosity inside a star, ie p > pcrit. We could calculate a crit value
    ///or, we could simply make this time variable, though that's kind of annoying
    valuef CLvis = 1.f * linear_damping;
    valuef n = 1;

    valuef PLvis = ternary(littledv < 0, -CLvis * sqrt((get_Gamma()/n) * p_star * A) * littledv, valuef(0.f));

    return ternary(linear_damping_timescale <= 0.f, PQvis, PQvis + PLvis);
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
        return ::calculate_p0e(p_star, e_star, W, w);
    }

    valuef calculate_p0(valuef W)
    {
        return ::calculate_p0(p_star, W, w);
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

    v3f calculate_ui(valuef gA, v3f gB, valuef W, const unit_metric<valuef, 3, 3>& cY)
    {
        valuef epsilon = calculate_epsilon(W);

        return ::calculate_ui(p_star, epsilon, Si, w, gA, gB, cY);
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
            value to_diff = safe_divide(w, p_star, 1e-6) * vi[k] * e_6phi;

            sum_interior_rhs += diff1(to_diff, k, d);
        }

        valuef Gamma = get_Gamma();

        valuef p0e = calculate_p0e(W);

        valuef degenerate = safe_divide(valuef{1}, pow(p0e, 1 - 1/Gamma), 1e-6);

        return -degenerate * (Pvis / Gamma) * sum_interior_rhs;
    }
};

template<typename T>
valuef full_hydrodynamic_args<T>::get_density(bssn_args& args, const derivative_data& d)
{
    hydrodynamic_concrete hydro_args(d.pos, d.dim, *this);

    //return hydro_args.p_star;

    return ternary(hydro_args.p_star >= min_p_star * 10, hydro_args.calculate_p0(args.W), valuef(0.f));
}

template<typename T>
valuef full_hydrodynamic_args<T>::get_energy(bssn_args& args, const derivative_data& d)
{
    //return get_density(args, d) * 10;

    hydrodynamic_concrete hydro_args(d.pos, d.dim, *this);

    return ternary(hydro_args.p_star >= min_p_star * 10, hydro_args.calculate_epsilon(args.W), valuef(0.f));
}

template<typename T>
v4f full_hydrodynamic_args<T>::get_4_velocity(bssn_args& args, const derivative_data& d)
{
    hydrodynamic_concrete hydro_args(d.pos, d.dim, *this);

    v3f ui = hydro_args.calculate_ui(args.gA, args.gB, args.W, args.cY);

    valuef u0 = safe_divide(hydro_args.w, hydro_args.p_star * args.gA);

    v4f velocity = {u0, ui.x(), ui.y(), ui.z()};

    return ternary(hydro_args.p_star >= min_p_star * 10, velocity, (v4f){1,0,0,0});
}

template<typename T>
v3f full_hydrodynamic_args<T>::get_colour(bssn_args& args, const derivative_data& d)
{
    v3i pos = d.pos;
    v3i dim = d.dim;

    v3f raw_colour = {this->colour[0][pos, dim], this->colour[1][pos, dim], this->colour[2][pos, dim]};

    return raw_colour / max(this->p_star[pos, dim], 1e-5f);
}

template<typename T>
valuef full_hydrodynamic_args<T>::adm_p(bssn_args& args, const derivative_data& d)
{
    hydrodynamic_concrete hydro_args(d.pos, d.dim, *this);

    valuef h = hydro_args.calculate_h_with_eos(args.W);

    return ternary(hydro_args.p_star <= 0,
                   {},
                   hydro_args.w * h * pow(args.W, 3.f) - hydro_args.eos(args.W));
}

template<typename T>
tensor<valuef, 3> full_hydrodynamic_args<T>::adm_Si(bssn_args& args, const derivative_data& d)
{
    v3f cSi = {this->Si[0][d.pos, d.dim], this->Si[1][d.pos, d.dim], this->Si[2][d.pos, d.dim]};

    valuef p_star = this->p_star[d.pos, d.dim];

    return ternary(p_star <= 0,
                   {},
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
                   {},
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
    s0.dissipation_coeff = 0.05;
    s0.dissipation_order = 4;

    buffer_descriptor s1;
    s1.name = "cs1";
    s1.dissipation_coeff = 0.05;
    s1.dissipation_order = 4;

    buffer_descriptor s2;
    s2.name = "cs2";
    s2.dissipation_coeff = 0.05;
    s2.dissipation_order = 4;

    buffer_descriptor c0;
    c0.name = "c0";
    c0.dissipation_coeff = p.dissipation_coeff;

    buffer_descriptor c1;
    c1.name = "c1";
    c1.dissipation_coeff = p.dissipation_coeff;

    buffer_descriptor c2;
    c2.name = "c2";
    c2.dissipation_coeff = p.dissipation_coeff;

    return {p, e, s0, s1, s2, c0, c1, c2};
}

std::vector<cl::buffer> hydrodynamic_buffers::get_buffers()
{
    return {p_star, e_star, Si[0], Si[1], Si[2], colour[0], colour[1], colour[2]};
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

    if(use_colour)
    {
        for(auto& i : colour)
        {
            i.alloc(sizeof(cl_float) * cells);
            i.set_to_zero(cqueue);
        }
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
                buffer<valuei> indices, eos_gpu eos_data, buffer<v3f> colour_in, bool use_colour)
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
    //with raised index
    v3f Si_cfl = {Si_cfl_b[0][pos, dim], Si_cfl_b[1][pos, dim], Si_cfl_b[2][pos, dim]};

    valuef mu_h = mu_h_cfl * pow(phi, -8);
    //with raised index
    v3f Si = Si_cfl * pow(phi, -10);

    //ok so, delta lambda = -2 pi phi^-3 ppw2p, where ppw2p = mu_h_cfl

    ///wait, i neglected Aij.. which isn't correct?
    /*{
        auto d0 = get_differentiation_variables<3>(phi, 0);
        auto d1 = get_differentiation_variables<3>(phi, 1);
        auto d2 = get_differentiation_variables<3>(phi, 2);

        valuef laplacian = (d0[0] + d0[2] + d1[0] + d1[2] + d2[0] + d2[2] - 6 * phi) / (scale.get() * scale.get());

        valuef ppw2p = mu_h_cfl;

        valuef rhs = -2 * M_PI * pow(phi, -3.f) * ppw2p;

        print("hi lap %f ppw2p %f err %f\n", laplacian, rhs, (rhs - laplacian));
    }*/


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

    //with raised index
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

    as_ref(hydro.p_star[pos, dim]) = p_star;
    as_ref(hydro.e_star[pos, dim]) = e_star;
    as_ref(hydro.Si[0][pos, dim]) = Si_lo_cfl[0];
    as_ref(hydro.Si[1][pos, dim]) = Si_lo_cfl[1];
    as_ref(hydro.Si[2][pos, dim]) = Si_lo_cfl[2];

    //strictly speaking i don't need to set these
    //in the most technical sense you might consider that we're initialising the boundary condition here
    //but w = P = 0 at the boundary
    as_ref(hydro.w[pos, dim]) = w;
    as_ref(hydro.P[pos, dim]) = eos(args.W, w, p_star, e_star);

    if(use_colour)
    {
        for(int i=0; i < (int)hydro.colour.size(); i++)
            as_ref(hydro.colour[i][pos, dim]) = colour_in[index][i] * p_star;
    }

    ///this looks correct too???
    /*if_e(pos.x() == 50 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        valuef adm_p = w * h * pow(args.W, 3.f) - eos(args.W, w, p_star, e_star);

        auto d0 = get_differentiation_variables<3>(phi, 0);
        auto d1 = get_differentiation_variables<3>(phi, 1);
        auto d2 = get_differentiation_variables<3>(phi, 2);

        tensor<valuef, 3, 3> Aij = args.cA;

        auto Kij = Aij / (args.W * args.W);

        auto lAij = pow(phi, -2) * Kij;

        auto Yij = args.cY / (args.W * args.W);

        auto lYij = Yij * pow(phi, 4.f);
        auto lAIJ = raise_both(lAij, lYij.invert());

        valuef middle = -(1.f/8.f) * pow(phi, -7.f) * sum_multiply(lAIJ, lAij);

        valuef laplacian = (d0[0] + d0[2] + d1[0] + d1[2] + d2[0] + d2[2] - 6 * phi) / (scale.get() * scale.get());

        valuef ppw2p = mu_h_cfl;

        valuef rhs = middle + -2 * M_PI * pow(phi, 5.f) * mu_h;
        //valuef rhs = middle + -2 * M_PI * pow(phi, -3.f) * ppw2p;

        print("hi lap %f ppw2p %f err %f rho %f\n", laplacian, rhs, (rhs - laplacian), adm_p);
    });*/
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

#define MIN_LAPSE 0.15f
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

    mut<valuef> P = declare_mut_e(max(eos(args.W, w, p_star, e_star), 0.f));

    if_e(args.gA >= MIN_VISCOSITY_LAPSE, [&]{
        valuef epsilon = calculate_epsilon(p_star, e_star, args.W, w);
        v3f vi = calculate_vi(args.gA, args.gB, args.W, w, epsilon, Si, args.cY, p_star);
        as_ref(P) += calculate_Pvis(args.W, vi, p_star, e_star, w, d, total_elapsed.get(), damping_timescale.get());
    });

    as_ref(P_out[pos, dim]) = as_constant(P);
}

void evolve_hydro_all(execution_context& ectx, bssn_args_mem<buffer<valuef>> in,
                  hydrodynamic_base_args<buffer<valuef>> h_base, hydrodynamic_base_args<buffer<valuef>> h_in, hydrodynamic_base_args<buffer_mut<valuef>> h_out,
                  hydrodynamic_utility_args<buffer<valuef>> util,
                  literal<v3i> idim, literal<valuef> scale, literal<valuef> timestep, literal<valuef> total_elapsed, literal<valuef> damping_timescale,
                  buffer<tensor<value<short>, 3>> positions, literal<valuei> positions_length, bool use_colour)
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

    valuei boundary_dist = distance_to_boundary(pos, dim);

    bssn_args args(pos, dim, in);
    hydrodynamic_concrete hydro_args(pos, dim, h_in, util);

    if_e(hydro_args.p_star <= min_p_star, [&]{
        as_ref(h_out.p_star[pos, dim]) = h_in.p_star[pos, dim];
        as_ref(h_out.e_star[pos, dim]) = h_in.e_star[pos, dim];
        as_ref(h_out.Si[0][pos, dim]) = h_in.Si[0][pos, dim];
        as_ref(h_out.Si[1][pos, dim]) = h_in.Si[1][pos, dim];
        as_ref(h_out.Si[2][pos, dim]) = h_in.Si[2][pos, dim];

        if(use_colour)
        {
            for(int i=0; i < h_out.colour.size(); i++)
                as_ref(h_out.colour[i][pos, dim]) = h_in.colour[i][pos, dim];
        }

        return_e();
    });

    ///todo, make all this a bit more generic
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

        if(use_colour)
        {
            for(int i=0; i < h_out.colour.size(); i++)
                as_ref(h_out.colour[i][pos, dim]) = h_base.colour[i][pos, dim] + damp * -h_in.colour[i][pos, dim] * timestep.get();
        }

        return_e();
    });

    v3f vi = hydro_args.calculate_vi(args.gA, args.gB, args.W, args.cY);

    valuef dp_star = hydro_args.advect_rhs(hydro_args.p_star, vi, d);
    mut<valuef> de_star = declare_mut_e(hydro_args.advect_rhs(hydro_args.e_star, vi, d));
    v3f dSi = hydro_args.advect_rhs(hydro_args.Si, vi, d);

    if_e(args.gA >= MIN_VISCOSITY_LAPSE, [&]{
        as_ref(de_star) += hydro_args.e_star_rhs(args.gA, args.gB, args.cY, args.W, vi, d, total_elapsed.get(), damping_timescale.get());
    });

    valuef w = hydro_args.w;
    valuef h = hydro_args.calculate_h_with_eos(args.W);

    v3f dSi_p1;

    ///we could use the advanced Si here
    for(int k=0; k < 3; k++)
    {
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

        valuef p5 = args.gA * h * (w - hydro_args.p_star * safe_divide(hydro_args.p_star, w)) * (diff1(args.W, k, d) / max(args.W, 0.1f));
        //valuef p5 = args.gA * h * safe_divide(w2_m_p2_calc, w) * (diff1(args.W, k, d) / max(args.W, 0.1f));

        dSi_p1[k] += (p1 + p2 + p3 + p4 + p5);
    }

    dSi += dSi_p1;

    v3f base_Si = {h_base.Si[0][pos, dim], h_base.Si[1][pos, dim], h_base.Si[2][pos, dim]};

    v3f fin_Si = base_Si + timestep.get() * dSi;

    valuef fin_p_star = h_base.p_star[pos, dim] + dp_star * timestep.get();
    valuef fin_e_star = h_base.e_star[pos, dim] + de_star * timestep.get();

    valuef boundary_damp = 0.25f * timestep.get();

    fin_p_star += ternary(boundary_dist <= 15, -hydro_args.p_star * boundary_damp, {});
    fin_e_star += ternary(boundary_dist <= 15, -hydro_args.e_star * boundary_damp, {});
    fin_Si += ternary(boundary_dist <= 15, -hydro_args.Si * boundary_damp, {});

    as_ref(h_out.p_star[pos, dim]) = max(fin_p_star, 0.f);
    as_ref(h_out.e_star[pos, dim]) = max(fin_e_star, 0.f);

    as_ref(h_out.Si[0][pos, dim]) = fin_Si[0];
    as_ref(h_out.Si[1][pos, dim]) = fin_Si[1];
    as_ref(h_out.Si[2][pos, dim]) = fin_Si[2];

    if(use_colour)
    {
        for(int i=0; i < (int)h_in.colour.size(); i++)
        {
            valuef dt_col = hydro_args.advect_rhs(h_in.colour[i][pos, dim], vi, d);

            valuef fin_col = h_base.colour[i][pos, dim] + dt_col * timestep.get();

            fin_col += ternary(boundary_dist <= 15, -h_in.colour[i][pos, dim] * boundary_damp, {});

            as_ref(h_out.colour[i][pos, dim]) = fin_col;
        }
    }
}

void finalise_hydro(execution_context& ectx, bssn_args_mem<buffer<valuef>> in,
                    hydrodynamic_base_args<buffer_mut<valuef>> hydro,
                    literal<v3i> idim,
                    buffer<tensor<value<short>, 3>> positions, literal<valuei> positions_length, bool use_colour)
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

    valuei boundary_dist = distance_to_boundary(pos, dim);

    //it should be impossible to hit the secondary boundary dist condition in a way that has
    //any physical impact
    if_e(hydro.p_star[pos, dim] <= min_p_star || boundary_dist <= 3, [&]{
        as_ref(hydro.p_star[pos, dim]) = valuef(0);
        as_ref(hydro.e_star[pos, dim]) = valuef(0);

        as_ref(hydro.Si[0][pos, dim]) = valuef(0);
        as_ref(hydro.Si[1][pos, dim]) = valuef(0);
        as_ref(hydro.Si[2][pos, dim]) = valuef(0);

        if(use_colour)
        {
            for(int i=0; i < hydro.colour.size(); i++)
                as_ref(hydro.colour[i][pos, dim]) = valuef(0);
        }

        return_e();
    });

    if_e(hydro.p_star[pos, dim] < min_p_star * 10, [&]{
        valuef e_star = declare_e(hydro.e_star[pos, dim]);

        as_ref(hydro.e_star[pos, dim]) = min(e_star, 10 * hydro.p_star[pos, dim]);
    });

    //test bound
    mut<valuef> bound = declare_mut_e(valuef(0.9));

    /*if_e((hydro.e_star[pos, dim] <= hydro.p_star[pos, dim]) || (hydro.p_star[pos, dim] <= min_p_star * 10), [&]{
        as_ref(bound) = valuef(0.2f);
    });*/

    bssn_args args(pos, dim, in);

    valuef p_star = hydro.p_star[pos, dim];
    valuef e_star = hydro.e_star[pos, dim];

    v3f Si = {hydro.Si[0][pos, dim], hydro.Si[1][pos, dim], hydro.Si[2][pos, dim]};
    pin(Si);

    valuef w = calculate_w(p_star, e_star, args.W, args.cY.invert(), Si);
    valuef epsilon = calculate_epsilon(p_star, e_star, args.W, w);
    valuef h = calculate_h_from_epsilon(epsilon);

    valuef cst = p_star * as_constant(bound) * h;

    v3f clamped = clamp(Si, -cst, cst);

    as_ref(hydro.Si[0][pos, dim]) = clamped[0];
    as_ref(hydro.Si[1][pos, dim]) = clamped[1];
    as_ref(hydro.Si[2][pos, dim]) = clamped[2];
}

hydrodynamic_plugin::hydrodynamic_plugin(cl::context ctx, float _linear_viscosity_timescale, bool _use_colour)
{
    linear_viscosity_timescale = _linear_viscosity_timescale;
    use_colour = _use_colour;

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(init_hydro, "init_hydro", use_colour);
    }, {"init_hydro"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(calculate_p_kern, "calculate_p");
    }, {"calculate_p"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(calculate_w_kern, "calculate_w");
    }, {"calculate_w"});

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(evolve_hydro_all, "evolve_hydro_all", use_colour);
    }, {"evolve_hydro_all"});

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(finalise_hydro, "finalise_hydro", use_colour);
    }, {"finalise_hydro"});
}

buffer_provider* hydrodynamic_plugin::get_buffer_factory(cl::context ctx)
{
    return new hydrodynamic_buffers(ctx, use_colour);
}

buffer_provider* hydrodynamic_plugin::get_utility_buffer_factory(cl::context ctx)
{
    return new hydrodynamic_utility_buffers(ctx);
}

void hydrodynamic_plugin::init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, cl::buffer u_buf, buffer_provider* to_init, buffer_provider* to_init_utility)
{
    neutron_star::all_numerical_eos_gpu neos(ctx);
    neos.init(cqueue, pack.stored_eos);

    std::vector<t3f> lin_cols;

    for(auto& i : pack.ns_colours)
        lin_cols.push_back(i.value_or((t3f){1,1,1}));

    cl::buffer lin_buf(ctx);
    lin_buf.alloc(sizeof(cl_float3) * lin_cols.size());
    lin_buf.write(cqueue, lin_cols);

    assert(lin_cols.size() == pack.stored_eos.size());

    hydrodynamic_buffers& bufs = *dynamic_cast<hydrodynamic_buffers*>(to_init);
    hydrodynamic_utility_buffers& ubufs = *dynamic_cast<hydrodynamic_utility_buffers*>(to_init_utility);

    {
        t3i dim = pack.dim;

        cl::args args;
        in.append_to(args);

        auto cl_in = bufs.get_buffers();

        for(auto& i : cl_in)
            args.push_back(i);

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
        args.push_back(lin_buf);

        cqueue.exec("init_hydro", args, {dim.x(), dim.y(), dim.z()}, {8,8,1});
    }
}

void hydrodynamic_plugin::step(cl::context ctx, cl::command_queue cqueue, const plugin_step_data& sdata)
{
    float damping_timescale = linear_viscosity_timescale;

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

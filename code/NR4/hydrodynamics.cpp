#include "hydrodynamics.hpp"
#include "init_general.hpp"

constexpr float min_p_star = 1e-7f;
constexpr float min_evolve_p_star = 1e-7f;

template<typename T>
inline
auto safe_divide(const auto& top, const T& bottom, float tol = 1e-7f)
{
    valuef bsign = ternary(bottom >= 0, valuef(1.f), valuef(-1.f));

    return ternary(fabs(bottom) <= tol, top / (bsign * tol), top / bottom);
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
//this function is numerically unstable
//todo: try setting this to zero where appropriate
v3f calculate_vi(valuef gA, v3f gB, valuef W, valuef w, valuef epsilon, v3f Si, const unit_metric<valuef, 3, 3>& cY, valuef p_star, bool viscosity)
{
    valuef h = calculate_h_from_epsilon(epsilon);

    v3f Si_upper = cY.invert().raise(Si);

    float bound = viscosity ? 1e-7f : 1e-7f;

    //note to self, actually hand derived this and am sure its correct
    v3f real_value = -gB + (W*W * gA / h) * safe_divide(Si_upper, w, bound);

    //return real_value;

    //returning -gB seems more proper to me as that's the limit as p* -> 0, but the paper specifically says to set vi = 0
    //return ternary(p_star <= min_evolve_p_star, -gB, real_value);
    return ternary(p_star <= min_evolve_p_star, {}, real_value);
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

valuef calculate_Pvis(valuef W, v3f vi, valuef p_star, valuef e_star, valuef w, const derivative_data& d,
                      valuef total_elapsed, valuef linear_damping_timescale,
                      float linear_strength, float quadratic_strength)
{
    valuef e_m6phi = pow(W, 3.f);

    valuef dkvk = 0;

    for(int k=0; k < 3; k++)
    {
        dkvk += 2 * diff1(vi[k], k, d);
    }

    valuef littledv = dkvk * d.scale;
    valuef Gamma = get_Gamma();

    valuef A = pow(e_star, Gamma) * pow(e_m6phi, Gamma - 1) * safe_divide(pow(p_star, Gamma - 1), pow(w, Gamma - 1));

    ///[0.1, 1.0]
    valuef CQvis = quadratic_strength;

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
    valuef CLvis = linear_strength * linear_damping;
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
    valuef Q;

    template<typename T>
    hydrodynamic_concrete(v3i pos, v3i dim, full_hydrodynamic_args<T> args)
    {
        p_star = max(args.p_star[pos, dim], 0.f);
        e_star = max(args.e_star[pos, dim], 0.f);
        Si = {args.Si[0][pos, dim], args.Si[1][pos, dim], args.Si[2][pos, dim]};
        w = args.w[pos, dim];
        Q = args.Q[pos, dim];
    }

    hydrodynamic_concrete(v3i pos, v3i dim, hydrodynamic_base_args<buffer<valuef>> bargs, hydrodynamic_utility_args<buffer<valuef>> uargs)
    {
        p_star = max(bargs.p_star[pos, dim], 0.f);
        e_star = max(bargs.e_star[pos, dim], 0.f);
        Si = {bargs.Si[0][pos, dim], bargs.Si[1][pos, dim], bargs.Si[2][pos, dim]};
        w = uargs.w[pos, dim];
        Q = uargs.Q[pos, dim];
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

    v3f calculate_vi(valuef gA, v3f gB, valuef W, const unit_metric<valuef, 3, 3>& cY, bool viscosity)
    {
        valuef epsilon = calculate_epsilon(W);

        return ::calculate_vi(gA, gB, W, w, epsilon, Si, cY, p_star, viscosity);
    }

    v3f calculate_ui(valuef gA, v3f gB, valuef W, const unit_metric<valuef, 3, 3>& cY)
    {
        valuef epsilon = calculate_epsilon(W);

        return ::calculate_ui(p_star, epsilon, Si, w, gA, gB, cY);
    }

    ///rhs here to specifically indicate that we're returning -(di Vec v^i), ie the negative
    valuef advect_rhs(valuef in, v3f vi, const derivative_data& d, valuef timestep)
    {
        using namespace single_source;

        #ifdef NAIVE_DISCRETISATION
        auto leib = [&](valuef v1, valuef v2, int i)
        {
            return diff1(v1 * v2, i, d);
            //leibnitzing it out introduces more error
            //return diff1(v1, i, d) * v2 + diff1(v2, i, d) * v1;
        };

        valuef sum = 0;

        for(int i=0; i < 3; i++)
        {
            sum += leib(in, vi[i], i);
        }

        return -sum;
        #endif

        #define VAN_LEER
        #ifdef VAN_LEER
        ///https://www.ita.uni-heidelberg.de/~dullemond/lectures/num_fluid_2012/Chapter_4.pdf 4.38
        auto get_delta = [&](valuef q, int which)
        {
            std::array<valuef, 3> v_adj = get_differentiation_variables<3, valuef>(vi[which], which);
            std::array<valuef, 7> q_adj = get_differentiation_variables<7, valuef>(in, which);
            std::array<valuef, 3> p_adj = get_differentiation_variables<3, valuef>(p_star, which);

            valuef v_phalf = safe_divide(p_adj[1] * v_adj[1] + p_adj[2] * v_adj[2], p_adj[1] + p_adj[2]);
            valuef v_mhalf = safe_divide(p_adj[1] * v_adj[1] + p_adj[0] * v_adj[0], p_adj[1] + p_adj[0]);

            valuef theta_mhalf = ternary(v_mhalf >= 0, valuef(1), valuef(-1));
            valuef theta_phalf = ternary(v_phalf >= 0, valuef(1), valuef(-1));

            valuef qm2 = q_adj.at(3 - 2);
            valuef qm1 = q_adj.at(3 - 1);
            valuef q0 = q_adj.at(3);
            valuef q1 = q_adj.at(3 + 1);
            valuef q2 = q_adj.at(3 + 2);

            valuef r_mhalf = ternary(v_mhalf >= 0, safe_divide(qm1 - qm2, q0 - qm1), safe_divide(q1 - q0, q0 - qm1));
            valuef r_phalf = ternary(v_phalf >= 0, safe_divide(q0 - qm1, q1 - q0), safe_divide(q2 - q1, q1 - q0));

            //superbee
            auto phi_r = [&](valuef r)
            {
                auto max3 = [&](valuef v1, valuef v2, valuef v3)
                {
                    return max(max(v1, v2), v3);
                };

                auto min3 = [&](valuef v1, valuef v2, valuef v3)
                {
                    return min(min(v1, v2), v3);
                };

                return max3(0.f, min(valuef(1.f), 2 * r), min(valuef(2), r));
                //return max(valuef(0.f), min3((1 + r)/2, 2, 2 * r));
            };

            valuef phi_mhalf = phi_r(r_mhalf);
            valuef phi_phalf = phi_r(r_phalf);

            valuef f_mhalf_1 = 0.5f * v_mhalf * ((1 + theta_mhalf) * qm1 + (1 - theta_mhalf) * q0);
            valuef f_phalf_1 = 0.5f * v_phalf * ((1 + theta_phalf) * q0 + (1 - theta_phalf) * q1);

            //uses the average flux approximation (?)
            //ideally, I wouldn't use that, the sole blocker is the lack of a good equation source
            valuef f_mhalf_2 = (1.f/2.f) * fabs(v_mhalf) * (1 - fabs(v_mhalf * timestep / d.scale)) * phi_mhalf * (q0 - qm1);
            valuef f_phalf_2 = (1.f/2.f) * fabs(v_phalf) * (1 - fabs(v_phalf * timestep / d.scale)) * phi_phalf * (q1 - q0);

            valuef f_mhalf = f_mhalf_1 + f_mhalf_2;
            valuef f_phalf = f_phalf_1 + f_phalf_2;

            return f_mhalf - f_phalf;
        };

        return (get_delta(in, 0) + get_delta(in, 1) + get_delta(in, 2)) / d.scale;
        #endif
    }

    v3f advect_rhs(v3f in, v3f vi, const derivative_data& d, valuef timestep)
    {
        v3f ret;

        for(int i=0; i < 3;  i++)
            ret[i] = advect_rhs(in[i], vi, d, timestep);

        return ret;
    }

    valuef calculate_Pvis(valuef W, v3f vi, const derivative_data& d, valuef total_elapsed, valuef damping_timescale, float linear_strength, float quadratic_strength)
    {
        return ::calculate_Pvis(W, vi, p_star, e_star, w, d, total_elapsed, damping_timescale, linear_strength, quadratic_strength);
    }

    valuef e_star_rhs(valuef W, valuef Q_vis, v3f vi, const derivative_data& d)
    {
        valuef e_6phi = pow(max(W, 0.1f), -3.f);

        valuef sum_interior_rhs = 0;

        for(int k=0; k < 3; k++)
        {
            value to_diff = safe_divide(w, p_star) * vi[k] * e_6phi;

            sum_interior_rhs += diff1(to_diff, k, d);
        }

        valuef Gamma = get_Gamma();

        valuef p0e = calculate_p0e(W);

        valuef degenerate = safe_divide(valuef{1}, pow(p0e, 1 - 1/Gamma));

        return -degenerate * (Q_vis / Gamma) * sum_interior_rhs;
    }

    v3f Si_rhs(valuef gA, v3f gB, valuef W, const unit_metric<valuef, 3, 3>& cY, valuef Q_vis, v3f vi, const derivative_data& d)
    {
        valuef P = max(eos(W) + Q, 0.f);

        valuef h = calculate_h_with_eos(W);

        v3f dSi;

        for(int k=0; k < 3; k++)
        {
            valuef p1 = (-gA * pow(max(W, 0.1f), -3.f)) * diff1(P, k, d);
            valuef p2 = -w * h * diff1(gA, k, d);

            valuef p3;

            for(int j=0; j < 3; j++)
            {
                p3 += -Si[j] * diff1(gB[j], k, d) ;
            }

            valuef p4;

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    valuef deriv = diff1(cY.invert()[i,j], k, d);

                    valuef l1 = Si[i] / h;
                    valuef l2 = safe_divide(Si[j], w);

                    p4 += 0.5f * gA * W * W * l1 * l2 * deriv;
                }
            }

            valuef p5 = gA * h * (w - p_star * safe_divide(p_star, w)) * (diff1(W, k, d) / max(W, 0.1f));

            dSi[k] = p1 + p2 + p3 + p4 + p5;
        }

        return dSi;
    }
};

template<typename T>
valuef full_hydrodynamic_args<T>::get_density(bssn_args& args, const derivative_data& d)
{
    hydrodynamic_concrete hydro_args(d.pos, d.dim, *this);

    return ternary(hydro_args.p_star >= min_p_star, hydro_args.calculate_p0(args.W), valuef(0.f));
}

template<typename T>
valuef full_hydrodynamic_args<T>::get_energy(bssn_args& args, const derivative_data& d)
{
    hydrodynamic_concrete hydro_args(d.pos, d.dim, *this);

    return ternary(hydro_args.p_star >= min_p_star, hydro_args.calculate_epsilon(args.W), valuef(0.f));
}

template<typename T>
v4f full_hydrodynamic_args<T>::get_4_velocity(bssn_args& args, const derivative_data& d)
{
    hydrodynamic_concrete hydro_args(d.pos, d.dim, *this);

    v3f ui = hydro_args.calculate_ui(args.gA, args.gB, args.W, args.cY);

    valuef u0 = safe_divide(hydro_args.w, hydro_args.p_star * args.gA);

    v4f velocity = {u0, ui.x(), ui.y(), ui.z()};

    return ternary(hydro_args.p_star >= min_p_star, velocity, (v4f){1,0,0,0});
}

template<typename T>
v3f full_hydrodynamic_args<T>::get_colour(bssn_args& args, const derivative_data& d)
{
    v3i pos = d.pos;
    v3i dim = d.dim;

    v3f raw_colour = {this->colour[0][pos, dim], this->colour[1][pos, dim], this->colour[2][pos, dim]};

    return raw_colour / max(this->p_star[pos, dim], 1e-7f);
}

template<typename T>
valuef full_hydrodynamic_args<T>::adm_p(bssn_args& args, const derivative_data& d)
{
    hydrodynamic_concrete hydro_args(d.pos, d.dim, *this);

    valuef h = hydro_args.calculate_h_with_eos(args.W);

    return ternary(hydro_args.p_star <= min_evolve_p_star, {}, hydro_args.w * h * pow(args.W, 3.f) - hydro_args.eos(args.W));
}

template<typename T>
tensor<valuef, 3> full_hydrodynamic_args<T>::adm_Si(bssn_args& args, const derivative_data& d)
{
    v3f cSi = {this->Si[0][d.pos, d.dim], this->Si[1][d.pos, d.dim], this->Si[2][d.pos, d.dim]};

    valuef p_star = this->p_star[d.pos, d.dim];

    return ternary(p_star <= min_evolve_p_star, {}, pow(args.W, 3.f) * cSi);
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
            W2_Sij[i, j] = ((pow(args.W, 5.f) / h) * safe_divide(hydro_args.Si[i], hydro_args.w)) * hydro_args.Si[j];
        }
    }

    return ternary(hydro_args.p_star <= min_evolve_p_star, {}, W2_Sij + hydro_args.eos(args.W) * args.cY.to_tensor());
}

template<typename T>
valuef full_hydrodynamic_args<T>::dbg(bssn_args& args, const derivative_data& d)
{
    return ternary(this->p_star[d.pos, d.dim] <= min_evolve_p_star, valuef(0), valuef(1));

    //return fabs(this->p_star[d.pos, d.dim]) * 500;
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
    p.dissipation_coeff = 0;
    p.dissipation_order = 4;

    buffer_descriptor e;
    e.name = "e*";
    e.dissipation_coeff = 0;
    e.dissipation_order = 4;

    buffer_descriptor s0;
    s0.name = "cs0";
    s0.dissipation_coeff = 0;
    s0.dissipation_order = 4;

    buffer_descriptor s1;
    s1.name = "cs1";
    s1.dissipation_coeff = 0;
    s1.dissipation_order = 4;

    buffer_descriptor s2;
    s2.name = "cs2";
    s2.dissipation_coeff = 0;
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
    buffer_descriptor Q;
    Q.name = "Q";

    buffer_descriptor w;
    w.name = "w";

    return {w, Q};
}

std::vector<cl::buffer> hydrodynamic_utility_buffers::get_buffers()
{
    return {w, Q};
}

void hydrodynamic_utility_buffers::allocate(cl::context ctx, cl::command_queue cqueue, t3i size)
{
    int64_t cells = int64_t{size.x()} * size.y() * size.z();

    w.alloc(sizeof(cl_float) * cells);
    Q.alloc(sizeof(cl_float) * cells);

    w.set_to_zero(cqueue);
    Q.set_to_zero(cqueue);

    for(int i=0; i < 8; i++)
        intermediate.emplace_back(ctx);

    ///p* e* Si0 Si1 Si2
    for(int i=0; i < 5; i++)
    {
        intermediate[i].alloc(sizeof(cl_float) * cells);
        intermediate[i].set_to_zero(cqueue);
    }

    if(use_colour)
    {
        ///r* g* b*
        for(int i=5; i < 8; i++)
        {
            intermediate[i].alloc(sizeof(cl_float) * cells);
            intermediate[i].set_to_zero(cqueue);
        }
    }

    dbg.alloc(sizeof(cl_long));
    dbg.set_to_zero(cqueue);
}

struct eos_gpu : value_impl::single_source::argument_pack
{
    buffer<valuef> pressures;
    buffer<valuef> max_densities;

    buffer<valuef> mu_to_p0;
    buffer<valuef> max_mus;

    literal<valuei> stride;
    literal<valuei> eos_count;

    void build(auto& in)
    {
        using namespace value_impl::builder;

        add(pressures, in);
        add(max_densities, in);

        add(mu_to_p0, in);
        add(max_mus, in);

        add(stride, in);
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

    valuef max_density = eos_data.max_densities[index];
    valuef max_mu = eos_data.max_mus[index];

    bssn_args args(pos, dim, in);

    auto pressure_to_p0 = [&](valuef P)
    {
        valuei offset = index * eos_data.stride.get();

        mut<valuei> i = declare_mut_e(valuei(0));
        mut<valuef> out = declare_mut_e(valuef(0));

        for_e(i < eos_data.stride.get() - 1, assign_b(i, i+1), [&]{
            valuef p1 = eos_data.pressures[offset + i];
            valuef p2 = eos_data.pressures[offset + i + 1];

            if_e(P >= p1 && P <= p2, [&]{
                valuef val = (P - p1) / (p2 - p1);

                as_ref(out) = (((valuef)i + val) / (valuef)eos_data.stride.get()) * max_density;

                break_e();
            });
        });

        if_e(i == eos_data.stride.get(), [&]{
            print("Error, overflowed pressure data\n");
        });

        return declare_e(out);
    };

    auto p0_to_pressure = [&](valuef p0)
    {
        valuei offset = index * eos_data.stride.get();

        valuef idx = clamp((p0 / max_density) * (valuef)eos_data.stride.get(), valuef(0), (valuef)eos_data.stride.get() - 2);

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

    valuef Gamma = get_Gamma();

    auto mu_to_p0 = [&](valuef mu)
    {
        valuei offset = index * eos_data.stride.get();

        valuef idx = clamp((mu / max_mu) * (valuef)eos_data.stride.get(), valuef(0), (valuef)eos_data.stride.get() - 2);

        valuei fidx = (valuei)idx;

        return mix(eos_data.mu_to_p0[offset + fidx], eos_data.mu_to_p0[offset + fidx + 1], idx - floor(idx));
    };

    auto mu_to_P = [&](valuef mu)
    {
        return p0_to_pressure(mu_to_p0(mu));
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

            valuef p_1 = mu_to_P(test_mu1);
            valuef p_2 = mu_to_P(test_mu2);

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

    valuef ysj = Yij.dot(Si, Si);
    pin(ysj);

    valuef u0 = 1;
    valuef mu = mu_h;

    for(int i=0; i < 100; i++)
    {
        //yijsisj = (mu + p)^2 W^2 (W^2 - 1)
        //(yijsisj) / (mu+p)^2 = W^4 - W^2
        //C = W^4 - W^2
        //W = sqrt(1 + sqrt(4C + 1)) / sqrt(2)
        valuef pressure = mu_to_P(mu);
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

    valuef pressure = mu_to_P(mu);
    valuef p0 = pressure_to_p0(pressure);

    valuef epsilon = (mu / p0) - 1;

    valuef gA = 1;
    v3f gB = {0,0,0};

    //fluid dynamics cannot have a singular initial slice
    //thing is we have 0 quantities at the singularity, so as long as you don't generate a literal NaN here, you're 100% fine

    valuef p_star = p0 * gA * u0 * pow(cW, -3);
    valuef e_star = pow(p0 * epsilon, (1/Gamma)) * gA * u0 * pow(cW, -3);

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

        u_i[s] = gB[s] * u0 + sum;
    }

    valuef h = calculate_h_from_epsilon(epsilon);

    v3f Si_lo_cfl = p_star * h * u_i;

    as_ref(hydro.p_star[pos, dim]) = p_star;
    as_ref(hydro.e_star[pos, dim]) = e_star;
    as_ref(hydro.Si[0][pos, dim]) = Si_lo_cfl[0];
    as_ref(hydro.Si[1][pos, dim]) = Si_lo_cfl[1];
    as_ref(hydro.Si[2][pos, dim]) = Si_lo_cfl[2];

    //strictly speaking i don't need to set these
    as_ref(hydro.w[pos, dim]) = w;
    as_ref(hydro.Q[pos, dim]) = valuef(0.f);

    if(use_colour)
    {
        for(int i=0; i < (int)hydro.colour.size(); i++)
            as_ref(hydro.colour[i][pos, dim]) = colour_in[index][i] * p_star;
    }
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

    valuef w = p_star;

    valuef p_sq = p_star * p_star;

    valuef cst = calculate_w_constant(W, icY, Si);

    //pin(p_sq);
    pin(cst);

    for(int i=0; i < 140; i++)
    {
        valuef D = w_next_interior(p_star, e_star, W, w);

        valuef w_next = sqrt(max(p_sq + cst * D*D, 0.f));

        pin(w_next);

        ///relaxation. Its not really necessary
        w = mix(w, w_next, valuef(0.9f));
        pin(w);
    }

    return w;
}

void calculate_w_kern(execution_context& ectx, bssn_args_mem<buffer<valuef>> in, hydrodynamic_base_args<buffer<valuef>> hydro, buffer_mut<valuef> w_out,
                      literal<v3i> idim, literal<valuef> scale,
                      literal<valuei> positions_length)
{
    using namespace single_source;

    valuei lid = value_impl::get_global_id(0);

    pin(lid);

    v3i dim = idim.get();

    if_e(lid >= positions_length.get(), []{
        return_e();
    });

    v3i pos = get_coordinate_including_boundary(lid, dim);
    pin(pos);

    valuef p_star = hydro.p_star[pos, dim];
    valuef e_star = hydro.e_star[pos, dim];
    v3f Si = {hydro.Si[0][pos, dim], hydro.Si[1][pos, dim], hydro.Si[2][pos, dim]};

    derivative_data d;
    d.pos = pos;
    d.dim = dim;
    d.scale = scale.get();

    if_e(p_star <= min_p_star, [&]{
        valuef dp_sum = 0;

        for(int i=0; i < 3; i++)
        {
            dp_sum += fabs(diff1(p_star, i, d));
        }

        if_e(dp_sum == 0, [&]{
            as_ref(w_out[pos, dim]) = valuef(0);
            return_e();
        });
    });

    bssn_args args(pos, dim, in);

    as_ref(w_out[pos, dim]) = calculate_w(p_star, e_star, args.W, args.cY.invert(), Si);
}

#define MIN_LAPSE 0.15f
#define MIN_VISCOSITY_LAPSE 0.4f

void calculate_Q_kern(execution_context& ectx, bssn_args_mem<buffer<valuef>> in, hydrodynamic_base_args<buffer<valuef>> hydro, buffer<valuef> w_in, buffer_mut<valuef> Q_out,
                      literal<v3i> idim, literal<valuef> scale, literal<valuef> total_elapsed, literal<valuef> damping_timescale,
                      literal<valuei> positions_length, float quadratic_strength, float linear_strength)
{
    using namespace single_source;

    valuei lid = value_impl::get_global_id(0);

    pin(lid);

    v3i dim = idim.get();

    if_e(lid >= positions_length.get(), []{
        return_e();
    });

    v3i pos = get_coordinate_including_boundary(lid, dim);
    pin(pos);

    valuef p_star = hydro.p_star[pos, dim];
    valuef e_star = hydro.e_star[pos, dim];
    v3f Si = {hydro.Si[0][pos, dim], hydro.Si[1][pos, dim], hydro.Si[2][pos, dim]};

    derivative_data d;
    d.pos = pos;
    d.dim = dim;
    d.scale = scale.get();

    if_e(p_star <= min_p_star, [&]{
        valuef dp_sum = 0;

        for(int i=0; i < 3; i++)
        {
            dp_sum += fabs(diff1(p_star, i, d));
        }

        if_e(dp_sum == 0, [&]{
            as_ref(Q_out[pos, dim]) = valuef(0);
            return_e();
        });
    });

    bssn_args args(pos, dim, in);

    valuef w = w_in[pos, dim];

    as_ref(Q_out[pos, dim]) = valuef(0.f);

    if_e(args.gA >= MIN_VISCOSITY_LAPSE, [&]{
        valuef epsilon = calculate_epsilon(p_star, e_star, args.W, w);
        v3f vi = calculate_vi(args.gA, args.gB, args.W, w, epsilon, Si, args.cY, p_star, true);

        valuef Q = calculate_Pvis(args.W, vi, p_star, e_star, w, d, total_elapsed.get(), damping_timescale.get(), linear_strength, quadratic_strength);

        as_ref(Q_out[pos, dim]) = Q;
    });
}

void evolve_hydro_all(execution_context& ectx, bssn_args_mem<buffer<valuef>> in,
                  hydrodynamic_base_args<buffer<valuef>> h_base, hydrodynamic_base_args<buffer<valuef>> h_in, hydrodynamic_base_args<buffer_mut<valuef>> h_out,
                  hydrodynamic_base_args<buffer_mut<valuef>> dt_inout,
                  hydrodynamic_utility_args<buffer<valuef>> util,
                  literal<v3i> idim, literal<valuef> scale, literal<valuef> timestep, literal<valuef> total_elapsed, literal<valuef> damping_timescale,
                  literal<valuei> positions_length,
                  literal<int> iteration, bool use_colour)
{
    using namespace single_source;

    valuei lid = value_impl::get_global_id(0);

    pin(lid);

    v3i dim = idim.get();

    if_e(lid >= positions_length.get(), []{
        return_e();
    });

    v3i pos = get_coordinate_including_boundary(lid, dim, 3);
    pin(pos);

    derivative_data d;
    d.pos = pos;
    d.dim = idim.get();
    d.scale = scale.get();

    valuei boundary_dist = distance_to_boundary(pos, dim);

    bssn_args args(pos, dim, in);
    hydrodynamic_concrete hydro_args(pos, dim, h_in, util);

    //crank nicolson
    auto write_result = [&](valuef dt_p_star, valuef dt_e_star, v3f dt_Si, v3f dt_col)
    {
        //predictor, ie euler
        if_e(iteration.get() == 0, [&]{
            valuef fin_p_star = h_base.p_star[pos, dim] + dt_p_star * timestep.get();
            valuef fin_e_star = h_base.e_star[pos, dim] + dt_e_star * timestep.get();

            fin_p_star = max(fin_p_star, 0.f);
            fin_e_star = max(fin_e_star, 0.f);

            as_ref(dt_inout.p_star[pos, dim]) = (fin_p_star - h_base.p_star[pos, dim]) / timestep.get();
            as_ref(dt_inout.e_star[pos, dim]) = (fin_e_star - h_base.e_star[pos, dim]) / timestep.get();

            for(int i=0; i < 3; i++)
                as_ref(dt_inout.Si[i][pos, dim]) = dt_Si[i];

            as_ref(h_out.p_star[pos, dim]) = fin_p_star;
            as_ref(h_out.e_star[pos, dim]) = fin_e_star;

            for(int i=0; i < 3; i++)
                as_ref(h_out.Si[i][pos, dim]) = h_base.Si[i][pos, dim] + dt_Si[i] * timestep.get();

            if(use_colour)
            {
                for(int i=0; i < 3; i++)
                {
                    valuef fin_colour = h_base.colour[i][pos, dim] + dt_col[i] * timestep.get();
                    fin_colour = max(fin_colour, 0.f);

                    as_ref(dt_inout.colour[i][pos, dim]) = (fin_colour - h_base.colour[i][pos, dim]) / timestep.get();

                    as_ref(h_out.colour[i][pos, dim]) = fin_colour;
                }
            }
        });

        //corrector, ie the next fixed point step for crank nicolson
        if_e(iteration.get() != 0, [&]{
            float relax = 0.f;

            valuef root_dp_star = declare_e(dt_inout.p_star[pos, dim]);
            valuef root_de_star = declare_e(dt_inout.e_star[pos, dim]);
            v3f root_dSi = declare_e(dt_inout.index_Si(pos, dim));

            v3f root_dcol;

            if(use_colour)
                root_dcol = declare_e(dt_inout.index_colour(pos, dim));

            //impl = 1 == backwards euler, impl = 0 == fowards euler. impl = 0.5 == crank nicolson/implicit midpoint
            float impl = 0.5;
            float expl = 1 - impl;

            auto apply = [&](auto x0, auto xi, auto f_x0, auto f_xi)
            {
                return relax * xi + (1 - relax) * (x0 + timestep.get() * (expl * f_x0 + impl * f_xi));
            };

            valuef fin_p_star = apply(h_base.p_star[pos, dim], h_in.p_star[pos, dim], root_dp_star, dt_p_star);
            valuef fin_e_star = apply(h_base.e_star[pos, dim], h_in.e_star[pos, dim], root_de_star, dt_e_star);
            v3f fin_Si = apply(h_base.index_Si(pos, dim), h_in.index_Si(pos, dim), root_dSi, dt_Si);

            fin_p_star = max(fin_p_star, 0.f);
            fin_e_star = max(fin_e_star, 0.f);

            as_ref(h_out.p_star[pos, dim]) = fin_p_star;
            as_ref(h_out.e_star[pos, dim]) = fin_e_star;

            for(int i=0; i < 3; i++)
                as_ref(h_out.Si[i][pos, dim]) = fin_Si[i];

            if(use_colour)
            {
                v3f fin_col = apply(h_base.index_colour(pos, dim), h_in.index_colour(pos, dim), root_dcol, dt_col);

                for(int i=0; i < 3; i++)
                    as_ref(h_out.colour[i][pos, dim]) = max(fin_col[i], 0.f);
            }
        });
    };

    //early terminate
    if_e(hydro_args.p_star <= min_p_star, [&]{
        valuef dp_sum = 0;

        for(int i=0; i < 3; i++)
        {
            dp_sum += fabs(diff1(hydro_args.p_star, i, d));
        }

        if_e(dp_sum == 0, [&]{
            write_result(0.f, 0.f, {}, {});

            return_e();
        });
    });

    //we're in a black hole, damp away the material
    if_e(args.gA < MIN_LAPSE, [&]{
        valuef damp = 0.1f;

        valuef dt_p_star = damp * (0 - h_in.p_star[pos, dim]);
        valuef dt_e_star = damp * (0 - h_in.e_star[pos, dim]);

        valuef dt_s0 = damp * (0 - h_in.Si[0][pos, dim]);
        valuef dt_s1 = damp * (0 - h_in.Si[1][pos, dim]);
        valuef dt_s2 = damp * (0 - h_in.Si[2][pos, dim]);

        valuef dt_col0 = damp * (0 - h_in.colour[0][pos, dim]);
        valuef dt_col1 = damp * (0 - h_in.colour[1][pos, dim]);
        valuef dt_col2 = damp * (0 - h_in.colour[2][pos, dim]);

        write_result(dt_p_star, dt_e_star, {dt_s0, dt_s1, dt_s2}, {dt_col0, dt_col1, dt_col2});

        return_e();
    });

    v3f vi = hydro_args.calculate_vi(args.gA, args.gB, args.W, args.cY, false);

    valuef dp_star = hydro_args.advect_rhs(hydro_args.p_star, vi, d, timestep.get());

    mut<valuef> de_star = declare_mut_e(hydro_args.advect_rhs(hydro_args.e_star, vi, d, timestep.get()));
    mut_v3f dSi = declare_mut_e(hydro_args.advect_rhs(hydro_args.Si, vi, d, timestep.get()));

    //only apply advection terms for matter which is ~0
    if_e(hydro_args.p_star >= min_evolve_p_star, [&]{
        valuef Q = hydro_args.Q;

        as_ref(dSi) += hydro_args.Si_rhs(args.gA, args.gB, args.W, args.cY, Q, vi, d);

        //adds the e* viscosity term. Unstable near or in a black hole
        if_e(args.gA >= MIN_VISCOSITY_LAPSE, [&]{
            //it can be helpful to calculate the velocity with slightly more restrictive tolerances
            //because the viscosity is a more unstable term
            //i don't use this property currently, but if you're finding that more visosity leads to blowups
            //in the halo around a star, it can be useful to tweak the tol inside calculate_vi for visco = true
            v3f vi2 = hydro_args.calculate_vi(args.gA, args.gB, args.W, args.cY, true);

            as_ref(de_star) += hydro_args.e_star_rhs(args.W, Q, vi2, d);
        });
    });

    valuef boundary_damp = 0.25f;

    dp_star += ternary(boundary_dist <= 15, -hydro_args.p_star * boundary_damp, {});
    de_star += ternary(boundary_dist <= 15, -hydro_args.e_star * boundary_damp, {});
    as_ref(dSi) += ternary(boundary_dist <= 15, -hydro_args.Si * boundary_damp, {});

    v3f dt_col;

    if(use_colour)
    {
        v3f col = h_in.index_colour(pos, dim);

        dt_col = hydro_args.advect_rhs(col, vi, d, timestep.get());
        dt_col += ternary(boundary_dist <= 15, -col * boundary_damp, {});
    }

    write_result(dp_star, de_star, as_constant(dSi), dt_col);
}

void enforce_hydro_constraints(execution_context& ectx, bssn_args_mem<buffer<valuef>> in,
                    hydrodynamic_base_args<buffer_mut<valuef>> hydro,
                    literal<v3i> idim,
                    literal<valuei> positions_length, bool use_colour)
{
    using namespace single_source;

    valuei lid = value_impl::get_global_id(0);

    pin(lid);

    v3i dim = idim.get();

    if_e(lid >= positions_length.get(), []{
        return_e();
    });

    v3i pos = get_coordinate_including_boundary(lid, dim);
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

    //as per paper, clamp e* to 10 p*. I haven't found this to be a problem
    //#define CLAMP_E_STAR
    #ifdef CLAMP_E_STAR
    if_e(hydro.p_star[pos, dim] <= min_p_star * 10, [&]{
        valuef e_star = declare_e(hydro.e_star[pos, dim]);

        as_ref(hydro.e_star[pos, dim]) = min(e_star, 10 * hydro.p_star[pos, dim]);
    });
    #endif

    //clamps Si. In some formulations, I've found this to be very important for stability
    //currently it seems unimportant for the van-leer scheme, but the evolution of Si can be unstable
    //#define CLAMP_VELOCITY
    #ifdef CLAMP_VELOCITY
    //test bound
    mut<valuef> bound = declare_mut_e(valuef(0.9f));

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

    //#define CIRCULAR_CLAMP
    #ifdef CIRCULAR_CLAMP
    ///(ab)^2 + (ac)^2 + (ad^2)
    ///a^2 (b^2 + c^2 + d^2)
    ///sqrt = a * sqrt(b^2 + c^2 + d^2)`
    valuef length = sqrt(dot(Si, Si));

    valuef clamped_length = min(length, p_star * h * as_constant(bound));
    v3f clamped = Si * (clamped_length / length);

    #else
    valuef cst = p_star * as_constant(bound) * h;
    v3f clamped = clamp(Si, -cst, cst);
    #endif

    as_ref(hydro.Si[0][pos, dim]) = clamped[0];
    as_ref(hydro.Si[1][pos, dim]) = clamped[1];
    as_ref(hydro.Si[2][pos, dim]) = clamped[2];
    #endif
}

void sum_rest_mass(execution_context& ectx, bssn_args_mem<buffer<valuef>> in,
                    hydrodynamic_base_args<buffer<valuef>> hydro,
                    hydrodynamic_utility_args<buffer<valuef>> util,
                    literal<v3i> idim,
                    literal<valuei> positions_length,
                    literal<valuef> scale, buffer_mut<value<std::int64_t>> sum)
{
    using namespace single_source;

    valuei lid = value_impl::get_global_id(0);

    pin(lid);

    v3i dim = idim.get();

    if_e(lid >= positions_length.get(), []{
        return_e();
    });

    v3i pos = get_coordinate_including_boundary(lid, dim);
    pin(pos);

    bssn_args args(pos, dim, in);
    hydrodynamic_concrete hydro_args(pos, dim, hydro, util);

    //surprise! p* is the conserved rest mass
    valuef m0 = hydro_args.p_star;

    valued as_double = (valued)m0 * pow(10., 12.) * (valued)pow(scale.get(), 3.f);

    value<std::int64_t> as_uint = (value<std::int64_t>)as_double;

    sum.atom_add_e(0, as_uint);
}

hydrodynamic_plugin::hydrodynamic_plugin(cl::context ctx, float _linear_viscosity_timescale, bool _use_colour, float _linear_viscosity_strength, float _quadratic_viscosity_strength)
{
    linear_viscosity_timescale = _linear_viscosity_timescale;
    use_colour = _use_colour;
    linear_viscosity_strength = _linear_viscosity_strength;
    quadratic_viscosity_strength = _quadratic_viscosity_strength;

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(init_hydro, "init_hydro", use_colour);
    }, {"init_hydro"});

    cl::async_build_and_cache(ctx, [this]{
        return value_impl::make_function(calculate_Q_kern, "calculate_Q", linear_viscosity_strength, quadratic_viscosity_strength);
    }, {"calculate_Q"});

    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(calculate_w_kern, "calculate_w");
    }, {"calculate_w"});

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(evolve_hydro_all, "evolve_hydro_all", use_colour);
    }, {"evolve_hydro_all"});

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(enforce_hydro_constraints, "enforce_hydro_constraints", use_colour);
    }, {"enforce_hydro_constraints"});

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(sum_rest_mass, "sum_rest_mass");
    }, {"sum_rest_mass"});
}

buffer_provider* hydrodynamic_plugin::get_buffer_factory(cl::context ctx)
{
    return new hydrodynamic_buffers(ctx, use_colour);
}

buffer_provider* hydrodynamic_plugin::get_utility_buffer_factory(cl::context ctx)
{
    return new hydrodynamic_utility_buffers(ctx, use_colour);
}

void hydrodynamic_plugin::init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, cl::buffer u_buf, buffer_provider* to_init, buffer_provider* to_init_utility)
{
    neutron_star::all_numerical_eos_gpu neos(ctx);
    neos.init(cqueue, pack.stored_eos);

    std::vector<t3f> lin_cols;

    for(auto& i : pack.ns_colours)
        lin_cols.push_back(i.value_or((t3f){1,1,1}));

    //buffer for storing our stars colours in
    cl::buffer lin_buf(ctx);
    lin_buf.alloc(sizeof(cl_float3) * lin_cols.size());
    lin_buf.write(cqueue, lin_cols);

    assert(lin_cols.size() == pack.stored_eos.size());

    hydrodynamic_buffers& bufs = *dynamic_cast<hydrodynamic_buffers*>(to_init);
    hydrodynamic_utility_buffers& ubufs = *dynamic_cast<hydrodynamic_utility_buffers*>(to_init_utility);

    //initialises the hydrodynamics
    {
        t3i dim = pack.dim;

        cl::args args;
        in.append_to(args);

        auto cl_in = bufs.get_buffers();

        for(auto& i : cl_in)
            args.push_back(i);

        args.push_back(ubufs.w);
        args.push_back(ubufs.Q);
        args.push_back(pack.dim);
        args.push_back(pack.scale);
        args.push_back(pack.disc.mu_h_cfl);
        args.push_back(pack.disc.cfl);
        args.push_back(u_buf);
        args.push_back(pack.disc.Si_cfl[0]);
        args.push_back(pack.disc.Si_cfl[1]);
        args.push_back(pack.disc.Si_cfl[2]);
        args.push_back(pack.disc.star_indices);
        args.push_back(neos.pressures, neos.max_densities, neos.mu_to_p0, neos.max_mus, neos.stride, neos.count);
        args.push_back(lin_buf);

        cqueue.exec("init_hydro", args, {dim.x(), dim.y(), dim.z()}, {8,8,1});
    }
}

int get_evolve_size_with_boundary(t3i dim, int boundary);

void hydrodynamic_plugin::step(cl::context ctx, cl::command_queue cqueue, const plugin_step_data& sdata)
{
    float damping_timescale = linear_viscosity_timescale;

    hydrodynamic_buffers& bufs_base = *dynamic_cast<hydrodynamic_buffers*>(sdata.buffers[sdata.base_idx]);
    hydrodynamic_buffers& bufs_in = *dynamic_cast<hydrodynamic_buffers*>(sdata.buffers[sdata.in_idx]);
    hydrodynamic_buffers& bufs_out = *dynamic_cast<hydrodynamic_buffers*>(sdata.buffers[sdata.out_idx]);
    hydrodynamic_utility_buffers& ubufs = *dynamic_cast<hydrodynamic_utility_buffers*>(sdata.utility_buffers);

    auto utility_buffers = ubufs.get_buffers();

    //calculate W, and Q. Q requires the derivative of W, and evolving hydro requires the derivative of Q
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
            args.push_back(ubufs.Q);

            args.push_back(sdata.dim);
            args.push_back(sdata.scale);
            args.push_back(sdata.total_elapsed);
            args.push_back(damping_timescale);
            args.push_back(sdata.evolve_length);

            cqueue.exec("calculate_Q", args, {sdata.evolve_length}, {128});
        }
    };

    std::vector<cl::buffer> cl_base = bufs_base.get_buffers();

    int elen = get_evolve_size_with_boundary(sdata.dim, 3);

    //evolve the hydrodynamics
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

        for(auto& i : ubufs.intermediate)
            args.push_back(i);

        for(auto& i : utility_buffers)
            args.push_back(i);

        args.push_back(sdata.dim);
        args.push_back(sdata.scale);
        args.push_back(sdata.timestep);
        args.push_back(sdata.total_elapsed);
        args.push_back(damping_timescale);
        args.push_back(elen);
        args.push_back(sdata.iteration);

        cqueue.exec("evolve_hydro_all", args, {elen}, {128});
    }

    /*{
        cl::args args;

        for(auto& i : sdata.bssn_buffers)
            args.push_back(i);

        for(auto& i : bufs_out.get_buffers())
            args.push_back(i);

        args.push_back(sdata.dim);
        args.push_back(sdata.evolve_length);

        cqueue.exec("enforce_hydro_constraints", args, {sdata.evolve_length}, {128});
    }*/
}

void hydrodynamic_plugin::finalise(cl::context ctx, cl::command_queue cqueue, const finalise_data& sdata)
{
    hydrodynamic_buffers& bufs = *dynamic_cast<hydrodynamic_buffers*>(sdata.inout);
    auto all = bufs.get_buffers();

    //enforce the constraints once at the end of the simulation step
    {
        cl::args args;

        for(auto& i : sdata.bssn_buffers)
            args.push_back(i);

        for(auto& i : all)
            args.push_back(i);

        args.push_back(sdata.dim);
        args.push_back(sdata.evolve_length);

        cqueue.exec("enforce_hydro_constraints", args, {sdata.evolve_length}, {128});
    }

    #define DEBUG_REST_MASS
    #ifdef DEBUG_REST_MASS
    hydrodynamic_utility_buffers& ubufs = *dynamic_cast<hydrodynamic_utility_buffers*>(sdata.utility_buffers);

    {
        cl::args args;

        for(auto& i : sdata.bssn_buffers)
            args.push_back(i);

        for(auto i : bufs.get_buffers())
            args.push_back(i);

        args.push_back(ubufs.w);

        args.push_back(sdata.dim);
        args.push_back(sdata.scale);
        args.push_back(sdata.evolve_length);

        cqueue.exec("calculate_w", args, {sdata.evolve_length}, {128});
    }

    {
        cl::args args;

        for(auto& i : sdata.bssn_buffers)
            args.push_back(i);

        for(auto i : bufs.get_buffers())
            args.push_back(i);

        args.push_back(ubufs.w);
        args.push_back(ubufs.Q);

        args.push_back(sdata.dim);
        args.push_back(sdata.scale);
        args.push_back(sdata.total_elapsed);
        args.push_back(linear_viscosity_timescale);
        args.push_back(sdata.evolve_length);

        cqueue.exec("calculate_Q", args, {sdata.evolve_length}, {128});
    }

    {
        ubufs.dbg.set_to_zero(cqueue);

        cl::args args;

        for(auto& i : sdata.bssn_buffers)
            args.push_back(i);

        for(auto& i : all)
            args.push_back(i);

        args.push_back(ubufs.w);
        args.push_back(ubufs.Q);

        args.push_back(sdata.dim);
        args.push_back(sdata.evolve_length);
        args.push_back(sdata.scale);
        args.push_back(ubufs.dbg);

        cqueue.exec("sum_rest_mass", args, {sdata.evolve_length}, {128});

        cl_long out = ubufs.dbg.read<cl_long>(cqueue).at(0);

        debug_rest_mass.push_back(out / pow(10., 12.));
    }
    #endif // DEBUG_REST_MASS
}

void hydrodynamic_plugin::add_args_provider(all_adm_args_mem& mem)
{
    mem.add(full_hydrodynamic_args<buffer<valuef>>());
}

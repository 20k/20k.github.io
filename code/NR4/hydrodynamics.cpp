#include "hydrodynamics.hpp"
#include "init_general.hpp"

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

    return pow(e_m6phi / max(w, 0.001f), Gamma-1) * pow(e_star, Gamma) * pow(p_star, Gamma - 2);
}

valuef calculate_p0e(valuef Gamma, valuef W, valuef w, valuef p_star, valuef e_star)
{
    valuef iv_au0 = p_star / max(w, 0.001f);

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
    valuef lP = P[d.pos, d.dim];
    valuef es = e_star[d.pos, d.dim];
    valuef ps = p_star[d.pos, d.dim];

    valuef epsilon = e_star_to_epsilon(ps, es, args.W, lw);

    valuef h = get_h_with_gamma_eos(epsilon);

    return (h * lw * (args.W * args.W * args.W) - lP);
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
    valuef es = e_star[d.pos, d.dim];
    v3f cSi = {Si[0][d.pos, d.dim], Si[1][d.pos, d.dim], Si[2][d.pos, d.dim]};
    valuef lw = w[d.pos, d.dim];
    valuef lP = P[d.pos, d.dim];

    valuef h = get_h_with_gamma_eos(es);

    tensor<valuef, 3, 3> W2_Sij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            W2_Sij[i, j] = (pow(args.W, 5.f) / max(lw * h, 0.001f)) * cSi[i] * cSi[j];
        }
    }

    return (W2_Sij + lP * args.cY.to_tensor());
}

template struct full_hydrodynamic_args<buffer<valuef>>;
template struct full_hydrodynamic_args<buffer_mut<valuef>>;

std::vector<buffer_descriptor> hydrodynamic_buffers::get_description()
{
    buffer_descriptor p;
    p.name = "p*";

    buffer_descriptor e;
    e.name = "e*";

    buffer_descriptor s0;
    s0.name = "cs0";

    buffer_descriptor s1;
    s1.name = "cs1";

    buffer_descriptor s2;
    s2.name = "cs2";

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
    vi0.name = "vi1";
    buffer_descriptor vi2;
    vi0.name = "vi2";

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

void init_hydro(execution_context& ectx, bssn_args_mem<buffer<valuef>> in, full_hydrodynamic_args<buffer_mut<valuef>> hydro, literal<v3i> ldim, literal<valuef> scale,
                buffer<valuef> mu_cfl_b, buffer<valuef> mu_h_cfl_b, buffer<valuef> pressure_cfl_b, buffer<valuef> cfl_b, std::array<buffer<valuef>, 3> Si_cfl_b,
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

                as_ref(out) = (((valuef)i + val) / (valuef)eos_data.pressure_stride.get()) * (valuef)eos_data.densities[index];

                break_e();
            });
        });

        return declare_e(out);
    };

    valuef mu_cfl = mu_cfl_b[pos, dim];
    valuef mu_h_cfl = mu_h_cfl_b[pos, dim];
    valuef pressure_cfl = pressure_cfl_b[pos, dim];
    valuef phi = cfl_b[pos, dim];
    v3f Si_cfl = {Si_cfl_b[0][pos, dim], Si_cfl_b[1][pos, dim], Si_cfl_b[2][pos, dim]};

    valuef mu = mu_cfl * pow(phi, -8);
    valuef mu_h = mu_h_cfl * pow(phi, -8);
    valuef pressure = pressure_cfl * pow(phi, -8);
    v3f Si = Si_cfl * pow(phi, -10);

    valuef u0 = sqrt((mu_h + pressure) / max(mu + pressure, 0.001f));

    valuef Gamma = get_Gamma();

    valuef p0_e = pressure / (Gamma - 1);

    valuef p0 = pressure_to_p0(pressure);


    /*
    if_e(pos.x() == dim.x()/2 + 20 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        value_base se;
        se.type = value_impl::op::SIDE_EFFECT;
        se.abstract_value = "printf(\"hi\\n\")";

        value_impl::get_context().add(se);
    });*/

    value gA = args.gA;

    //fluid dynamics cannot have a singular initial slice, so setting the clamping pretty high here because its irrelevant
    //thing is we have 0 quantities at the singularity, so as long as you don't generate a literal NaN here, you're 100% fine
    valuef cW = max(args.W, 0.01f);

    valuef p_star = p0 * gA * u0 * pow(cW, -3);
    valuef e_star = pow(p0_e, (1/Gamma)) * gA * u0 * pow(cW, -3);

    metric<valuef, 3, 3> Yij = args.cY / (cW*cW);

    v3f Si_lo_cfl = pow(cW, -3) * Yij.lower(Si);

    as_ref(hydro.p_star[pos, dim]) = p_star;
    as_ref(hydro.e_star[pos, dim]) = e_star;
    as_ref(hydro.Si[0][pos, dim]) = Si_lo_cfl[0];
    as_ref(hydro.Si[1][pos, dim]) = Si_lo_cfl[1];
    as_ref(hydro.Si[2][pos, dim]) = Si_lo_cfl[2];

    as_ref(hydro.w[pos, dim]) = p_star * gA * u0;
    as_ref(hydro.P[pos, dim]) = (Gamma - 1) * p0_e;
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
        p_star = args.p_star[pos, dim];
        e_star = args.e_star[pos, dim];
        Si = {args.Si[0][pos, dim], args.Si[1][pos, dim], args.Si[2][pos, dim]};
        w = args.w[pos, dim];
        P = args.P[pos, dim];
    }

    template<typename T>
    hydrodynamic_concrete(v3i pos, v3i dim, hydrodynamic_base_args<T> bargs, hydrodynamic_utility_args<T> uargs)
    {
        p_star = bargs.p_star[pos, dim];
        e_star = bargs.e_star[pos, dim];
        Si = {bargs.Si[0][pos, dim], bargs.Si[1][pos, dim], bargs.Si[2][pos, dim]};
        w = uargs.w[pos, dim];
        P = uargs.P[pos, dim];
    }
};

valuef w_next_interior(valuef p_star, valuef e_star, valuef W, valuef w_prev, valuef Gamma)
{
    valuef A = pow(max(W, 0.0001f), 3.f * Gamma - 3.f);
    valuef wG = pow(w_prev, Gamma - 1);

    valuef D = wG / max(wG + A * Gamma * pow(e_star, Gamma) * pow(max(p_star, 0.0001f), Gamma - 2), 0.001f);

    return D;
}

valuef calculate_w(valuef p_star, valuef e_star, valuef W, valuef Gamma, inverse_metric<valuef, 3, 3> icY, v3f Si)
{
    using namespace single_source;

    valuef w = 1.f;

    valuef p_sq = p_star * p_star;
    valuef cst = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cst += icY[i, j] * Si[i] * Si[j];
        }
    }

    pin(p_sq);
    pin(cst);

    for(int i=0; i < 20; i++)
    {
        valuef D = w_next_interior(p_star, e_star, W, w, Gamma);

        valuef w_next = sqrt(max(p_sq + cst * D*D, 0.f));

        pin(w_next);

        w = w_next;
    }

    return w;
}

/*value h = calculate_h_with_gamma_eos(chi, W);

tensor<value, 3> ret = -gB;

for(int i=0; i < 3; i++)
{
    value sum = 0;

    for(int j=0; j < 3; j++)
    {
        sum += divide_with_limit(gA * icY.idx(i, j) * cS.idx(j) * chi, W * h, 0.f, DIVISION_TOL);
    }

    ret.idx(i) += sum;
}

return ret;*/

v3f calculate_vi(valuef gA, v3f gB, valuef W, valuef w, valuef epsilon, v3f Si, const unit_metric<valuef, 3, 3>& cY)
{
    valuef h = get_h_with_gamma_eos(epsilon);

    return -gB + ((W*W * gA) / max(w*h, 0.001f)) * cY.invert().raise(Si);
}

constexpr float min_p_star = 1e-6f;

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

    if_e(hydro_args.p_star < min_p_star, [&]{
        as_ref(hydro.P[pos, dim]) = valuef(0);
        as_ref(hydro.w[pos, dim]) = valuef(0);

        return_e();
    });

    bssn_args args(pos, dim, in);

    valuef w = calculate_w(hydro_args.p_star, hydro_args.e_star, args.W, get_Gamma(), args.cY.invert(), hydro_args.Si);
    valuef P = gamma_eos(get_Gamma(), args.W, w, hydro_args.p_star, hydro_args.e_star);

    /*if_e(pos.x() == dim.x()/2 + 20 && pos.y() == dim.y()/2 && pos.z() == dim.z()/2, [&]{
        value_base se;
        se.type = value_impl::op::SIDE_EFFECT;
        se.abstract_value = "printf(\"p %f pstar %f\\n\"," + value_to_string(P) + "," + value_to_string(hydro_args.p_star) + ")";

        value_impl::get_context().add(se);
    });*/

    as_ref(hydro.w[pos, dim]) = w;
    as_ref(hydro.P[pos, dim]) = P;
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

    /*as_ref(h_out.p_star[pos, dim]) = hydro_args.p_star;
    as_ref(h_out.e_star[pos, dim]) = hydro_args.e_star;

    for(int i=0; i < 3; i++)
        as_ref(h_out.Si[i][pos, dim]) = Si[i];

    return;*/

    valuef epsilon = e_star_to_epsilon(p_star, e_star, args.W, w);

    v3f vi = calculate_vi(args.gA, args.gB, args.W, w, epsilon, Si, args.cY);

    valuef dp_star = 0;
    valuef de_star = 0;
    v3f dSi_p1;

    for(int i=0; i < 3; i++)
    {
        dp_star += diff1(p_star * vi[i], i, d);
        de_star += diff1(e_star * vi[i], i, d);

        for(int k=0; k < 3; k++)
        {
            dSi_p1[k] += diff1(Si[k] * vi[i], i, d);
        }
    }

    dp_star = -dp_star;
    de_star = -de_star;
    dSi_p1 = -dSi_p1;

    valuef h = get_h_with_gamma_eos(epsilon);

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

        valuef p4_prefix = args.gA * args.W * args.W / max(2 * w * h, 0.0001f);

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                p4 += Si[i] * Si[j] * diff1(args.cY.invert()[i,j], k, d);
            }
        }

        p4 = p4_prefix * p4;

        valuef p5 = (args.gA * h * (w*w - p_star * p_star) / max(w, 0.001f)) * (diff1(args.W, k, d) / max(args.W, 0.001f));

        dSi_p1[k] += p1 + p2 + p3 + p4 + p5;
    }

    valuef fin_p_star = max(h_base.p_star[pos, dim] + timestep.get() * dp_star, 0.f);
    valuef fin_e_star = max(h_base.e_star[pos, dim] + timestep.get() * de_star, 0.f);

    valuef max_p = 1;

    fin_e_star = ternary(fin_p_star < (1e-5f * max_p), min(fin_e_star, 10 * fin_p_star), fin_e_star);

    v3f fin_Si;

    for(int i=0; i < 3; i++)
    {
        fin_Si[i] = h_base.Si[i][pos, dim] + timestep.get() * dSi_p1[i];
    }

    if_e(fin_p_star <= min_p_star, [&]{
        as_ref(h_out.p_star[pos, dim]) = valuef(0);
        as_ref(h_out.e_star[pos, dim]) = valuef(0);

        for(int i=0; i < 3; i++)
        {
            as_ref(h_out.Si[i][pos, dim]) = valuef(0);
        }

        //return_e();
    });

    if_e(fin_p_star > min_p_star, [&]{
        as_ref(h_out.p_star[pos, dim]) = fin_p_star;
        as_ref(h_out.e_star[pos, dim]) = fin_e_star;

        for(int i=0; i < 3; i++)
        {
            as_ref(h_out.Si[i][pos, dim]) = fin_Si[i];
        }
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
}

buffer_provider* hydrodynamic_plugin::get_buffer_factory(cl::context ctx)
{
    return new hydrodynamic_buffers(ctx);
}

buffer_provider* hydrodynamic_plugin::get_utility_buffer_factory(cl::context ctx)
{
    return new hydrodynamic_utility_buffers(ctx);
}

void hydrodynamic_plugin::init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, buffer_provider* to_init, buffer_provider* to_init_utility)
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
}

void hydrodynamic_plugin::add_args_provider(all_adm_args_mem& mem)
{
    mem.add(full_hydrodynamic_args<buffer<valuef>>());
}

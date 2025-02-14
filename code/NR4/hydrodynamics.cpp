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


template<typename T>
valuef hydrodynamic_args<T>::adm_p(bssn_args& args, const derivative_data& d)
{
    valuef lw = w[d.pos, d.dim];
    valuef lP = P[d.pos, d.dim];
    valuef es = e_star[d.pos, d.dim];

    valuef h = get_h_with_gamma_eos(es);

    return h * lw * (args.W * args.W * args.W) - lP;
}

template<typename T>
tensor<valuef, 3> hydrodynamic_args<T>::adm_Si(bssn_args& args, const derivative_data& d)
{
    v3f cSi = {Si[0][d.pos, d.dim], Si[1][d.pos, d.dim], Si[2][d.pos, d.dim]};

    return pow(args.W, 3.f) * cSi;
}

template<typename T>
tensor<valuef, 3, 3> hydrodynamic_args<T>::adm_W2_Sij(bssn_args& args, const derivative_data& d)
{
    valuef ps = p_star[d.pos, d.dim];
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

    return W2_Sij + lP * args.cY.to_tensor();
}

template struct hydrodynamic_args<buffer<valuef>>;
template struct hydrodynamic_args<buffer_mut<valuef>>;

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

void init_hydro(execution_context& ectx, bssn_args_mem<buffer<valuef>> in, hydrodynamic_args<buffer_mut<valuef>> hydro, literal<v3i> ldim, literal<valuef> scale,
                buffer<valuef> mu_cfl_b, buffer<valuef> mu_h_cfl_b, buffer<valuef> pressure_cfl_b, buffer<valuef> cfl_b, std::array<buffer<valuef>, 3> Si_cfl_b)
{
    using namespace single_source;

    valuei lid = value_impl::get_global_id(0);

    pin(lid);

    v3i dim = ldim.get();

    if_e(lid >= dim.x() * dim.y() * dim.z(), []{
        return_e();
    });

    ///if i was smart, i'd use the structure of the grid to do this directly
    v3i pos = get_coordinate(lid, dim);
    pin(pos);

    bssn_args args(pos, dim, in);

    valuef mu_cfl = mu_cfl_b[pos, dim];
    valuef mu_h_cfl = mu_h_cfl_b[pos, dim];
    valuef pressure_cfl = pressure_cfl_b[pos, dim];
    valuef phi = cfl_b[pos, dim];
    v3f Si_cfl = {Si_cfl_b[0][pos, dim], Si_cfl_b[1][pos, dim], Si_cfl_b[2][pos, dim]};

    valuef mu = mu_cfl * pow(phi, -8);
    valuef mu_h = mu_h_cfl * pow(phi, -8);
    valuef pressure = pressure_cfl * pow(phi, -8);
    v3f Si = Si_cfl * pow(phi, -10);

    valuef u0 = sqrt((mu_h + pressure) / (mu + pressure));

    valuef Gamma = get_Gamma();

    valuef p0_e = pressure / (Gamma - 1);
    valuef p0 = mu - p0_e;

    value gA = args.gA;

    //fluid dynamics cannot have a singular initial slice, so setting the clamping pretty high here because its irrelevant
    //thing is we have 0 quantities at the singularity, so as long as you don't generate a literal NaN here, you're 100% fine
    valuef cW = max(args.W, 0.1f);

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

hydrodynamic_plugin::hydrodynamic_plugin(cl::context ctx)
{
    cl::async_build_and_cache(ctx, []{
        return value_impl::make_function(init_hydro, "init_hydro");
    }, {"init_hydro"});
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
    hydrodynamic_buffers& bufs = *dynamic_cast<hydrodynamic_buffers*>(to_init);
    hydrodynamic_utility_buffers& ubufs = *dynamic_cast<hydrodynamic_utility_buffers*>(to_init_utility);

    {
        t3i dim = pack.dim;

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
        args.push_back(pack.disc.mu_cfl);
        args.push_back(pack.disc.mu_h_cfl);
        args.push_back(pack.disc.pressure_cfl);
        args.push_back(pack.disc.cfl);
        args.push_back(pack.disc.Si_cfl[0]);
        args.push_back(pack.disc.Si_cfl[1]);
        args.push_back(pack.disc.Si_cfl[2]);

        cqueue.exec("init_hydro", args, {dim.x(), dim.y(), dim.z()}, {8,8,1});
    }
}

void hydrodynamic_plugin::add_args_provider(all_adm_args_mem& mem)
{
    mem.add(hydrodynamic_args<buffer<valuef>>());
}

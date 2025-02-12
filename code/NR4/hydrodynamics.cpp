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

void hydrodynamic_adm::add_adm_S(bssn_args& args, valuef& in)
{

}
///W is our utility variable SIGH
void hydrodynamic_adm::add_adm_p(bssn_args& args, valuef& in)
{

}
void hydrodynamic_adm::add_adm_Si(bssn_args& args, tensor<valuef, 3>& in)
{

}
void hydrodynamic_adm::add_adm_W2_Sij(bssn_args& args, tensor<valuef, 3, 3>& in)
{

}

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

void hydrodynamic_plugin::init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, buffer_provider* to_init)
{
    hydrodynamic_buffers* bufs = dynamic_cast<hydrodynamic_buffers*>(to_init);

    assert(bufs);
}

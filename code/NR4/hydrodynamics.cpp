#include "hydrodynamics.hpp"
#include "init_general.hpp"

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
                buffer<valuef> mu_cfl, buffer<valuef> mu_h_cfl, buffer<valuef> pressure_cfl, buffer<valuef> cfl, std::array<buffer<valuef>, 3> Si_cfl)
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

    /*valuef p_star = hydro.p_star[pos, dim];
    valuef e_star = hydro.e_star[pos, dim];
    v3f = hydro.p_star[pos, dim];*/
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

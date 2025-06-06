#ifndef INIT_GENERAL_HPP_INCLUDED
#define INIT_GENERAL_HPP_INCLUDED

#include "init_black_hole.hpp"
#include "tensor_algebra.hpp"
#include "laplace.hpp"
#include "init_neutron_star.hpp"
#include "plugin.hpp"

template<typename Type, typename Func>
inline
cl::buffer discretise(cl::context& ctx, cl::command_queue& cqueue, Func&& func, tensor<int, 3> cdim, float cscale)
{
    auto kern = [func](execution_context& ctx, buffer_mut<value<Type>> out, literal<v3i> dim, literal<valuef> scale)
    {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        if_e(lid >= dim.get().x() * dim.get().y() * dim.get().z(), [&]{
            return_e();
        });

        v3i pos = get_coordinate(lid, {dim.get().x(), dim.get().y(), dim.get().z()});

        as_ref(out[pos, dim.get()]) = func(pos, dim.get(), scale.get());
    };

    std::string str = value_impl::make_function(kern, "discretise");

    cl::program prog = cl::build_program_with_cache(ctx, {str}, false);

    cl::kernel k(prog, "discretise");

    cl::buffer buf(ctx);
    buf.alloc(sizeof(Type) * cdim.x() * cdim.y() * cdim.z());

    cl::args args;
    args.push_back(buf, cdim, cscale);

    k.set_args(args);

    cqueue.exec(k, {cdim.x() * cdim.y() * cdim.z()}, {128});

    return buf;
}

///this initial setup is horrendous
struct discretised_initial_data
{
    cl::buffer mu_h_cfl;
    cl::buffer cfl; //for black holes this is inited to 1/a
    std::array<cl::buffer, 6> AIJ_cfl;
    std::array<cl::buffer, 3> Si_cfl;
    cl::buffer star_indices;
    std::array<cl::buffer, 3> col;

    discretised_initial_data(cl::context& ctx) : mu_h_cfl(ctx), cfl(ctx), AIJ_cfl{ctx, ctx, ctx, ctx, ctx, ctx}, Si_cfl{ctx, ctx, ctx}, star_indices(ctx), col{ctx, ctx, ctx}{}

    void init(cl::command_queue& cqueue, t3i dim);
};

struct initial_pack
{
    discretised_initial_data disc;
    int neutron_index = 0;
    std::vector<neutron_star::numerical_eos> stored_eos;

    tensor<int, 3> dim;
    float scale = 0.f;
    cl::buffer aij_aIJ_buf;

    initial_pack(cl::context& ctx, cl::command_queue& cqueue, tensor<int, 3> _dim, float _scale) : disc(ctx), aij_aIJ_buf(ctx)
    {
        dim = _dim;
        scale = _scale;

        disc.init(cqueue, dim);

        aij_aIJ_buf.alloc(sizeof(cl_float) * dim.x() * dim.y() * dim.z());
        aij_aIJ_buf.set_to_zero(cqueue);
    }

    void add(cl::command_queue& cqueue, const black_hole_data& bh)
    {
        for(int i=0; i < 6; i++)
        {
            cl::args args;
            args.push_back(disc.AIJ_cfl[i]);
            args.push_back(bh.aij[i]);
            args.push_back(dim);

            cqueue.exec("sum_buffers", args, {dim.x() * dim.y() * dim.z()}, {128});
        }

        cl::args args;
        args.push_back(disc.cfl);
        args.push_back(bh.conformal_guess);
        args.push_back(dim);

        cqueue.exec("sum_buffers", args, {dim.x() * dim.y() * dim.z()}, {128});
    }

    void add(cl::context& ctx, cl::command_queue& cqueue, neutron_star::data& ns)
    {
        stored_eos.push_back(ns.stored);
        ns.add_to_solution(ctx, cqueue, disc, dim, scale, neutron_index++);
    }

    void add(cl::context& ctx, cl::command_queue& cqueue, const black_hole_params& bh)
    {
        black_hole_data dat = init_black_hole(ctx, cqueue, bh, dim, scale);
        add(cqueue, dat);
    }

    void finalise(cl::command_queue& cqueue)
    {
        cl::args args;
        args.push_back(aij_aIJ_buf);

        for(int i=0; i < 6; i++)
            args.push_back(disc.AIJ_cfl[i]);

        args.push_back(dim);

        cqueue.exec("aijaij", args, {dim.x() * dim.y() * dim.z()}, {128});
    }

    void push(cl::args& args)
    {
        args.push_back(disc.cfl);
        args.push_back(aij_aIJ_buf);
        args.push_back(disc.mu_h_cfl);
    }
};

struct initial_params
{
    float N = 2;
    float lapse_damp_timescale = 20.f;
    float linear_viscosity_timescale = 0;
    float time_between_snapshots = 2;
    float linear_viscosity_strength = 0.1f;
    float quadratic_viscosity_strength = 0.1f;

    t3i dim = {155, 155, 155};
    float simulation_width = 40;

    std::vector<neutron_star::data> params_ns;
    std::vector<black_hole_params> params_bh;

    void add(const neutron_star::parameters& ns)
    {
        neutron_star::data dat(ns);
        params_ns.push_back(dat);

        std::cout << "Msols " << dat.sol.M_msol << std::endl;
    }

    void add(const black_hole_params& bh)
    {
        params_bh.push_back(bh);
    }

    std::pair<cl::buffer, initial_pack> build(cl::context& ctx, cl::command_queue& cqueue, float simulation_width, bssn_buffer_pack& to_fill);
    bool hydrodynamics_wants_colour();
    bool hydrodynamics_enabled();
};

void boot_initial_kernels(cl::context ctx);

laplace_solver& get_laplace_solver(cl::context ctx);

std::vector<float> extract_adm_masses(cl::context& ctx, cl::command_queue& cqueue, cl::buffer u_buf, t3i u_dim, float scale, const std::vector<black_hole_params>& params_bh);

#endif // INIT_GENERAL_HPP_INCLUDED

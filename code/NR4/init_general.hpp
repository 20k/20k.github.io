#ifndef INIT_GENERAL_HPP_INCLUDED
#define INIT_GENERAL_HPP_INCLUDED

#include "init_black_hole.hpp"
#include "tensor_algebra.hpp"
#include "laplace.hpp"
#include "init_neutron_star.hpp"
#include "tov.hpp"
#include "plugin.hpp"

struct discretised_initial_data
{
    cl::buffer mu_cfl;
    cl::buffer mu_h_cfl;
    cl::buffer pressure_cfl;
    cl::buffer cfl; //for black holes this is inited to 1/a
    std::array<cl::buffer, 6> AIJ_cfl;
    std::array<cl::buffer, 3> Si_cfl;

    discretised_initial_data(cl::context& ctx) : mu_cfl(ctx), mu_h_cfl(ctx), pressure_cfl(ctx), cfl(ctx), AIJ_cfl{ctx, ctx, ctx, ctx, ctx, ctx}, Si_cfl{ctx, ctx, ctx}{}

    void init(cl::command_queue& cqueue, t3i dim)
    {
        int64_t cells = int64_t{dim.x()} * dim.y() * dim.z();

        mu_cfl.alloc(sizeof(cl_float) * cells);
        mu_h_cfl.alloc(sizeof(cl_float) * cells);
        pressure_cfl.alloc(sizeof(cl_float) * cells);
        cfl.alloc(sizeof(cl_float) * cells);

        mu_cfl.set_to_zero(cqueue);
        mu_h_cfl.set_to_zero(cqueue);
        pressure_cfl.set_to_zero(cqueue);
        cfl.fill(cqueue, cl_float{1});

        for(auto& i : AIJ_cfl)
        {
            i.alloc(sizeof(cl_float) * cells);
            i.set_to_zero(cqueue);
        }

        for(auto& i : Si_cfl)
        {
            i.alloc(sizeof(cl_float) * cells);
            i.set_to_zero(cqueue);
        }
    }
};

struct initial_pack
{
    discretised_initial_data disc;

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

    void add(cl::context& ctx, cl::command_queue& cqueue, const neutron_star::parameters& ns)
    {
        tov::parameters params;
        params.K = ns.K;
        params.Gamma = ns.Gamma;

        tov::integration_state st = tov::make_integration_state_si(ns.p0_c_kg_m3, 1e-6, params);
        tov::integration_solution sol = tov::solve_tov(st, params, 1e-6, 0.);

        neutron_star::add_to_solution(ctx, cqueue, disc, ns, sol, dim, scale);
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

struct all_laplace_args : value_impl::single_source::argument_pack
{
    buffer<valuef> cfl;
    buffer<valuef> aij_aIJ;
    buffer<valuef> mu_h_cfl;

    void build(auto& in)
    {
        using namespace value_impl::builder;

        add(cfl, in);
        add(aij_aIJ, in);
        add(mu_h_cfl, in);
    }
};

struct initial_conditions
{
    tensor<int, 3> dim;

    std::vector<black_hole_params> params_bh;
    std::vector<neutron_star::parameters> params_ns;

    laplace_solver laplace;

    initial_conditions(cl::context& ctx, cl::command_queue& cqueue, tensor<int, 3> _dim)
    {
        dim = _dim;

        {
            auto sum_buffers = [](execution_context& ctx, buffer_mut<valuef> inout, buffer<valuef> in, literal<v3i> dim)
            {
                using namespace single_source;

                valuei lid = value_impl::get_global_id(0);

                pin(lid);

                if_e(lid >= dim.get().x() * dim.get().y() * dim.get().z(), [&]{
                    return_e();
                });

                as_ref(inout[lid]) = as_constant(inout[lid]) + in[lid];
            };

            cl::async_build_and_cache(ctx, [=]{
                return value_impl::make_function(sum_buffers, "sum_buffers");
            }, {"sum_buffers"});
        }

        {
            auto calculate_aijaIJ = [](execution_context& ectx, buffer_mut<valuef> aij_aIJ_out, std::array<buffer<valuef>, 6> aIJ_packed, literal<v3i> dim)
            {
                using namespace single_source;

                valuei lid = value_impl::get_global_id(0);

                pin(lid);

                if_e(lid >= dim.get().x() * dim.get().y() * dim.get().z(), [&]{
                    return_e();
                });

                metric<valuef, 3, 3> met;

                for(int i=0; i < 3; i++)
                    met[i, i] = 1;

                int index_table[3][3] = {{0, 1, 2},
                                         {1, 3, 4},
                                         {2, 4, 5}};

                tensor<valuef, 3, 3> aIJ;

                for(int i=0; i < 3; i++)
                    for(int j=0; j < 3; j++)
                        aIJ[i, j] = aIJ_packed[index_table[i][j]][lid];

                tensor<valuef, 3, 3> aij = lower_both(aIJ, met);

                as_ref(aij_aIJ_out[lid]) = sum_multiply(aij, aIJ);
            };

            cl::async_build_and_cache(ctx, [=]{
                return value_impl::make_function(calculate_aijaIJ, "aijaij");
            }, {"aijaij"});
        }

        laplace.boot(ctx, [](laplace_params params, all_laplace_args args)
        {
            v3i pos = params.pos;
            v3i dim = params.dim;
            auto cfl = args.cfl[pos, dim] + params.u[pos, dim];
            auto mu_h = args.mu_h_cfl[pos, dim];

            return -(1.f/8.f) * pow(cfl, -7.f) * args.aij_aIJ[pos, dim] - 2 * M_PI * pow(cfl, valuef(-3)) * mu_h;
        }, all_laplace_args(), "laplace_rb_mg");

        {
            auto calculate_bssn_variables = [](execution_context& ectx,
                                               bssn_args_mem<buffer_mut<valuef>> out,
                                               buffer<valuef> cfl_reg, buffer<valuef> u,
                                               std::array<buffer<valuef>, 6> aIJ_summed,
                                               literal<v3i> dim) {
                using namespace single_source;

                valuei lid = value_impl::get_global_id(0);

                pin(lid);

                if_e(lid >= dim.get().x() * dim.get().y() * dim.get().z(), [&] {
                    return_e();
                });

                valuef cfl = cfl_reg[lid] + u[lid];

                metric<valuef, 3, 3> flat;

                for(int i=0; i < 3; i++)
                    flat[i, i] = 1;

                metric<valuef, 3, 3> Yij = flat * pow(cfl, 4.f);

                int index_table[3][3] = {{0, 1, 2},
                                         {1, 3, 4},
                                         {2, 4, 5}};

                tensor<valuef, 3, 3> baIJ;

                for(int i=0; i < 3; i++)
                {
                    for(int j=0; j < 3; j++)
                    {
                        baIJ[i, j] = aIJ_summed[index_table[i][j]][lid];
                    }
                }

                tensor<valuef, 3, 3> Kij = lower_both(baIJ, flat) * pow(cfl, -2.f);

                valuef gA = 1/(pow(cfl, valuef(2)));
                //valuef gA = 1;
                tensor<valuef, 3> gB = {0,0,0};
                tensor<valuef, 3> cG = {0,0,0};

                valuef W = pow(Yij.det(), -1/6.f);
                metric<valuef, 3, 3> cY = W*W * Yij;
                //valuef K = trace(Kij, Yij.invert()); // 0
                valuef K = 0;

                cY = flat;

                tensor<valuef, 3, 3> cA = W*W * (Kij - (1.f/3.f) * Yij.to_tensor() * K);

                std::array<valuef, 6> packed_cA = extract_symmetry(cA);
                std::array<valuef, 6> packed_cY = extract_symmetry(cY.to_tensor());

                for(int i=0; i < 6; i++)
                {
                    as_ref(out.cY[i][lid]) = packed_cY[i];
                    as_ref(out.cA[i][lid]) = packed_cA[i];
                }

                as_ref(out.K[lid]) = K;
                as_ref(out.W[lid]) = W;

                for(int i=0; i < 3; i++)
                    as_ref(out.cG[i][lid]) = cG[i];

                as_ref(out.gA[lid]) = gA;

                for(int i=0; i < 3; i++)
                    as_ref(out.gB[i][lid]) = gB[i];
            };

            cl::async_build_and_cache(ctx, [=]{
                return value_impl::make_function(calculate_bssn_variables, "calculate_bssn_variables");
            }, {"calculate_bssn_variables"});
        }

        {
            auto fetch_linear = [](execution_context& ectx, buffer<valuef> buf, literal<v3f> pos, literal<v3i> dim, buffer_mut<valuef> out)
            {
                using namespace single_source;

                valuei lid = value_impl::get_global_id(0);

                if_e(lid != 0, [&]{
                    return_e();
                });

                as_ref(out[0]) = buffer_read_linear(buf, pos.get(), dim.get());
            };

            cl::async_build_and_cache(ctx, [=]{
                return value_impl::make_function(fetch_linear, "fetch_linear");
            }, {"fetch_linear"});
        }
    }

    void add(const black_hole_params& bh)
    {
        params_bh.push_back(bh);
    }

    void add(const neutron_star::parameters& ns)
    {
        params_ns.push_back(ns);
    }

    std::vector<float> extract_adm_masses(cl::context& ctx, cl::command_queue& cqueue, cl::buffer u_buf, t3i u_dim, float scale)
    {
        std::vector<float> ret;

        ///https://arxiv.org/pdf/gr-qc/0610128 6
        for(const black_hole_params& bh : params_bh)
        {
            ///Mi = mi(1 + ui + sum(m_j / 2d_ij) i != j
            t3f pos = world_to_grid(bh.position, u_dim, scale);

            cl::buffer u_read(ctx);
            u_read.alloc(sizeof(cl_float));

            cl::args args;
            args.push_back(u_buf, pos, u_dim, u_read);

            cqueue.exec("fetch_linear", args, {1}, {1});

            float u = u_read.read<float>(cqueue).at(0);

            float sum = 0;

            for(const black_hole_params& bh2 : params_bh)
            {
                if(&bh == &bh2)
                    continue;

                sum += bh2.bare_mass / (2 * (bh2.position - bh.position).length());
            }

            float adm_mass = bh.bare_mass * (1 + u + sum);

            ret.push_back(adm_mass);
        }

        return ret;
    }

    std::pair<cl::buffer, initial_pack> build(cl::context& ctx, cl::command_queue& cqueue, float simulation_width, bssn_buffer_pack& to_fill)
    {
        auto [u_found, pack] = laplace.solve(ctx, cqueue, simulation_width, dim,
                                            [&ctx, &cqueue, this](t3i idim, float iscale)
        {
            initial_pack pack(ctx, cqueue, idim, iscale);

            for(auto& i : params_bh)
                pack.add(ctx, cqueue, i);

            for(auto& i : params_ns)
                pack.add(ctx, cqueue, i);

            pack.finalise(cqueue);
            return pack;
        });

        {
            cl::args args;

            to_fill.for_each([&](cl::buffer& buf) {
                args.push_back(buf);
            });

            args.push_back(pack.disc.cfl);
            args.push_back(u_found);

            for(int i=0; i < 6; i++)
                args.push_back(pack.disc.AIJ_cfl[i]);

            args.push_back(dim);

            cqueue.exec("calculate_bssn_variables", args, {dim.x() * dim.y() * dim.z()}, {128});
        }

        return {u_found, pack};
    }
};

#endif // INIT_GENERAL_HPP_INCLUDED

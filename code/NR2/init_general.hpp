#ifndef INIT_GENERAL_HPP_INCLUDED
#define INIT_GENERAL_HPP_INCLUDED

#include "init_black_hole.hpp"
#include "tensor_algebra.hpp"

struct initial_conditions
{
    std::array<cl::buffer, 6> aIJ_summed;
    cl::buffer cfl_summed;
    tensor<int, 3> dim;

    initial_conditions(cl::context& ctx, cl::command_queue& cqueue, tensor<int, 3> _dim) : aIJ_summed{ctx, ctx, ctx, ctx, ctx, ctx}, cfl_summed{ctx}
    {
        dim = _dim;

        for(int i=0; i < 6; i++)
        {
            aIJ_summed[i].alloc(sizeof(cl_float) * dim.x() * dim.y() * dim.z());
            aIJ_summed[i].set_to_zero(cqueue);
        }

        cfl_summed.alloc(sizeof(cl_float) * dim.x() * dim.y() * dim.z());
        cfl_summed.fill(cqueue, 1.f);

        auto sum_buffers = [&](execution_context& ctx, buffer_mut<valuef> inout, buffer_mut<valuef> in)
        {
            using namespace single_source;

            valuei lid = value_impl::get_global_id(0);

            pin(lid);

            if_e(lid >= dim.x() * dim.y() * dim.z(), [&]{
                return_e();
            });

            as_ref(inout[lid]) = as_constant(inout[lid]) + in[lid];
        };

        cl::program prog(ctx, value_impl::make_function(sum_buffers, "sum_buffers"), false);
        prog.build(ctx, "");

        ctx.register_program(prog);
    }

    void add(cl::command_queue& cqueue, black_hole_data& bh)
    {
        for(int i=0; i < 6; i++)
        {
            cl::args args;
            args.push_back(aIJ_summed[i]);
            args.push_back(bh.aij[i]);

            cqueue.exec("sum_buffers", args, {dim.x() * dim.y() * dim.z()}, {128});
        }

        cl::args args;
        args.push_back(cfl_summed);
        args.push_back(bh.conformal_guess);

        cqueue.exec("sum_buffers", args, {dim.x() * dim.y() * dim.z()}, {128});
    }

    void build(cl::context& ctx, cl::command_queue& cqueue, float scale, bssn_buffer_pack& to_fill)
    {
        cl::buffer aij_aIJ_buf(ctx);
        aij_aIJ_buf.alloc(sizeof(cl_float) * dim.x() * dim.y() * dim.z());

        auto calculate_aijaIJ = [&](execution_context& ectx, buffer_mut<valuef> aij_aIJ_out, std::array<buffer<valuef>, 6> aIJ_packed)
        {
            using namespace single_source;

            valuei lid = value_impl::get_global_id(0);

            pin(lid);

            if_e(lid >= dim.x() * dim.y() * dim.z(), [&]{
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

        cl::program calc(ctx, value_impl::make_function(calculate_aijaIJ, "aijaij"), false);
        calc.build(ctx, "");

        ctx.register_program(calc);

        {
            auto laplace = [](execution_context& ectx, buffer_mut<valuef> out, buffer<valuef> in, buffer<valuef> cfl, buffer<valuef> aij_aIJ, literal<valuef> lscale, literal<v3i> ldim)
            {
                using namespace single_source;

                valuei lid = value_impl::get_global_id(0);

                pin(lid);

                auto dim = ldim.get();

                if_e(lid >= dim.x() * dim.y() * dim.z(), [&]{
                    return_e();
                });

                v3i pos = get_coordinate(lid, dim);

                if_e(pos.x() == 0 || pos.y() == 0 || pos.z() == 0 ||
                     pos.x() == dim.x() - 1 || pos.y() == dim.y() - 1 || pos.z() == dim.z() - 1, [&] {
                    as_ref(out[lid]) = in[lid];

                    return_e();
                });

                valuef rhs = -(1.f/8.f) * pow(cfl[lid] + in[lid], -7.f) * aij_aIJ[lid];

                valuef h2f0 = lscale.get() * lscale.get() * rhs;

                valuef uxm1 = in[pos - (v3i){1, 0, 0}, dim];
                valuef uxp1 = in[pos + (v3i){1, 0, 0}, dim];
                valuef uym1 = in[pos - (v3i){0, 1, 0}, dim];
                valuef uyp1 = in[pos + (v3i){0, 1, 0}, dim];
                valuef uzm1 = in[pos - (v3i){0, 0, 1}, dim];
                valuef uzp1 = in[pos + (v3i){0, 0, 1}, dim];

                valuef Xs = uxm1 + uxp1;
                valuef Ys = uyp1 + uym1;
                valuef Zs = uzp1 + uzm1;

                valuef u0n1 = (1/6.f) * (Xs + Ys + Zs - h2f0);

                valuef u = in[pos, dim];

                /*valuef err = u0n1 - u;

                if(fabs(err) > etol)
                {
                    atomic_xchg(still_going, 1);
                }

                buffer_out[IDX(ix, iy, iz)] = mix(u, u0n1, 0.9f);*/

                as_ref(out[lid]) = mix(u, u0n1, valuef(0.9f));
            };

            cl::program calc(ctx, value_impl::make_function(laplace, "laplace"), false);
            calc.build(ctx, "");

            ctx.register_program(calc);
        }

        {
            cl::args args;
            args.push_back(aij_aIJ_buf);

            for(int i=0; i < 6; i++)
                args.push_back(aIJ_summed[i]);

            cqueue.exec("aijaij", args, {dim.x() * dim.y() * dim.z()}, {128});
        }

        cl::buffer u_found(ctx);
        cl_int3 size = {dim.x(), dim.y(), dim.z()};

        {

            std::array<cl::buffer, 2> u{ctx, ctx};

            for(int i=0; i < 2; i++)
            {
                u[i].alloc(sizeof(cl_float) * dim.x() * dim.y() * dim.z());
                u[i].set_to_zero(cqueue);
            }

            for(int i=0; i < 1000; i++)
            {
                cl::args args;
                args.push_back(u[i]);
                args.push_back(u[(i + 1) % 2]);
                args.push_back(cfl_summed);
                args.push_back(aij_aIJ_buf);
                args.push_back(scale);
                args.push_back(size);

                cqueue.exec("laplace", args, {dim.x() * dim.y() * dim.z()}, {128});

                u_found = u[(i + 1) % 2];
            }
        }

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

                valuef gA = 1;
                tensor<valuef, 3> gB = {0,0,0};
                tensor<valuef, 3> cG = {0,0,0};

                valuef W = pow(Yij.det(), -1/6.f);
                metric<valuef, 3, 3> cY = W*W * Yij;
                valuef K = trace(Kij, Yij.invert()); // 0

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

            std::string str = value_impl::make_function(calculate_bssn_variables, "calculate_bssn_variables");

            cl::program p(ctx, str, false);
            p.build(ctx, "");

            ctx.register_program(p);
        }

        {
            cl::args args;

            to_fill.for_each([&](cl::buffer& buf)
            {
                args.push_back(buf);
            });

            args.push_back(cfl_summed);
            args.push_back(u_found);

            for(int i=0; i < 6; i++)
                args.push_back(aIJ_summed[i]);

            args.push_back(size);

            cqueue.exec("calculate_bssn_variables", args, {dim.x() * dim.y() * dim.z()}, {128});
        }
    }
};

#endif // INIT_GENERAL_HPP_INCLUDED

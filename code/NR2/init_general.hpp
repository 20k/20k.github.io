#ifndef INIT_GENERAL_HPP_INCLUDED
#define INIT_GENERAL_HPP_INCLUDED

#include "init_black_hole.hpp"
#include "tensor_algebra.hpp"

template<typename T>
inline
auto buffer_read_nearest_clamp(T buf, v3i pos, v3i dim)
{
    pos = clamp(pos, (v3i){0,0,0}, dim);

    return buf[pos, dim];
}

template<typename T>
inline
auto buffer_read_linear(T buf, v3f pos, v3i dim)
{
    v3f floored = floor(pos);
    v3f frac = pos - floored;

    v3i ipos = (v3i)floored;

    auto c000 = buffer_read_nearest_clamp(buf, ipos + (v3i){0,0,0}, dim);
    auto c100 = buffer_read_nearest_clamp(buf, ipos + (v3i){1,0,0}, dim);

    auto c010 = buffer_read_nearest_clamp(buf, ipos + (v3i){0,1,0}, dim);
    auto c110 = buffer_read_nearest_clamp(buf, ipos + (v3i){1,1,0}, dim);

    auto c001 = buffer_read_nearest_clamp(buf, ipos + (v3i){0,0,1}, dim);
    auto c101 = buffer_read_nearest_clamp(buf, ipos + (v3i){1,0,1}, dim);

    auto c011 = buffer_read_nearest_clamp(buf, ipos + (v3i){0,1,1}, dim);
    auto c111 = buffer_read_nearest_clamp(buf, ipos + (v3i){1,1,1}, dim);

    ///numerically symmetric across the centre of dim
    auto c00 = c000 - frac.x() * (c000 - c100);
    auto c01 = c001 - frac.x() * (c001 - c101);

    auto c10 = c010 - frac.x() * (c010 - c110);
    auto c11 = c011 - frac.x() * (c011 - c111);

    auto c0 = c00 - frac.y() * (c00 - c10);
    auto c1 = c01 - frac.y() * (c01 - c11);

    return c0 - frac.z() * (c0 - c1);
}

inline
valuef get_scaled_coordinate(valuei in, valuei dimension_upper, valuei dimension_lower)
{
    valuei upper_centre = (dimension_upper - 1)/2;

    valuei upper_offset = in - upper_centre;

    valuef scale = (valuef)(dimension_upper - 1) / (valuef)(dimension_lower - 1);

    ///so lets say we have [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] with a dimension of 13
    ///this gives a middle value of 6, which is the 7th value
    ///Then we want to scale it to a dimension of 7
    ///to get [0:0, 1:0.5, 2:1, 3:1.5, 4:2, 5:2.5, 6:3, 7:3.5, 8:4, 9:4.5, 10:5, 11:5.5, 12:6]
    ///so... it should just be a straight division by the scale?

    return (valuef)in / scale;
}

inline
v3f get_scaled_coordinate_vec(v3i in, v3i dimension_upper, v3i dimension_lower)
{
    return (v3f){get_scaled_coordinate(in.x(), dimension_upper.x(), dimension_lower.x()),
                 get_scaled_coordinate(in.y(), dimension_upper.y(), dimension_lower.y()),
                 get_scaled_coordinate(in.z(), dimension_upper.z(), dimension_lower.z())};
}


inline
void upscale_buffer_with_boundary(execution_context& ctx, buffer<valuef> in, buffer_mut<valuef> out, literal<v3i> in_dim, literal<v3i> out_dim, literal<valuef> boundary)
{
    using namespace single_source;

    valuei lid = value_impl::get_global_id(0);

    pin(lid);

    auto dim = out_dim.get();

    if_e(lid >= dim.x() * dim.y() * dim.z(), [&]{
        return_e();
    });

    v3i pos = get_coordinate(lid, dim);

    v3f lower_pos = get_scaled_coordinate_vec(pos, out_dim.get(), in_dim.get());

    if_e(pos.x() == 0 || pos.y() == 0 || pos.z() == 0 ||
         pos.x() == out_dim.get().x() ||  pos.y() == out_dim.get().y() || pos.z() == out_dim.get().z(), [&]{
        as_ref(out[pos, out_dim.get()]) = boundary.get();

        return_e();
    });

    ///buffer read linear
    valuef val = buffer_read_linear(in, lower_pos, in_dim.get());

    as_ref(out[pos, out_dim.get()]) = val;
}

struct initial_pack
{
    std::array<cl::buffer, 6> aIJ_summed;
    cl::buffer cfl_summed;
    tensor<int, 3> dim;
    float scale = 0.f;
    cl::buffer aij_aIJ_buf;

    initial_pack(cl::context& ctx, cl::command_queue& cqueue, tensor<int, 3> _dim, float _scale) : aIJ_summed{ctx, ctx, ctx, ctx, ctx, ctx}, cfl_summed{ctx}, aij_aIJ_buf{ctx}
    {
        dim = _dim;
        scale = _scale;

        for(int i=0; i < 6; i++)
        {
            aIJ_summed[i].alloc(sizeof(cl_float) * dim.x() * dim.y() * dim.z());
            aIJ_summed[i].set_to_zero(cqueue);
        }

        cfl_summed.alloc(sizeof(cl_float) * dim.x() * dim.y() * dim.z());
        cfl_summed.fill(cqueue, 1.f);

        aij_aIJ_buf.alloc(sizeof(cl_float) * dim.x() * dim.y() * dim.z());
        aij_aIJ_buf.set_to_zero(cqueue);
    }

    void add(cl::command_queue& cqueue, const black_hole_data& bh)
    {
        for(int i=0; i < 6; i++)
        {
            cl::args args;
            args.push_back(aIJ_summed[i]);
            args.push_back(bh.aij[i]);
            args.push_back(dim);

            cqueue.exec("sum_buffers", args, {dim.x() * dim.y() * dim.z()}, {128});
        }

        cl::args args;
        args.push_back(cfl_summed);
        args.push_back(bh.conformal_guess);
        args.push_back(dim);

        cqueue.exec("sum_buffers", args, {dim.x() * dim.y() * dim.z()}, {128});
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
            args.push_back(aIJ_summed[i]);

        args.push_back(dim);

        cqueue.exec("aijaij", args, {dim.x() * dim.y() * dim.z()}, {128});
    }
};

struct initial_conditions
{
    tensor<int, 3> dim;

    std::vector<black_hole_params> params_bh;

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

            std::string sum_str = value_impl::make_function(sum_buffers, "sum_buffers");

            cl::program prog(ctx, sum_str, false);
            prog.build(ctx, "");

            ctx.register_program(prog);
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

            cl::program calc(ctx, value_impl::make_function(calculate_aijaIJ, "aijaij"), false);
            calc.build(ctx, "");

            ctx.register_program(calc);
        }

        {
            auto laplace = [](execution_context& ectx, buffer_mut<valuef> inout, buffer<valuef> cfl, buffer<valuef> aij_aIJ,
                              literal<valuef> lscale, literal<v3i> ldim, literal<valuei> iteration,
                              buffer_mut<valuei> still_going, literal<valuef> relax)
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

                    return_e();
                });

                valuei lix = pos.x() + (pos.z() % 2);
                valuei liy = pos.y();

                if_e(((lix + liy) % 2) == (iteration.get() % 2), [] {
                    return_e();
                });

                valuef rhs = -(1.f/8.f) * pow(cfl[lid] + inout[lid], -7.f) * aij_aIJ[lid];

                valuef h2f0 = lscale.get() * lscale.get() * rhs;

                valuef uxm1 = inout[pos - (v3i){1, 0, 0}, dim];
                valuef uxp1 = inout[pos + (v3i){1, 0, 0}, dim];
                valuef uym1 = inout[pos - (v3i){0, 1, 0}, dim];
                valuef uyp1 = inout[pos + (v3i){0, 1, 0}, dim];
                valuef uzm1 = inout[pos - (v3i){0, 0, 1}, dim];
                valuef uzp1 = inout[pos + (v3i){0, 0, 1}, dim];

                valuef Xs = uxm1 + uxp1;
                valuef Ys = uyp1 + uym1;
                valuef Zs = uzp1 + uzm1;

                valuef u0n1 = (1/6.f) * (Xs + Ys + Zs - h2f0);

                valuef u = inout[pos, dim];

                /*if_e(pos.x() == 128 && pos.y() == 128 && pos.z() == 128, [&]{
                    value_base se;
                    se.type = value_impl::op::SIDE_EFFECT;
                    se.abstract_value = "printf(\"%.23f\\n\"," + value_to_string(u) + ")";

                    value_impl::get_context().add(se);
                });*/

                as_ref(inout[lid]) = mix(u, u0n1, relax.get());

                valuef etol = 1e-9f;

                if_e(fabs(u0n1 - u) > etol, [&]{
                    still_going.atom_xchg_e(valuei(0), valuei(1));
                });
            };

            cl::program calc(ctx, value_impl::make_function(laplace, "laplace"), false);
            calc.build(ctx, "");

            ctx.register_program(calc);
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

            std::string str = value_impl::make_function(calculate_bssn_variables, "calculate_bssn_variables");

            cl::program p(ctx, str, false);
            p.build(ctx, "");

            ctx.register_program(p);
        }

        {
            std::string str = value_impl::make_function(upscale_buffer_with_boundary, "upscale");

            cl::program p(ctx, str, false);
            p.build(ctx, "");

            ctx.register_program(p);
        }
    }

    void add(const black_hole_params& bh)
    {
        params_bh.push_back(bh);
    }

    void build(cl::context& ctx, cl::command_queue& cqueue, float simulation_width, bssn_buffer_pack& to_fill)
    {
        auto get_u_at_dim = [&](t3i dim, float simulation_width, float relax, std::optional<std::tuple<cl::buffer, t3i>> u_old)
        {
            float scale = simulation_width / (dim.x() - 1);

            cl::buffer u_found(ctx);

            initial_pack pack(ctx, cqueue, dim, scale);

            for(auto& i : params_bh)
                pack.add(ctx, cqueue, i);

            pack.finalise(cqueue);

            {
                cl::buffer still_going(ctx);
                still_going.alloc(sizeof(cl_int));
                still_going.set_to_zero(cqueue);

                u_found.alloc(sizeof(cl_float) * dim.x() * dim.y() * dim.z());
                //this is not for safety, this is the boundary condition
                u_found.set_to_zero(cqueue);

                if(u_old.has_value())
                {
                    auto [u_old_buf, u_old_dim] = u_old.value();

                    cl::args args;
                    args.push_back(u_old_buf, u_found, u_old_dim, dim, 0.f);

                    cqueue.exec("upscale", args, {dim.x() * dim.y() * dim.z()}, {128});
                }

                for(int i=0; i < 100000; i++)
                {
                    bool check = (i % 500) == 0;

                    if(check)
                        still_going.set_to_zero(cqueue);

                    cl::args args;
                    args.push_back(u_found);
                    args.push_back(pack.cfl_summed);
                    args.push_back(pack.aij_aIJ_buf);
                    args.push_back(scale);
                    args.push_back(dim);
                    args.push_back(i);
                    args.push_back(still_going);
                    args.push_back(relax);

                    cqueue.exec("laplace", args, {dim.x() * dim.y() * dim.z()}, {128});

                    if(check)
                    {
                        printf("Broke %i\n", i);

                        bool going = still_going.read<int>(cqueue)[0] == 1;

                        if(!going)
                            break;
                    }
                }
            }

            return std::pair{u_found, pack};
        };

        std::array<t3i, 5> dims = {(t3i){63, 63, 63}, (t3i){95, 95, 95}, (t3i){127, 127, 127}, (t3i){197, 197, 197}, dim};
        std::array<float, 5> relax = {0.4f, 0.5f, 0.5f, 0.7f, 0.9f};
        std::optional<std::tuple<cl::buffer, initial_pack>> last;

        {
            std::optional<std::pair<cl::buffer, t3i>> last_u;

            for(int i=0; i < 5; i++)
            {
                last = get_u_at_dim(dims[i], simulation_width, relax[i], last_u);

                last_u = {std::get<0>(last.value()), dims[i]};
            }
        }

        {
            auto [u_found, pack] = last.value();

            cl::args args;

            to_fill.for_each([&](cl::buffer& buf) {
                args.push_back(buf);
            });

            args.push_back(pack.cfl_summed);
            args.push_back(u_found);

            for(int i=0; i < 6; i++)
                args.push_back(pack.aIJ_summed[i]);

            args.push_back(dim);

            cqueue.exec("calculate_bssn_variables", args, {dim.x() * dim.y() * dim.z()}, {128});
        }
    }
};

#endif // INIT_GENERAL_HPP_INCLUDED

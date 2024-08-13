#ifndef INIT_GENERAL_HPP_INCLUDED
#define INIT_GENERAL_HPP_INCLUDED

#include "init_black_hole.hpp"
#include "tensor_algebra.hpp"

/*
float get_scaled_coordinate(int in, int dimension_upper, int dimension_lower)
{
    int upper_centre = (dimension_upper - 1)/2;

    int upper_offset = in - upper_centre;

    float scale = (float)(dimension_upper - 1) / (dimension_lower - 1);

    ///so lets say we have [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] with a dimension of 13
    ///this gives a middle value of 6, which is the 7th value
    ///Then we want to scale it to a dimension of 7
    ///to get [0:0, 1:0.5, 2:1, 3:1.5, 4:2, 5:2.5, 6:3, 7:3.5, 8:4, 9:4.5, 10:5, 11:5.5, 12:6]
    ///so... it should just be a straight division by the scale?

    return in / scale;
}

float3 get_scaled_coordinate_vec(int3 in, int3 dimension_upper, int3 dimension_lower)
{
    return (float3){get_scaled_coordinate(in.x, dimension_upper.x, dimension_lower.x),
                    get_scaled_coordinate(in.y, dimension_upper.y, dimension_lower.y),
                    get_scaled_coordinate(in.z, dimension_upper.z, dimension_lower.z)};
}

///out is > in
///this incorrectly does not produce a symmetric result
__kernel
void upscale_u(__global float* u_in, __global float* u_out, int4 in_dim, int4 out_dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= out_dim.x || iy >= out_dim.y || iz >= out_dim.z)
        return;

    float3 lower_pos = get_scaled_coordinate_vec((int3){ix, iy, iz}, out_dim.xyz, in_dim.xyz);

    float val = buffer_read_linear(u_in, lower_pos, in_dim);

    //int3 half_lower = (in_dim.xyz - 1) / 2;
    //float val = buffer_read_nearest(u_in, convert_int3(round_away_from_vec(lower_pos, convert_float3(half_lower))), in_dim);

    ///todo: remove this
    if(ix == 0 || iy == 0 || iz == 0 || ix == out_dim.x - 1 || iy == out_dim.y - 1 || iz == out_dim.z - 1)
        val = U_BOUNDARY;

    u_out[IDXD(ix, iy, iz, out_dim)] = val;
}
*/

/*
float buffer_read_nearest(__global const float* const buffer, int3 position, int3 dim)
{
    return buffer[position.z * dim.x * dim.y + position.y * dim.x + position.x];
}

float buffer_read_nearest_clamp(__global const float* const buffer, int3 position, int3 dim)
{
    position = clamp(position, (int3)(0,0,0), dim.xyz - 1);

    return buffer[position.z * dim.x * dim.y + position.y * dim.x + position.x];
}

float buffer_read_linear2(__global const float* const buffer, float px, float py, float pz, int dx, int dy, int dz)
{
    float3 floored = floor((float3)(px, py, pz));
    int3 dim = (int3)(dx, dy, dz);
    float3 position = (float3)(px, py, pz);

    int3 ipos = (int3)(floored.x, floored.y, floored.z);

    float c000 = buffer_read_nearest_clamp(buffer, ipos + (int3)(0,0,0), dim);
    float c100 = buffer_read_nearest_clamp(buffer, ipos + (int3)(1,0,0), dim);

    float c010 = buffer_read_nearest_clamp(buffer, ipos + (int3)(0,1,0), dim);
    float c110 = buffer_read_nearest_clamp(buffer, ipos + (int3)(1,1,0), dim);

    float c001 = buffer_read_nearest_clamp(buffer, ipos + (int3)(0,0,1), dim);
    float c101 = buffer_read_nearest_clamp(buffer, ipos + (int3)(1,0,1), dim);

    float c011 = buffer_read_nearest_clamp(buffer, ipos + (int3)(0,1,1), dim);
    float c111 = buffer_read_nearest_clamp(buffer, ipos + (int3)(1,1,1), dim);

    float3 frac = position - floored;

    ///numerically symmetric across the centre of dim
    float c00 = c000 - frac.x * (c000 - c100);
    float c01 = c001 - frac.x * (c001 - c101);

    float c10 = c010 - frac.x * (c010 - c110);
    float c11 = c011 - frac.x * (c011 - c111);

    float c0 = c00 - frac.y * (c00 - c10);
    float c1 = c01 - frac.y * (c01 - c11);

    return c0 - frac.z * (c0 - c1);
}
*/

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
};

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

    void add(cl::command_queue& cqueue, black_hole_data& bh)
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
            auto laplace = [](execution_context& ectx, buffer_mut<valuef> inout, buffer<valuef> cfl, buffer<valuef> aij_aIJ,
                              literal<valuef> lscale, literal<v3i> ldim, literal<valuei> iteration,
                              buffer_mut<valuei> still_going)
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

                as_ref(inout[lid]) = mix(u, u0n1, valuef(0.9f));

                valuef etol = 1e-7f;

                if_e(fabs(u0n1 - u) > etol, [&]{
                    still_going.atom_xchg_e(valuei(0), valuei(1));
                });
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
            cl::buffer still_going(ctx);
            still_going.alloc(sizeof(cl_int));
            still_going.set_to_zero(cqueue);

            u_found.alloc(sizeof(cl_float) * dim.x() * dim.y() * dim.z());
            //this is not for safety, this is the boundary condition
            u_found.set_to_zero(cqueue);

            for(int i=0; i < 100000; i++)
            {
                bool check = (i % 500) == 0;

                if(check)
                    still_going.set_to_zero(cqueue);

                cl::args args;
                args.push_back(u_found);
                args.push_back(cfl_summed);
                args.push_back(aij_aIJ_buf);
                args.push_back(scale);
                args.push_back(size);
                args.push_back(i);
                args.push_back(still_going);

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
            cl::args args;

            to_fill.for_each([&](cl::buffer& buf) {
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

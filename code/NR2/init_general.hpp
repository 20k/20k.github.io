#ifndef INIT_GENERAL_HPP_INCLUDED
#define INIT_GENERAL_HPP_INCLUDED

#include "init_black_hole.hpp"

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
};

#endif // INIT_GENERAL_HPP_INCLUDED

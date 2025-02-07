#ifndef LAPLACE_HPP_INCLUDED
#define LAPLACE_HPP_INCLUDED

#include "../common/single_source.hpp"

/**
Ok. What I want is to be able to solve laplacians
laplacians have two things: a rhs, and a set of buffers
so, what I want is to pass in that rhs, and those buffers, and win. To pass in the rhs, it'll need to be a function of position
so lets have it take a function thats a function of position, and our buffers
*/

/*
eg
template<typename T>
struct bssn_derivatives_mem : value_impl::single_source::argument_pack
{
    std::array<std::array<T, 3>, 6> dcY;
    std::array<T, 3> dgA;
    std::array<std::array<T, 3>, 3> dgB;
    std::array<T, 3> dW;

    void build(auto& in)
    {
        using namespace value_impl::builder;

        add(dcY, in);
        add(dgA, in);
        add(dgB, in);
        add(dW, in);
    }
};*/

struct laplace_solver
{
    template<typename T, typename U>
    void boot(cl::context ctx, T get_rhs, U arg_pack_unused, std::string kname)
    {
        static_assert(std::is_base_of_v<value_impl::single_source::argument_pack, U>);

        auto laplace_rb_mg = [get_rhs](execution_context& ectx, buffer_mut<valuef> inout, U pack,
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

            valuef rhs = get_rhs(inout, pack, lid);

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

            pin(Xs);
            pin(Ys);
            pin(Zs);

            valuef u0n1 = (1/6.f) * (Xs + Ys + Zs - h2f0);

            valuef u = inout[pos, dim];

            as_ref(inout[lid]) = mix(u, u0n1, relax.get());

            valuef etol = 1e-6f;

            if_e(fabs(u0n1 - u) > etol, [&]{
                still_going.atom_xchg_e(valuei(0), valuei(1));
            });
        };

        cl::async_build_and_cache(ctx, [=] {
            return value_impl::make_function(laplace_rb_mg, kname);
        }, {kname});
    }

    cl::buffer solve(cl::context ctx, float scale, t3f dim)
    {
        cl::buffer ret(ctx);
        ret.alloc(sizeof(cl_float) * dim.x() * dim.y() * dim.z());

        return ret;
    }
};

#endif // LAPLACE_HPP_INCLUDED

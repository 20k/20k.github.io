#ifndef LAPLACE_HPP_INCLUDED
#define LAPLACE_HPP_INCLUDED

#include "../common/single_source.hpp"
#include "bssn.hpp"
#include "value_alias.hpp"

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
    v3f ret;

    for(int i=0; i < 3; i++)
        ret[i] = get_scaled_coordinate(in[i], dimension_upper[i], dimension_lower[i]);

    return ret;
}

inline
void upscale_buffer(execution_context& ctx, buffer<valuef> in, buffer_mut<valuef> out, literal<v3i> in_dim, literal<v3i> out_dim)
{
    using namespace single_source;

    valuei lid = value_impl::get_global_id(0);

    pin(lid);

    auto dim = out_dim.get();

    if_e(lid >= dim.x() * dim.y() * dim.z(), [&]{
        return_e();
    });

    v3i pos = get_coordinate(lid, dim);

    v3f lower_pos = get_scaled_coordinate_vec(pos, dim, in_dim.get());

    if_e(pos.x() == 0 || pos.y() == 0 || pos.z() == 0 ||
         pos.x() == dim.x() - 1 ||  pos.y() == dim.y() - 1 || pos.z() == dim.z() - 1, [&]{

        return_e();
    });

    ///trilinear interpolation
    as_ref(out[pos, dim]) = buffer_read_linear(in, lower_pos, in_dim.get());
}

inline
int init_laplace(cl::context ctx)
{
    cl::async_build_and_cache(ctx, [=]{
        return value_impl::make_function(upscale_buffer, "upscale");
    }, {"upscale"});
    return 0;
}

struct laplace_params
{
    buffer<valuef> u;
    valuef scale;
    v3i dim;
    v3i pos;
};

struct laplace_solver
{
    std::string kernel_name;

    template<typename T, typename U>
    void boot(cl::context ctx, T get_rhs, U arg_pack_unused, std::string kname)
    {
        static int _ = init_laplace(ctx);

        static_assert(std::is_base_of_v<value_impl::single_source::argument_pack, U>);

        kernel_name = kname;

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

            laplace_params params;
            params.u = inout;
            params.scale = lscale.get();
            params.dim = dim;
            params.pos = pos;

            valuef rhs = get_rhs(params, pack);

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

            /*if_e(pos.x() == 128 && pos.y() == 128 && pos.z() == 128, [&]{
                value_base se;
                se.type = value_impl::op::SIDE_EFFECT;
                se.abstract_value = "printf(\"%.23f\\n\"," + value_to_string(u) + ")";

                value_impl::get_context().add(se);
            });*/

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

    template<typename T>
    auto solve(cl::context ctx, cl::command_queue cqueue, float simulation_width, t3i dim, T&& data_getter)
    {
        cl::buffer ret(ctx);
        ret.alloc(sizeof(cl_float) * dim.x() * dim.y() * dim.z());

        auto get_u_at_dim = [&](t3i dim, float simulation_width, float relax, std::optional<std::tuple<cl::buffer, t3i>> u_old)
        {
            float scale = simulation_width / (dim.x() - 1);

            cl::buffer u_found(ctx);

            auto data = data_getter(dim, scale);

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
                    args.push_back(u_old_buf, u_found, u_old_dim, dim);

                    cqueue.exec("upscale", args, {dim.x() * dim.y() * dim.z()}, {128});
                }

                for(int i=0; i < 100000; i++)
                {
                    bool check = (i % 500) == 0;

                    if(check)
                        still_going.set_to_zero(cqueue);

                    cl::args args;
                    args.push_back(u_found);

                    data.push(args);

                    args.push_back(scale);
                    args.push_back(dim);
                    args.push_back(i);
                    args.push_back(still_going);
                    args.push_back(relax);

                    cqueue.exec(kernel_name, args, {dim.x() * dim.y() * dim.z()}, {128});

                    if(check)
                    {
                        printf("Checked at %i\n", i);

                        bool going = still_going.read<int>(cqueue).at(0) == 1;

                        if(!going)
                            break;
                    }
                }
            }

            return std::pair{u_found, data};
        };

        std::vector<t3i> dims;
        std::vector<float> relax;

        int max_refinement_levels = 5;

        ///generate a dims array which gets progressively larger, eg
        ///63, 95, 127, 197, 223
        ///is generated in reverse
        for(int i=0; i < max_refinement_levels; i++)
        {
            float div = pow(1.25, i + 1);
            ///exact params here are pretty unimportant
            float rel = mix(0.7f, 0.3f, (float)i / (max_refinement_levels-1));

            t3i next_dim = (t3i)((t3f)dim / div);

            if((next_dim.x() % 2) == 0)
                next_dim += (t3i){1,1,1};

            dims.insert(dims.begin(), next_dim);
            relax.insert(relax.begin(), rel);
        }

        dims.insert(dims.begin(), {51, 51, 51});
        relax.insert(relax.begin(), 0.3f);

        dims.push_back(dim);
        relax.push_back(0.8f);

        for(int i=0; i < (int)dims.size(); i++)
        {
            printf("Dims %i %f\n", dims[i].x(), relax[i]);
        }

        {
            std::optional<std::pair<cl::buffer, t3i>> last_u;

            for(int i=0; i < dims.size(); i++)
            {
                auto out = get_u_at_dim(dims[i], simulation_width, relax[i], last_u);

                auto& [last_buf, _] = out;

                last_u = {last_buf, dims[i]};

                if(i == (int)dims.size() - 1)
                    return out;
            }
        }

        assert(false);
    }
};

#endif // LAPLACE_HPP_INCLUDED

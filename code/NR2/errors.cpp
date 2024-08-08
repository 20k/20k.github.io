#include "errors.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include "bssn.hpp"
#include "derivatives.hpp"
#include "tensor_algebra.hpp"

std::string make_hamiltonian_error()
{
    auto func = [&](execution_context&,
                    bssn_args_mem<buffer<valuef>> args_in,
                    bssn_derivatives_mem<buffer<derivative_t>> derivatives,
                    buffer_mut<valuef> out,
                    literal<v3i> ldim, literal<valuef> scale) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        v3i pos = get_coordinate(lid, dim);

        if_e(pos.x() <= 2 || pos.x() >= dim.x() - 3 ||
             pos.y() <= 2 || pos.y() >= dim.y() - 3 ||
             pos.z() <= 2 || pos.z() >= dim.z() - 3, [&] {

            as_ref(out[pos, dim]) = valuef(0.f);
            return_e();
        });

        bssn_args args(pos, dim, args_in);
        bssn_derivatives derivs(pos, dim, derivatives);

        derivative_data d;
        d.pos = pos;
        d.dim = dim;
        d.scale = scale.get();

        valuef hamiltonian = calculate_hamiltonian_constraint(args, derivs, d);

        as_ref(out[pos, dim]) = hamiltonian;
    };

    return value_impl::make_function(func, "calculate_hamiltonian");
}

std::string make_cG_error(int idx)
{
    auto func = [&](execution_context&,
                    bssn_args_mem<buffer<valuef>> args_in,
                    bssn_derivatives_mem<buffer<derivative_t>> derivatives,
                    buffer_mut<valuef> out,
                    literal<v3i> ldim, literal<valuef> scale) {
                using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        v3i pos = get_coordinate(lid, dim);

        if_e(pos.x() <= 2 || pos.x() >= dim.x() - 3 ||
             pos.y() <= 2 || pos.y() >= dim.y() - 3 ||
             pos.z() <= 2 || pos.z() >= dim.z() - 3, [&] {

            as_ref(out[pos, dim]) = valuef(0.f);
            return_e();
        });

        bssn_args args(pos, dim, args_in);
        bssn_derivatives derivs(pos, dim, derivatives);

        inverse_metric<valuef, 3, 3> icY = args.cY.invert();
        pin(icY);

        tensor<valuef, 3, 3, 3> christoff2 = christoffel_symbols_2(icY, derivs.dcY);
        pin(christoff2);

        tensor<valuef, 3> calculated_cG;

        for(int i=0; i < 3; i++)
        {
            valuef sum = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    sum += icY[m, n] * christoff2[i, m, n];
                }
            }

            calculated_cG[i] = sum;
        }

        tensor<valuef, 3> Gi = args.cG - calculated_cG;

        as_ref(out[pos, dim]) = Gi[idx];
    };

    return value_impl::make_function(func, "calculate_cGi" + std::to_string(idx));
}

std::string make_momentum_error(int idx)
{
    auto func = [&](execution_context&,
                    bssn_args_mem<buffer<valuef>> args_in,
                    bssn_derivatives_mem<buffer<derivative_t>> derivatives,
                    buffer_mut<valuef> out,
                    literal<v3i> ldim, literal<valuef> scale) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        v3i pos = get_coordinate(lid, dim);

        if_e(pos.x() <= 2 || pos.x() >= dim.x() - 3 ||
             pos.y() <= 2 || pos.y() >= dim.y() - 3 ||
             pos.z() <= 2 || pos.z() >= dim.z() - 3, [&] {

            as_ref(out[pos, dim]) = valuef(0.f);
            return_e();
        });

        bssn_args args(pos, dim, args_in);

        derivative_data d;
        d.pos = pos;
        d.dim = dim;
        d.scale = scale.get();

        tensor<valuef, 3> Mi = calculate_momentum_constraint(args, d);

        as_ref(out[pos, dim]) = Mi[idx];
    };

    return value_impl::make_function(func, "calculate_Mi" + std::to_string(idx));
}

std::string make_global_sum()
{
     auto func = [&](execution_context&,
                     buffer<valuef> in,
                     buffer_mut<value<std::int64_t>> sum,
                     literal<valuei> num) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        if_e(lid >= num.get(), [&] {
            return_e();
        });

        valued as_double = ((valued)fabs(in[lid])) * pow(10., 8.);

        value<std::int64_t> as_uint = (value<std::int64_t>)as_double;

        sum.atom_add_e(0, as_uint);
    };

    return value_impl::make_function(func, "sum");
}

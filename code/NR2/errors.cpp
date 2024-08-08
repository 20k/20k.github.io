#include "errors.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include "bssn.hpp"
#include "derivatives.hpp"

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

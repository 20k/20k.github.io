#include "errors.hpp"
#include "../common/value2.hpp"
#include "../common/single_source.hpp"
#include <iostream>
#include "bssn.hpp"

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

        ///todo: genericise
        valuei x = lid % dim.x();
        valuei y = (lid / dim.x()) % dim.y();
        valuei z = lid / (dim.x() * dim.y());

        pin(x);
        pin(y);
        pin(z);

        v3i pos = {x, y, z};

        bssn_args args(pos, dim, args_in);
        bssn_derivatives derivs(pos, dim, derivatives);

        valuef hamiltonian = calculate_hamiltonian_constraint(args, derivs, scale.get());

        as_ref(out[pos, dim]) = hamiltonian;
    };

    return value_impl::make_function(func, "calculate_hamiltonian");
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

        valued as_double = ((valued)in[lid]) * pow(10., 8.);
    };

    return value_impl::make_function(func, "sum");
}

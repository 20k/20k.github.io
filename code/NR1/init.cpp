#include "init.hpp"
#include "bssn.hpp"
#include "../common/vec/dual.hpp"
#include "derivatives.hpp"
#include "tensor_algebra.hpp"

template<typename T>
using dual = dual_types::dual_v<T>;

auto diff_analytic(auto&& func, const v4f& position, int direction) {
    auto pinned = position;
    single_source::pin(pinned);

    m44f metric = func(pinned);

    tensor<valuef, 4, 4> differentiated;

    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            dual<value_base> as_dual = replay_value_base<dual<value_base>>(metric[i, j], [&](const value_base& in)
            {
                if(equivalent(in, pinned[direction]))
                    return dual<value_base>(in, in.make_constant_of_type(1.f));
                else
                    return dual<value_base>(in, in.make_constant_of_type(0.f));
            });

            differentiated[i, j].set_from_base(as_dual.dual);
        }
    }

    return differentiated;
}


auto wave_function = []<typename T>(const tensor<T, 4>& position)
{
    float A = 0.1f;
    float d = 1;

    auto H = A * sin(2 * std::numbers::pi_v<float> * (position.y() - position.x()) / d);

    metric<T, 4, 4> m;
    m[0, 0] = -1 * (1 - H);
    m[1, 1] = (1 - H);
    m[2, 2] = 1;
    m[3, 3] = 1;

    return m;
};

std::string make_initial_conditions()
{
    auto init = [&](execution_context&, bssn_args_mem<buffer_mut<valuef>> to_fill, literal<v3i> ldim, literal<valuef> scale) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        v3i pos = get_coordinate(lid, dim);

        v3f wpos = ((v3f)pos) * scale.get();

        metric<valuef, 4, 4> Guv = wave_function((v4f){0, wpos.x(), wpos.y(), wpos.z()});

        tensor<valuef, 4, 4, 4> dGuv;

        for(int k=0; k < 4; k++)
        {
            auto ldguv = diff_analytic(wave_function, (v4f){0, wpos.x(), wpos.y(), wpos.z()}, k);

            for(int i=0; i < 4; i++)
            {
                for(int j=0; j < 4; j++)
                {
                    dGuv[k, i, j] = ldguv[i, j];
                }
            }
        }

        metric<valuef, 3, 3> Yij;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                Yij[i, j] = Guv[i+1, j+1];
            }
        }

        tensor<valuef, 3, 3, 3> Yij_derivatives;

        for(int k=0; k < 3; k++)
        {
            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    Yij_derivatives[k, i, j] = dGuv[k+1, i+1, j+1];
                }
            }
        }

        tensor<valuef, 3, 3, 3> Yij_christoffel = christoffel_symbols_2(Yij.invert(), Yij_derivatives);

        pin(Yij_christoffel);

        tensor<valuef, 3> gB_lower;
        tensor<valuef, 3, 3> dgB_lower;

        for(int i=0; i < 3; i++)
        {
            gB_lower[i] = Guv[0, i+1];

            for(int k=0; k < 3; k++)
            {
                dgB_lower[k, i] = dGuv[k+1, 0, i+1];
            }
        }

        tensor<valuef, 3> gB = raise_index(gB_lower, Yij.invert(), 0);

        pin(gB);

        valuef gB_sum = sum_multiply(gB, gB_lower);

        ///g00 = nini - n^2
        ///g00 - nini = -n^2
        ///-g00 + nini = n^2
        ///n = sqrt(-g00 + nini)
        valuef gA = sqrt(-Guv[0, 0] + gB_sum);

        ///https://clas.ucdenver.edu/math-clinic/sites/default/files/attached-files/master_project_mach_.pdf 4-19a
        tensor<valuef, 3, 3> gBjDi = covariant_derivative_low_vec(gB_lower, dgB_lower, Yij_christoffel);

        pin(gBjDi);

        tensor<valuef, 3, 3> Kij;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                Kij[i, j] = (1/(2 * gA)) * (gBjDi[j, i] + gBjDi[i, j] - dGuv[0, i+1, j+1]);
            }
        }

        valuef W = pow(Yij.det(), -1/6.f);
        metric<valuef, 3, 3> cY = W*W * Yij;
        valuef K = trace(Kij, Yij.invert());

        tensor<valuef, 3, 3> cA = W*W * (Kij - (1.f/3.f) * Yij.to_tensor() * K);

        tensor<int, 2> index_table[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

        for(int i=0; i < 6; i++)
        {
            tensor<int, 2> idx = index_table[i];

            as_ref(to_fill.cY[i][lid]) = cY[idx.x(), idx.y()];
            as_ref(to_fill.cA[i][lid]) = cA[idx.x(), idx.y()];
        }

        for(int i=0; i < 3; i++)
        {
            as_ref(to_fill.cG[i][lid]) = valuef(0);
            as_ref(to_fill.gB[i][lid]) = gB[i];
        }

        as_ref(to_fill.gA[lid]) = gA;
        as_ref(to_fill.W[lid]) = W;
        as_ref(to_fill.K[lid]) = K;
    };

    return value_impl::make_function(init, "init");
}

std::string init_christoffel()
{
     auto init = [&](execution_context&, bssn_args_mem<buffer_mut<valuef>> to_fill, literal<v3i> ldim, literal<valuef> scale) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        v3i dim = ldim.get();

        if_e(lid >= dim.x() * dim.y() * dim.z(), [&] {
            return_e();
        });

        v3i pos = get_coordinate(lid, dim);

        metric<valuef, 3, 3> cY;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                int index_table[3][3] = {{0, 1, 2},
                                         {1, 3, 4},
                                         {2, 4, 5}};

                cY[i, j] = to_fill.cY[index_table[i][j]][pos, dim];
            }
        }

        auto icY = cY.invert();

        tensor<valuef, 3, 3, 3> dcY;

        for(int k=0; k < 3; k++)
        {
            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    dcY[k, i, j] = diff1(cY[i, j], k, scale.get());
                }
            }
        }

        auto christoff2 = christoffel_symbols_2(icY, dcY);

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

        for(int i=0; i < 3; i++)
        {
            as_ref(to_fill.cG[i][lid]) = calculated_cG[i];
        }
     };

     return value_impl::make_function(init, "init_christoffel");
}

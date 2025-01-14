#ifndef TENSOR_ALGEBRA_HPP_INCLUDED
#define TENSOR_ALGEBRA_HPP_INCLUDED

#include "../common/value2.hpp"
#include "derivatives.hpp"

template<typename T, int N, typename U>
inline
tensor<T, N, N> lie_derivative_weight(const tensor<T, N>& B, const tensor<T, N, N>& mT, const U& d)
{
    tensor<T, N, N> lie;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            T sum = 0;
            T sum2 = 0;

            for(int k=0; k < N; k++)
            {
                sum += B[k] * diff1(mT[i, j], k, d);
                sum += mT[i, k] * diff1(B[k], j, d);
                sum += mT[j, k] * diff1(B[k], i, d);
                sum2 += diff1(B[k], k, d);
            }

            lie[i, j] = sum - (2.f/3.f) * mT[i, j] * sum2;
        }
    }

    return lie;
}

///derivatives are passed in like dkMij
///https://en.wikipedia.org/wiki/Christoffel_symbols#Christoffel_symbols_of_the_second_kind_(symmetric_definition)
///the christoffel symbols are calculated normally in bssn with the conformal metric tensor
template<typename T, int N>
inline
tensor<T, N, N, N> christoffel_symbols_2(const inverse_metric<T, N, N>& inverse, const tensor<T, N, N, N>& derivatives)
{
    tensor<T, N, N, N> christoff;

    for(int i=0; i < N; i++)
    {
        for(int k=0; k < N; k++)
        {
            for(int l=0; l < N; l++)
            {
                T sum = 0;

                for(int m=0; m < N; m++)
                {
                    T local = 0;

                    local += derivatives[l, m, k];
                    local += derivatives[k, m, l];
                    local += -derivatives[m, k, l];

                    sum += local * inverse[i, m];
                }

                christoff[i, k, l] = T(0.5f) * sum;
            }
        }
    }

    return christoff;
}

template<typename T, int N>
inline
tensor<T, N, N, N> christoffel_symbols_1(const tensor<T, N, N, N>& derivatives)
{
    tensor<T, N, N, N> christoff;

    for(int c=0; c < N; c++)
    {
        for(int a=0; a < N; a++)
        {
            for(int b=0; b < N; b++)
            {
                christoff[c, a, b] = 0.5f * (derivatives[b, c, a] + derivatives[a, c, b] - derivatives[c, a, b]);
            }
        }
    }

    return christoff;
}

///https://en.wikipedia.org/wiki/Covariant_derivative#Covariant_derivative_by_field_type
///a partial derivative is a lower index vector
template<typename T, int N, typename U>
inline
tensor<T, N, N> double_covariant_derivative(const T& in, const tensor<T, N>& first_derivatives,
                                            const tensor<T, N, N, N>& christoff2, const U& d)
{
    tensor<T, N, N> lac;

    for(int a=0; a < N; a++)
    {
        for(int c=0; c < N; c++)
        {
            T sum = 0;

            for(int b=0; b < N; b++)
            {
                sum += christoff2[b, c, a] * diff1(in, b, d);
            }

            lac[a, c] = diff2(in, a, c, first_derivatives[a], first_derivatives[c], d) - sum;
        }
    }

    return lac;
}

///https://arxiv.org/pdf/gr-qc/9810065.pdf
template<typename T, int N>
inline
T trace(const tensor<T, N, N>& mT, const inverse_metric<T, N, N>& inverse)
{
    T ret = 0;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            ret = fma(inverse[i, j], mT[i, j], ret);
        }
    }

    return ret;
}

template<typename T, int N>
inline
tensor<T, N, N> trace_free(const tensor<T, N, N>& mT, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    tensor<T, N, N> TF;
    T t = trace(mT, inverse);

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            TF[i, j] = fma(-(1.f/3.f) * met[i, j], t, mT[i, j]);
        }
    }

    return TF;
}

///https://en.wikipedia.org/wiki/Covariant_derivative#Covariant_derivative_by_field_type
///for the tensor DcDa, this returns idx(a, c)
template<typename T, int N, typename U>
inline
tensor<T, N, N> covariant_derivative_low_vec(const tensor<T, N>& v_in, const tensor<T, N, N, N>& christoff2, const U& d)
{
    tensor<T, N, N> lac;

    for(int a=0; a < N; a++)
    {
        for(int c=0; c < N; c++)
        {
            T sum = 0;

            for(int b=0; b < N; b++)
            {
                sum += christoff2[b, c, a] * v_in[b];
            }

            lac[a, c] = diff1(v_in[a], c, d) - sum;
        }
    }

    return lac;
}

///https://en.wikipedia.org/wiki/Covariant_derivative#Covariant_derivative_by_field_type
///for the tensor DcDa, this returns idx(a, c)
template<typename T, int N>
inline
tensor<T, N, N> covariant_derivative_low_vec(const tensor<T, N>& v_in, const tensor<T, N, N>& d_v_in, const tensor<T, N, N, N>& christoff2)
{
    tensor<T, N, N> lac;

    for(int a=0; a < N; a++)
    {
        for(int c=0; c < N; c++)
        {
            T sum = 0;

            for(int b=0; b < N; b++)
            {
                sum += christoff2[b, c, a] * v_in[b];
            }

            lac[a, c] = d_v_in[c, a] - sum;
        }
    }

    return lac;
}

template<typename T, int N>
inline
tensor<T, N, N> covariant_derivative_high_vec(const tensor<T, N>& v_in, const tensor<T, N, N>& derivatives, const tensor<T, N, N, N>& christoff2)
{
    tensor<T, N, N> lab;

    for(int a=0; a < N; a++)
    {
        for(int b=0; b < N; b++)
        {
            T sum = 0;

            for(int c=0; c < N; c++)
            {
                sum += christoff2[a, b, c] * v_in[c];
            }

            lab[a, b] = derivatives[b, a] + sum;
        }
    }

    return lab;
}

inline
tensor<valuef, 3, 3, 3> get_eijk()
{
    auto eijk_func = [](int i, int j, int k)
    {
        if((i == 0 && j == 1 && k == 2) || (i == 1 && j == 2 && k == 0) || (i == 2 && j == 0 && k == 1))
            return 1;

        if(i == j || j == k || k == i)
            return 0;

        return -1;
    };

    tensor<valuef, 3, 3, 3> eijk;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                eijk[i, j, k] = eijk_func(i, j, k);
            }
        }
    }

    return eijk;
}

template<typename T, int N>
inline
tensor<T, N, N> raise_both(const tensor<T, N, N>& mT, const inverse_metric<T, N, N>& inverse)
{
    return raise_index(raise_index(mT, inverse, 0), inverse, 1);
}

template<typename T, int N>
inline
tensor<T, N, N> lower_both(const tensor<T, N, N>& mT, const metric<T, N, N>& met)
{
    return lower_index(lower_index(mT, met, 0), met, 1);
}

///https://arxiv.org/pdf/1503.08455.pdf (8)
inline
metric<valuef, 4, 4> calculate_real_metric(const metric<valuef, 3, 3>& adm, const valuef& gA, const tensor<valuef, 3>& gB)
{
    tensor<valuef, 3> lower_gB = lower_index(gB, adm, 0);

    metric<valuef, 4, 4> ret;

    valuef gB_sum = 0;

    for(int i=0; i < 3; i++)
    {
        gB_sum = gB_sum + lower_gB[i] * gB[i];
    }

    ///https://arxiv.org/pdf/gr-qc/0703035.pdf 4.43
    ret[0, 0] = -gA * gA + gB_sum;

    ///latin indices really run from 1-4
    for(int i=1; i < 4; i++)
    {
        ///https://arxiv.org/pdf/gr-qc/0703035.pdf 4.45
        ret[i, 0] = lower_gB[i - 1];
        ret[0, i] = ret[i, 0]; ///symmetry
    }

    for(int i=1; i < 4; i++)
    {
        for(int j=1; j < 4; j++)
        {
            ret[i, j] = adm[i-1, j-1];
        }
    }

    return ret;
}

///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses (3.33)
inline
tensor<valuef, 4> get_adm_hypersurface_normal_raised(const valuef& gA, const tensor<valuef, 3>& gB)
{
    return {1/gA, -gB.idx(0)/gA, -gB.idx(1)/gA, -gB.idx(2)/gA};
}

inline
tensor<valuef, 4> get_adm_hypersurface_normal_lowered(const valuef& gA)
{
    return {-gA, 0, 0, 0};
}

#endif // TENSOR_ALGEBRA_HPP_INCLUDED

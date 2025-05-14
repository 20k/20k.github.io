#ifndef STDMATH_HPP_INCLUDED
#define STDMATH_HPP_INCLUDED

#include <cmath>
#include <assert.h>
#include <stdexcept>
#include <algorithm>

namespace stdmath
{
    constexpr
    auto usin = []<typename T>(const T& in)
    {
        using std::sin;
        return sin(in);
    };

    constexpr
    auto ucos = []<typename T>(const T& in)
    {
        using std::cos;
        return cos(in);
    };

    constexpr
    auto utan = []<typename T>(const T& in)
    {
        using std::tan;
        return tan(in);
    };

    constexpr
    auto op_plus = []<typename T>(const T& v1, const T& v2)
    {
        return v1 + v2;
    };

    constexpr
    auto op_minus = []<typename T>(const T& v1, const T& v2)
    {
        return v1 - v2;
    };

    constexpr
    auto op_unary_minus = []<typename T>(const T& v1)
    {
        return -v1;
    };

    constexpr
    auto op_multiply = []<typename T>(const T& v1, const T& v2)
    {
        return v1 * v2;
    };

    constexpr
    auto op_divide = []<typename T>(const T& v1, const T& v2)
    {
        return v1 / v2;
    };

    constexpr
    auto op_lt = []<typename T>(const T& v1, const T& v2)
    {
        return v1 < v2;
    };

    constexpr
    auto op_lte = []<typename T>(const T& v1, const T& v2)
    {
        return v1 <= v2;
    };

    constexpr
    auto op_eq = []<typename T>(const T& v1, const T& v2)
    {
        return v1 == v2;
    };

    constexpr
    auto op_neq = []<typename T>(const T& v1, const T& v2)
    {
        return v1 != v2;
    };

    constexpr
    auto op_gt = []<typename T>(const T& v1, const T& v2)
    {
        return v1 > v2;
    };

    constexpr
    auto op_gte = []<typename T>(const T& v1, const T& v2)
    {
        return v1 >= v2;
    };

    constexpr
    auto op_not = []<typename T>(const T& v1)
    {
        return !v1;
    };

    constexpr
    auto usign = []<typename T>(T in)
    {
        if constexpr(std::is_same_v<T, bool>)
        {
            assert(false);
            return false;
        }
        else
        {
            if(in == T(-0.0))
                return T(-0.0);

            if(in == T(0.0))
                return T(0.0);

            if(in > 0)
                return T(1);

            if(in < 0)
                return T(-1);

            using std::isnan;

            if(isnan(in))
                return T(0);
        }

        throw std::runtime_error("Bad sign function");
    };

    constexpr
    auto uisfinite = []<typename T>(const T& in)
    {
        using std::isfinite;

        if constexpr(std::is_arithmetic_v<T>)
            return (int)isfinite(in);
        else
            return isfinite(in);
    };

    constexpr
    auto ufloor = []<typename T>(const T& in)
    {
        using std::floor;
        return floor(in);
    };

    constexpr
    auto uceil = []<typename T>(const T& in)
    {
        using std::ceil;
        return ceil(in);
    };

    constexpr
    auto uround = []<typename T>(const T& in)
    {
        using std::round;
        return round(in);
    };

    constexpr
    auto ufmod = []<typename T>(const T& v1, const T& v2)
    {
        using std::fmod;

        if constexpr(std::is_integral_v<T>)
            return v1 % v2;
        else
            return fmod(v1, v2);
    };

    constexpr
    auto ufabs = []<typename T>(const T& in)
    {
        if constexpr(std::is_unsigned_v<T> || std::is_same_v<T, bool>)
            return in;
        else
        {
            using std::fabs;
            using std::abs;

            if constexpr(std::is_integral_v<T>)
            {
                static_assert(std::is_same_v<decltype(abs(in)), T>);
                return abs(in);
            }
            else
            {
                static_assert(std::is_same_v<decltype(fabs(in)), T>);
                return fabs(in);
            }
        }
    };

    constexpr
    auto usqrt = []<typename T>(const T& in)
    {
        using std::sqrt;

        if constexpr(std::is_integral_v<T>)
            return sqrt((float)in);
        else
            return sqrt(in);
    };

    constexpr
    auto uinverse_sqrt = []<typename T>(const T& in)
    {
        return 1/usqrt(in);
    };

    template<typename U>
    constexpr
    auto ucast = []<typename T>(const T& in)
    {
        return (U)in;
    };

    constexpr
    auto ulog = []<typename T>(const T& in)
    {
        using std::log;

        if constexpr(std::is_integral_v<T>)
            return log((float)in);
        else
            return log(in);
    };

    constexpr
    auto ulog2 = []<typename T>(const T& in)
    {
        using std::log2;

        if constexpr(std::is_integral_v<T>)
            return log2((float)in);
        else
            return log2(in);
    };

    constexpr
    auto uternary = []<typename T, typename U>(const T& condition, const U& if_true, const U& if_false)
    {
        if constexpr(std::is_arithmetic_v<U>)
            return condition ? if_true : if_false;
        else
            return ternary(condition, if_true, if_false);
    };

    constexpr
    auto usinh = []<typename T>(const T& in)
    {
        using std::sinh;

        if constexpr(std::is_integral_v<T>)
            return sinh((float)in);
        else
            return sinh(in);
    };

    constexpr
    auto ucosh = []<typename T>(const T& in)
    {
        using std::cosh;

        if constexpr(std::is_integral_v<T>)
            return cosh((float)in);
        else
            return cosh(in);
    };

    constexpr
    auto utanh = []<typename T>(const T& in)
    {
        using std::tanh;

        if constexpr(std::is_integral_v<T>)
            return tanh((float)in);
        else
            return tanh(in);
    };

    constexpr
    auto uasinh = []<typename T>(const T& in)
    {
        using std::asinh;

        if constexpr(std::is_integral_v<T>)
            return asinh((float)in);
        else
            return asinh(in);
    };

    constexpr
    auto uacosh = []<typename T>(const T& in)
    {
        using std::acosh;

        if constexpr(std::is_integral_v<T>)
            return acosh((float)in);
        else
            return acosh(in);
    };

    constexpr
    auto uatanh = []<typename T>(const T& in)
    {
        using std::atanh;

        if constexpr(std::is_integral_v<T>)
            return atanh((float)in);
        else
            return atanh(in);
    };

    constexpr
    auto uasin = []<typename T>(const T& in)
    {
        using std::asin;

        if constexpr(std::is_integral_v<T>)
            return asin((float)in);
        else
            return asin(in);
    };

    constexpr
    auto uacos = []<typename T>(const T& in)
    {
        using std::acos;

        if constexpr(std::is_integral_v<T>)
            return acos((float)in);
        else
            return acos(in);
    };

    constexpr
    auto uatan = []<typename T>(const T& in)
    {
        using std::atan;

        if constexpr(std::is_integral_v<T>)
            return atan((float)in);
        else
            return atan(in);
    };

    constexpr
    auto uatan2 = []<typename T>(const T& y, const T& x)
    {
        using std::atan2;

        if constexpr(std::is_integral_v<T>)
            return atan2((float)y, (float)x);
        else
            return atan2(y, x);
    };

    constexpr
    auto umin = []<typename T>(const T& v1, const T& v2)
    {
        using std::min;

        return min(v1, v2);
    };

    constexpr
    auto umax = []<typename T>(const T& v1, const T& v2)
    {
        using std::max;

        return max(v1, v2);
    };

    constexpr
    auto uclamp = []<typename T>(const T& v1, const T& v2, const T& v3)
    {
        using std::clamp;

        return clamp(v1, v2, v3);
    };

    constexpr
    auto upow = []<typename T>(const T& v1, const T& v2)
    {
        using std::pow;

        return pow(v1, v2);
    };

    constexpr
    auto uexp = []<typename T>(const T& v1)
    {
        using std::exp;

        if constexpr(std::is_integral_v<T>)
            return exp((float)v1);
        else
            return exp(v1);
    };

    constexpr
    auto ufma = []<typename T>(const T& v1, const T& v2, const T& v3)
    {
        if constexpr(std::is_integral_v<T>)
            return v1 * v2 + v3;
        else
        {
            using std::fma;

            static_assert(std::is_same_v<decltype(fma(v1, v2, v3)), T>);

            return fma(v1, v2, v3);
        }
    };
}

#endif // STDMATH_HPP_INCLUDED

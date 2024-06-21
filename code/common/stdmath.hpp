#ifndef STDMATH_HPP_INCLUDED
#define STDMATH_HPP_INCLUDED

#include <cmath>

namespace stdmath
{
    auto usin = []<typename T>(const T& in)
    {
        using std::sin;
        return sin(in);
    };

    auto ucos = []<typename T>(const T& in)
    {
        using std::cos;
        return cos(in);
    };

    auto utan = []<typename T>(const T& in)
    {
        using std::tan;
        return tan(in);
    };

    auto op_plus = []<typename T>(const T& v1, const T& v2)
    {
        return v1 + v2;
    };

    auto op_minus = []<typename T>(const T& v1, const T& v2)
    {
        return v1 - v2;
    };

    auto op_unary_minus = []<typename T>(const T& v1)
    {
        return -v1;
    };

    auto op_multiply = []<typename T>(const T& v1, const T& v2)
    {
        return v1 * v2;
    };

    auto op_divide = []<typename T>(const T& v1, const T& v2)
    {
        return v1 / v2;
    };

    auto op_lt = []<typename T>(const T& v1, const T& v2)
    {
        return v1 < v2;
    };

    auto op_lte = []<typename T>(const T& v1, const T& v2)
    {
        return v1 <= v2;
    };

    auto op_eq = []<typename T>(const T& v1, const T& v2)
    {
        return v1 == v2;
    };

    auto op_gt = []<typename T>(const T& v1, const T& v2)
    {
        return v1 > v2;
    };

    auto op_gte = []<typename T>(const T& v1, const T& v2)
    {
        return v1 >= v2;
    };

    auto usign = []<typename T>(T in)
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

        throw std::runtime_error("Bad sign function");
    };

    auto uisfinite = []<typename T>(const T& in)
    {
        using std::isfinite;

        if constexpr(std::is_arithmetic_v<T>)
            return (int)isfinite(in);
        else
            return isfinite(in);
    };

    auto ufloor = []<typename T>(const T& in)
    {
        using std::floor;
        return floor(in);
    };

    auto uceil = []<typename T>(const T& in)
    {
        using std::ceil;
        return ceil(in);
    };

    auto ufmod = []<typename T>(const T& v1, const T& v2)
    {
        using std::fmod;

        if constexpr(std::is_integral_v<T>)
            return v1 % v2;
        else
            return fmod(v1, v2);
    };

    auto ufabs = []<typename T>(const T& in)
    {
        using std::fabs;
        using std::abs;

        if constexpr(std::is_integral_v<T>)
            return abs(in);
        else
            return fabs(in);
    };

    auto usqrt = []<typename T>(const T& in)
    {
        using std::sqrt;

        if constexpr(std::is_integral_v<T>)
            return sqrt((float)in);
        else
            return sqrt(in);
    };

    auto uinverse_sqrt = []<typename T>(const T& in)
    {
        return 1/usqrt(in);
    };

    template<typename U>
    auto ucast = []<typename T>(const T& in)
    {
        return (U)in;
    };

    auto ulog = []<typename T>(const T& in)
    {
        using std::log;

        if constexpr(std::is_integral_v<T>)
            return log((float)in);
        else
            return log(in);
    };

    auto uternary = []<typename T, typename U>(const T& condition, const U& if_true, const U& if_false)
    {
        if constexpr(std::is_arithmetic_v<U>)
            return condition ? if_true : if_false;
        else
            return ternary(condition, if_true, if_false);
    };

    auto uatan = []<typename T>(const T& in)
    {
        using std::atan;

        if constexpr(std::is_integral_v<T>)
            return atan((float)in);
        else
            return atan(in);
    };

    auto uatan2 = []<typename T>(const T& y, const T& x)
    {
        using std::atan2;

        if constexpr(std::is_integral_v<T>)
            return atan2((float)y, (float)x);
        else
            return atan2(y, x);
    };

    auto umin = []<typename T>(const T& v1, const T& v2)
    {
        using std::min;

        return min(v1, v2);
    };

    auto umax = []<typename T>(const T& v1, const T& v2)
    {
        using std::max;

        return max(v1, v2);
    };
}

#endif // STDMATH_HPP_INCLUDED

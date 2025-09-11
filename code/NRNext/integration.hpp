#ifndef INTEGRATION_HPP_INCLUDED
#define INTEGRATION_HPP_INCLUDED

inline
float no_opt(float v)
{
    return v;
}

template<typename T, typename U>
inline
auto integrate_1d_trapezoidal(const T& func, int n, const U& upper, const U& lower)
{
    using namespace single_source;

    using variable_type = decltype(func(U()));

    variable_type sum = 0;

    auto lmix = [&](auto& a, auto& b, auto& t)
    {
        auto imx = no_opt(1-t);
        auto imimx = no_opt(1-imx);

        auto p1 = no_opt(imx * a);
        auto p2 = no_opt(imimx * b);

        return no_opt(p1 + p2);
    };

    for(int k=1; k < n; k++)
    {
        float frac = (float)k/n;

        auto coordinate = lmix(lower, upper, frac);

        //auto coordinate = lower + k * (upper - lower) / n;

        auto val = func(coordinate);

        sum += val;
    }

    return ((upper - lower) / n) * (0.5f * (func(lower) + func(upper)) + sum);
}

template<typename T, typename U>
inline
auto integrate_3d_trapezoidal(const T& func, int n, const U& upper, const U& lower)
{
    auto z_integral = [&](auto z)
    {
        auto y_integral = [&](auto y)
        {
            auto x_integral = [&](auto x)
            {
                return func(x,y,z);
            };

            return integrate_1d_trapezoidal(x_integral, n, upper[0], lower[0]);
        };

        return integrate_1d_trapezoidal(y_integral, n, upper[1], lower[1]);
    };

    return integrate_1d_trapezoidal(z_integral, n, upper[2], lower[2]);
}

#endif // INTEGRATION_HPP_INCLUDED

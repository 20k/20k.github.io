#ifndef INTEGRATION_HPP_INCLUDED
#define INTEGRATION_HPP_INCLUDED

///todo: fixme args
template<typename T, typename U>
inline
auto integrate_1d_trapezoidal(const T& func, int n, const U& upper, const U& lower)
{
    using variable_type = decltype(func(U()));

    variable_type sum = 0;

    for(int k=1; k < n; k++)
    {
        auto coordinate = lower + k * (upper - lower) / n;

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

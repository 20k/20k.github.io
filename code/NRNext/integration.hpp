#ifndef INTEGRATION_HPP_INCLUDED
#define INTEGRATION_HPP_INCLUDED

#include <span>

template<typename T>
inline
T symmetric_sum(std::span<T> in)
{
    if(in.size() == 0)
        return T{};

    if(in.size() == 1)
        return in.front();

    if(in.size() == 2)
        return in[0] + in[1];

    //odd
    if((in.size() % 2) != 0)
    {
        int middle = (in.size() - 1) / 2;

        std::span left = in.subspan(0, middle);
        std::span right = in.subspan(middle + 1);

        return symmetric_sum(left) + symmetric_sum(right) + in[middle];
    }
    //even
    else
    {
        int middle = in.size() / 2;

        std::span left = in.subspan(0, middle);
        std::span right = in.subspan(middle);

        return symmetric_sum(left) + symmetric_sum(right);
    }
}

///todo: fixme args
template<typename T, typename U>
inline
auto integrate_1d_trapezoidal(const T& func, int n, const U& upper, const U& lower)
{
    using variable_type = decltype(func(U()));

    std::vector<variable_type> sum;

    for(int k=1; k < n; k++)
    {
        auto coordinate = lower + k * (upper - lower) / n;

        sum.push_back(func(coordinate));
    }

    return ((upper - lower) / n) * (0.5f * (func(lower) + func(upper)) + symmetric_sum(std::span{sum}));
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

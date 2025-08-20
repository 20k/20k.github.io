#ifndef INTERPOLATION_HPP_INCLUDED
#define INTERPOLATION_HPP_INCLUDED

#include "../common/single_source.hpp"

template<typename T, typename... U>
inline
auto function_trilinear(T&& func, v3f pos, U&&... args)
{
    using namespace single_source;

    v3f floored = floor(pos);
    pin(floored);
    v3f frac = pos - floored;
    pin(frac);

    v3i ipos = (v3i)floored;

    auto c000 = func(ipos + (v3i){0,0,0}, std::forward<U>(args)...);
    auto c100 = func(ipos + (v3i){1,0,0}, std::forward<U>(args)...);

    auto c010 = func(ipos + (v3i){0,1,0}, std::forward<U>(args)...);
    auto c110 = func(ipos + (v3i){1,1,0}, std::forward<U>(args)...);

    auto c001 = func(ipos + (v3i){0,0,1}, std::forward<U>(args)...);
    auto c101 = func(ipos + (v3i){1,0,1}, std::forward<U>(args)...);

    auto c011 = func(ipos + (v3i){0,1,1}, std::forward<U>(args)...);
    auto c111 = func(ipos + (v3i){1,1,1}, std::forward<U>(args)...);

    auto lmix = [&](auto& a, auto& b, auto& t)
    {
        auto imx = no_opt(1-t);
        auto imimx = no_opt(1-imx);

        auto p1 = no_opt(imx * a);
        auto p2 = no_opt(imimx * b);

        return no_opt(p1 + p2);
    };

    auto c00 = lmix(c000, c100, frac.x());
    auto c01 = lmix(c010, c110, frac.x());

    auto c10 = lmix(c001, c101, frac.x());
    auto c11 = lmix(c011, c111, frac.x());

    auto c0 = lmix(c00, c01, frac.y());
    auto c1 = lmix(c10, c11, frac.y());

    return lmix(c0, c1, frac.z());
}

template<typename T>
inline
auto function_quadlinear(T&& func, v4f pos)
{
    v4f floored = floor(pos);
    v4i ipos = (v4i)floored;

    v4f frac = pos - floored;

    auto a000 = func(ipos + (v4i){0,0,0,0});
    auto a100 = func(ipos + (v4i){1,0,0,0});

    auto a010 = func(ipos + (v4i){0,1,0,0});
    auto a110 = func(ipos + (v4i){1,1,0,0});

    auto a001 = func(ipos + (v4i){0,0,1,0});
    auto a101 = func(ipos + (v4i){1,0,1,0});

    auto a011 = func(ipos + (v4i){0,1,1,0});
    auto a111 = func(ipos + (v4i){1,1,1,0});

    auto a00 = mix(a000, a100, frac.x());
    auto a01 = mix(a010, a110, frac.x());

    auto a10 = mix(a001, a101, frac.x());
    auto a11 = mix(a011, a111, frac.x());

    auto a0 = mix(a00, a01, frac.y());
    auto a1 = mix(a10, a11, frac.y());

    auto linear_1 = mix(a0, a1, frac.z());

    auto c000 = func(ipos + (v4i){0,0,0,1});
    auto c100 = func(ipos + (v4i){1,0,0,1});

    auto c010 = func(ipos + (v4i){0,1,0,1});
    auto c110 = func(ipos + (v4i){1,1,0,1});

    auto c001 = func(ipos + (v4i){0,0,1,1});
    auto c101 = func(ipos + (v4i){1,0,1,1});

    auto c011 = func(ipos + (v4i){0,1,1,1});
    auto c111 = func(ipos + (v4i){1,1,1,1});

    auto c00 = mix(c000, c100, frac.x());
    auto c01 = mix(c010, c110, frac.x());

    auto c10 = mix(c001, c101, frac.x());
    auto c11 = mix(c011, c111, frac.x());

    auto c0 = mix(c00, c01, frac.y());
    auto c1 = mix(c10, c11, frac.y());

    auto linear_2 = mix(c0, c1, frac.z());

    return mix(linear_1, linear_2, frac.w());
    //return linear_1 - frac.w() * (linear_1 - linear_2);
}

#endif // INTERPOLATION_HPP_INCLUDED

#ifndef INTERPOLATION_HPP_INCLUDED
#define INTERPOLATION_HPP_INCLUDED

#include "../common/single_source.hpp"

template<typename T>
inline
auto function_trilinear(T&& func, v3f pos)
{
    v3f floored = floor(pos);
    v3f frac = pos - floored;

    v3i ipos = (v3i)floored;

    auto c000 = func(ipos + (v3i){0,0,0});
    auto c100 = func(ipos + (v3i){1,0,0});

    auto c010 = func(ipos + (v3i){0,1,0});
    auto c110 = func(ipos + (v3i){1,1,0});

    auto c001 = func(ipos + (v3i){0,0,1});
    auto c101 = func(ipos + (v3i){1,0,1});

    auto c011 = func(ipos + (v3i){0,1,1});
    auto c111 = func(ipos + (v3i){1,1,1});

    //https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0811r2.html
    auto lmix = [&](auto& a, auto& b, auto& t)
    {
        //return i1 + a * (i2 - i1);

        //return mix(a, b, t);

        return t * b + (1 - t) * a;

        //return i1 * (1-a) + i2 * a;
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
auto function_trilinear_precise_old(T&& func, v3f pos)
{
    using namespace single_source;

    v3f floored = floor(pos);
    pin(floored);
    v3f frac = pos - floored;
    pin(frac);

    v3i ipos = (v3i)floored;

    auto c000 = func(ipos + (v3i){0,0,0});
    auto c100 = func(ipos + (v3i){1,0,0});

    auto c010 = func(ipos + (v3i){0,1,0});
    auto c110 = func(ipos + (v3i){1,1,0});

    auto c001 = func(ipos + (v3i){0,0,1});
    auto c101 = func(ipos + (v3i){1,0,1});

    auto c011 = func(ipos + (v3i){0,1,1});
    auto c111 = func(ipos + (v3i){1,1,1});

    //https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0811r2.html
    auto lmix = [&](auto& a, auto& b, auto& t)
    {
        /*auto c1 = t * b;
        pin(c1);

        auto c2 = (1-t);
        pin(c2);

        auto c3 = c2 * a;
        pin(c3);

        auto p1 = c1 + c3;
        pin(p1);

        auto diff = b - a;
        pin(diff);

        auto tdiff = t * diff;
        pin(tdiff);

        auto x = a + tdiff;
        pin(x);

        auto cond_1 = a <= 0 && b >= 0 || a >= 0 && b <= 0;
        auto cond_2 = (b > a) == (t > 1);

        auto c2_selected = ternary(cond_2, max(b, x), min(b, x));

        auto c3_selected = ternary(t == valuef(1), b, c2_selected);

        return ternary(cond_1, p1, c3_selected);*/

        auto imx = 1-t;
        pin(imx);
        auto imimx = 1-imx;
        pin(imimx);

        auto p1 = imx * a;
        pin(p1);
        auto p2 = imimx * b;
        pin(p2);

        auto out = p1 + p2;
        pin(out);
        return out;

        //return p1 + p2;

        //return ternary(cond_1, p1, c2_selected);

        //return i1 + a * (i2 - i1);

        //return i1 * (1-a) + i2 * a;
    };

    auto c00 = lmix(c000, c100, frac.x());
    auto c01 = lmix(c010, c110, frac.x());

    auto c10 = lmix(c001, c101, frac.x());
    auto c11 = lmix(c011, c111, frac.x());

    auto c0 = lmix(c00, c01, frac.y());
    auto c1 = lmix(c10, c11, frac.y());

    return lmix(c0, c1, frac.z());
}

template<typename T, typename U>
inline
auto linear_1d(std::array<T, 4> vals, U frac)
{
    using namespace single_source;

    pin(vals[0]);
    pin(vals[1]);
    pin(vals[2]);
    pin(vals[3]);
    pin(frac);

    auto x = frac;

    auto a = vals[0];
    auto b = vals[1];
    auto c = vals[2];
    auto d = vals[3];

    auto a3 = -(a/6) * pow(x, 3);
    auto b3 = (b/2) * pow(x, 3);
    auto c3 = -(c/2) * pow(x, 3);
    auto d3 = (d/6) * pow(x, 3);

    auto a2 = (a/2) * pow(x, 2);
    auto b2 = -b * pow(x, 2);
    auto c2 = (c/2) * pow(x, 2);
    //auto d2 = 0;

    auto a1 = -(a/3) * x;
    auto b1 = -(b/2) * x;
    auto c1 = c * x;
    auto d1 = -(d/6) * x;

    auto cst = b;

    pin(a3);
    pin(b3);
    pin(c3);
    pin(d3);
    pin(a2);
    pin(b2);
    pin(c2);
    //pin(d2);
    pin(a1);
    pin(b1);
    pin(c1);
    pin(d1);
    pin(cst);

    return ((a3 + d3) + (b3 + c3)) + ((b2 + c2) + a2) + ((a1 + d1) + (b1 + c1)) + cst;
}

template<typename T>
inline
auto function_trilinear_precise(T&& func, v3f pos)
{
    return function_trilinear(func, pos);

    #ifdef BICUBIC
    using namespace single_source;

    v3f floored = floor(pos);
    pin(floored);
    v3f frac = pos - floored;
    pin(frac);

    auto t = [func](v3i ipos, v3f frac)
    {
        v3i offset = {0, 0, 1};

        auto cm1 = func(ipos - offset);
        auto c0 = func(ipos);

        auto cp1 = func(ipos + offset);
        auto cp2 = func(ipos + 2 * offset);

        return linear_1d(std::array{cm1, c0, cp1, cp2}, frac.z());
    };

    auto u = [t](v3i ipos, v3f frac)
    {
        v3i offset = {0, 1, 0};

        auto cm1 = t(ipos - offset, frac);
        auto c0 = t(ipos, frac);

        auto cp1 = t(ipos + offset, frac);
        auto cp2 = t(ipos + 2 * offset, frac);

        return linear_1d(std::array{cm1, c0, cp1, cp2}, frac.y());
    };

    auto f = [u](v3i ipos, v3f frac)
    {
        v3i offset = {1, 0, 0};

        auto cm1 = u(ipos - offset, frac);
        auto c0 = u(ipos, frac);

        auto cp1 = u(ipos + offset, frac);
        auto cp2 = u(ipos + 2 * offset, frac);

        return linear_1d(std::array{cm1, c0, cp1, cp2}, frac.x());
    };

    return f((v3i)floored, frac);
    #endif
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

    /*auto a00 = a000 - frac.x() * (a000 - a100);
    auto a01 = a001 - frac.x() * (a001 - a101);

    auto a10 = a010 - frac.x() * (a010 - a110);
    auto a11 = a011 - frac.x() * (a011 - a111);

    auto a0 = a00 - frac.y() * (a00 - a10);
    auto a1 = a01 - frac.y() * (a01 - a11);

    auto linear_1 = a0 - frac.z() * (a0 - a1);*/

    auto c000 = func(ipos + (v4i){0,0,0,1});
    auto c100 = func(ipos + (v4i){1,0,0,1});

    auto c010 = func(ipos + (v4i){0,1,0,1});
    auto c110 = func(ipos + (v4i){1,1,0,1});

    auto c001 = func(ipos + (v4i){0,0,1,1});
    auto c101 = func(ipos + (v4i){1,0,1,1});

    auto c011 = func(ipos + (v4i){0,1,1,1});
    auto c111 = func(ipos + (v4i){1,1,1,1});

    /*auto c00 = c000 - frac.x() * (c000 - c100);
    auto c01 = c001 - frac.x() * (c001 - c101);

    auto c10 = c010 - frac.x() * (c010 - c110);
    auto c11 = c011 - frac.x() * (c011 - c111);

    auto c0 = c00 - frac.y() * (c00 - c10);
    auto c1 = c01 - frac.y() * (c01 - c11);

    auto linear_2 = c0 - frac.z() * (c0 - c1);*/

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

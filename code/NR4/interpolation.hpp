#ifndef INTERPOLATION_HPP_INCLUDED
#define INTERPOLATION_HPP_INCLUDED

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

    ///numerically symmetric across the centre of dim
    auto c00 = c000 - frac.x() * (c000 - c100);
    auto c01 = c001 - frac.x() * (c001 - c101);

    auto c10 = c010 - frac.x() * (c010 - c110);
    auto c11 = c011 - frac.x() * (c011 - c111);

    auto c0 = c00 - frac.y() * (c00 - c10);
    auto c1 = c01 - frac.y() * (c01 - c11);

    return c0 - frac.z() * (c0 - c1);
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

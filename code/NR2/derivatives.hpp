#ifndef DERIVATIVES_HPP_INCLUDED
#define DERIVATIVES_HPP_INCLUDED

#include "../common/value2.hpp"
#include "../common/single_source.hpp"

using valuef = value<float>;
using valuei = value<int>;
using v3i = tensor<value<int>, 3>;

struct derivative_data
{
    v3i pos;
    v3i dim;
    valuef scale;
};

valuei distance_to_boundary(valuei pos, valuei dim);
valuei distance_to_boundary(v3i pos, v3i dim);

//1st derivative
valuef diff1(const valuef& in, int direction, const derivative_data& d);
//2nd derivative
valuef diff2(const valuef& in, int idx, int idy, const valuef& dx, const valuef& dy, const derivative_data& d);

valuef diff2nd(const valuef& in, int idx);
valuef diff4th(const valuef& in, int idx);
//6th derivative with second order accuracy
valuef diff6th(const valuef& in, int idx);
valuef diff8th(const valuef& in, int idx);
valuef diff10th(const valuef& in, int idx);

valuef diff1_boundary(single_source::buffer<valuef> in, int direction, const valuef& scale, v3i pos, v3i dim);

#endif // DERIVATIVES_HPP_INCLUDED

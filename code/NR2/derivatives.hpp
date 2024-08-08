#ifndef DERIVATIVES_HPP_INCLUDED
#define DERIVATIVES_HPP_INCLUDED

#include "../common/value2.hpp"
#include "../common/single_source.hpp"

using valuef = value<float>;
using v3i = tensor<value<int>, 3>;

//1st derivative
valuef diff1(const valuef& in, int direction, const valuef& scale);
//2nd derivative
valuef diff2(const valuef& in, int idx, int idy, const valuef& dx, const valuef& dy, const valuef& scale);
//6th derivative with second order accuracy
valuef diff6th(const valuef& in, int idx, const valuef& scale);

valuef diff1_boundary(single_source::buffer<valuef> in, int direction, const valuef& scale, v3i pos, v3i dim);

#endif // DERIVATIVES_HPP_INCLUDED

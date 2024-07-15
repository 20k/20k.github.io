#ifndef DERIVATIVES_HPP_INCLUDED
#define DERIVATIVES_HPP_INCLUDED

#include "../common/value2.hpp"

using valuef = value<float>;

//1st derivative
valuef diff1(const valuef& in, int direction, const valuef& scale);
//2nd derivative
valuef diff2(const valuef& in, int idx, int idy, const valuef& dx, const valuef& dy, const valuef& scale);
//6th derivative with second order accuracy
valuef diff6th(const valuef& in, int idx, const valuef& scale);

#endif // DERIVATIVES_HPP_INCLUDED

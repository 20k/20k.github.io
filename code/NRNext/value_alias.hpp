#ifndef VALUE_ALIAS_HPP_INCLUDED
#define VALUE_ALIAS_HPP_INCLUDED

#include "../common/value2.hpp"

using derivative_t = value<float16>;
using valuef = value<float>;
using valued = value<double>;
using valuei = value<int>;
using valueh = value<float16>;
using valuei64 = value<std::int64_t>;

using v2f = tensor<valuef, 2>;
using v3f = tensor<valuef, 3>;
using v4f = tensor<valuef, 4>;
using v2i = tensor<valuei, 2>;
using v3i = tensor<valuei, 3>;
using v4i = tensor<valuei, 4>;
using m44f = metric<valuef, 4, 4>;
using v3h = tensor<valueh, 3>;

using mut_v4f = tensor<mut<valuef>, 4>;
using mut_v3f = tensor<mut<valuef>, 3>;

using t3i = tensor<int, 3>;
using t3f = tensor<float, 3>;
using momentum_t = valueh;
using block_precision_t = valuef;

#endif // VALUE_ALIAS_HPP_INCLUDED

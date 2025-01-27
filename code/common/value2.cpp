#define VALUE2_BUILD
#include "value2.hpp"

namespace value_impl
{
template<typename T, typename U>
inline
auto replay_constant(const value<T>& v1, const value<T>& v2, const value<T>& v3, U&& func) -> std::optional<value<std::remove_reference_t<decltype(func(T(), T(), T()))>>>;

template<typename T, typename U>
inline
auto replay_constant(const value<T>& v1, const value<T>& v2, U&& func) -> std::optional<value<std::remove_reference_t<decltype(func(T(), T()))>>>;

template<typename T, typename U>
inline
auto replay_constant(const value<T>& v1, U&& func) -> std::optional<value<std::remove_reference_t<decltype(func(T()))>>>;

template<typename T, typename U>
inline
auto replay_constant(const value<T>& v1, const value<T>& v2, const value<T>& v3, U&& func) -> std::optional<value<std::remove_reference_t<decltype(func(T(), T()))>>>
{
    if(!v1.is_concrete_type() || !v2.is_concrete_type() || !v3.is_concrete_type())
        return std::nullopt;

    if(v1.concrete.index() != v2.concrete.index())
        return std::nullopt;

    if(v1.concrete.index() != v3.concrete.index())
        return std::nullopt;

    if(!std::holds_alternative<T>(v1.concrete))
        return std::nullopt;

    return func(std::get<T>(v1.concrete), std::get<T>(v2.concrete), std::get<T>(v3.concrete));
}

template<typename T, typename U>
inline
auto replay_constant(const value<T>& v1, const value<T>& v2, U&& func) -> std::optional<value<std::remove_reference_t<decltype(func(T(), T()))>>>
{
    if(!v1.is_concrete_type() || !v2.is_concrete_type())
        return std::nullopt;

    if(v1.concrete.index() != v2.concrete.index())
        return std::nullopt;

    if(!std::holds_alternative<T>(v1.concrete))
        return std::nullopt;

    return func(std::get<T>(v1.concrete), std::get<T>(v2.concrete));
}

template<typename T, typename U>
inline
auto replay_constant(const value<T>& v1, U&& func) -> std::optional<value<std::remove_reference_t<decltype(func(T()))>>>
{
    if(!v1.is_concrete_type())
        return std::nullopt;

    if(!std::holds_alternative<T>(v1.concrete))
        return std::nullopt;

    return func(std::get<T>(v1.concrete));
}

template<typename U>
inline
std::optional<value_base> replay_constant(const value_base& v1, const value_base& v2, const value_base& v3, U&& func)
{
    if(!v1.is_concrete_type() || !v2.is_concrete_type() || !v3.is_concrete_type())
        return std::nullopt;

    if(v1.concrete.index() != v2.concrete.index())
        return std::nullopt;

    if(v1.concrete.index() != v3.concrete.index())
        return std::nullopt;

    std::optional<value_base> out;

    std::visit([&]<typename V>(const V& i1)
    {
        value_base b;
        b.type = op::VALUE;
        b.concrete = func(i1, std::get<V>(v2.concrete), std::get<V>(v3.concrete));

        out = b;
    }, v1.concrete);

    return out;
}

template<typename U>
inline
std::optional<value_base> replay_constant(const value_base& v1, const value_base& v2, U&& func)
{
    if(!v1.is_concrete_type() || !v2.is_concrete_type())
        return std::nullopt;

    if(v1.concrete.index() != v2.concrete.index())
        return std::nullopt;

    std::optional<value_base> out;

    std::visit([&]<typename V>(const V& i1)
    {
        value_base b;
        b.type = op::VALUE;
        b.concrete = func(i1, std::get<V>(v2.concrete));

        out = b;
    }, v1.concrete);

    return out;
}

template<typename U>
inline
std::optional<value_base> replay_constant(const value_base& v1, U&& func)
{
    if(!v1.is_concrete_type())
        return std::nullopt;

    std::optional<value_base> out;

    std::visit([&](auto&& i1) {
        value_base b;
        b.type = op::VALUE;
        b.concrete = func(i1);

        out = b;
    }, v1.concrete);

    return out;
}
}

#define PROPAGATE_BASE3(vop, func) if(in.type == op::vop) { \
    out = replay_constant(in.args[0], in.args[1], in.args[2], func);\
}

#define PROPAGATE_BASE2(vop, func) if(in.type == op::vop) { \
    out = replay_constant(in.args[0], in.args[1], func);\
}

#define PROPAGATE_BASE1(vop, func) if(in.type == op::vop) { \
    out = replay_constant(in.args[0], func);\
}

value_base value_impl::optimise(const value_base& in)
{
    /*std::visit([&]<typename T>(const T& _){
        bool is_bad = std::is_same_v<T, double>;

        if(is_bad && in.type != op::CAST && in.type != op::FABS)
        {
            std::cout << "My Index " << in.concrete.index() << std::endl;
            std::cout << in.args[0].concrete.index() << std::endl;
            //std::cout << in.args[0].concrete.index() << " " << in.args[1].concrete.index() << " " << in.args[2].concrete.index() << std::endl;

            std::cout << "who " << value_to_string(in.args[0]) << std::endl;
            std::cout << "bad " << value_to_string(in) << std::endl;

            //assert(false);
        }

        //assert(!is_bad);
    }, in.concrete);*/

    using namespace stdmath;

    ///do constant propagation here

    bool all_constant = true;

    for(const auto& i : in.args)
    {
        if(!i.is_concrete_type())
            all_constant = false;
    }

    if(in.type == op::CAST)
    {
        if(in.args[1].is_concrete_type())
        {
            std::optional<value_base> out;

            std::visit([&]<typename U, typename T>(const U& unused, const T& in)
            {
                value_base cvt;
                cvt.type = op::VALUE;
                cvt.concrete = ucast<U>(in);

                out = cvt;
            }, in.concrete, in.args[1].concrete);

            if(out.has_value())
                return out.value();
        }
    }

    if(all_constant)
    {
        std::optional<value_base> out;

        PROPAGATE_BASE2(PLUS, op_plus);
        PROPAGATE_BASE2(MINUS, op_minus);
        PROPAGATE_BASE2(MULTIPLY, op_multiply);
        PROPAGATE_BASE2(DIVIDE, op_divide);
        PROPAGATE_BASE1(UMINUS, op_unary_minus);

        PROPAGATE_BASE3(FMA, ufma);

        PROPAGATE_BASE1(SIN, usin);
        PROPAGATE_BASE1(COS, ucos);
        PROPAGATE_BASE1(TAN, utan);

        PROPAGATE_BASE1(SINH, usinh);
        PROPAGATE_BASE1(COSH, ucosh);
        PROPAGATE_BASE1(TANH, utanh);

        PROPAGATE_BASE1(ASIN, uasin);
        PROPAGATE_BASE1(ACOS, uacos);
        PROPAGATE_BASE1(ATAN, uatan);

        PROPAGATE_BASE2(ATAN2, uatan2);

        PROPAGATE_BASE1(LOG, ulog);
        PROPAGATE_BASE1(LOG2, ulog2);
        PROPAGATE_BASE1(SQRT, usqrt);
        PROPAGATE_BASE1(FABS, ufabs);
        PROPAGATE_BASE1(ISFINITE, uisfinite);
        PROPAGATE_BASE2(FMOD, ufmod);
        PROPAGATE_BASE2(MOD, ufmod);
        PROPAGATE_BASE1(SIGN, usign);
        PROPAGATE_BASE1(FLOOR, ufloor);
        PROPAGATE_BASE1(CEIL, uceil);
        PROPAGATE_BASE1(INVERSE_SQRT, uinverse_sqrt);
        PROPAGATE_BASE2(POW, upow);
        PROPAGATE_BASE1(EXP, uexp);

        PROPAGATE_BASE2(MIN, umin);
        PROPAGATE_BASE2(MAX, umax);
        PROPAGATE_BASE3(CLAMP, uclamp);

        if(out)
            return out.value();
    }

    if(in.type == op::PLUS)
    {
        if(equivalent(in.args[0], in.args[0].make_constant_of_type(0.f)))
           return in.args[1];

        if(equivalent(in.args[1], in.args[1].make_constant_of_type(0.f)))
           return in.args[0];

        if(in.args[1].type == op::UMINUS)
            return in.args[0] - in.args[1].args[0];

        if(in.args[0].type == op::UMINUS)
            return in.args[1] - in.args[0].args[0];

        #ifdef FMAIFY
        if(in.is_floating_point_type())
        {
            if(in.args[0].type == op::MULTIPLY)
                return fma(in.args[0].args[0], in.args[0].args[1], in.args[1]);
            if(in.args[1].type == op::MULTIPLY)
                return fma(in.args[1].args[0], in.args[1].args[1], in.args[0]);
        }
        #endif
    }

    if(in.type == op::MULTIPLY)
    {
        value_base zero0 = in.args[0].make_constant_of_type(0.f);
        value_base zero1 = in.args[1].make_constant_of_type(0.f);

        if(equivalent(in.args[0], zero0))
            return zero0;

        if(equivalent(in.args[1], zero1))
            return zero1;

        if(equivalent(in.args[0], in.args[0].make_constant_of_type(1.f)))
            return in.args[1];

        if(equivalent(in.args[1], in.args[1].make_constant_of_type(1.f)))
            return in.args[0];

        if(equivalent(in.args[0], in.args[0].make_constant_of_type(-1.f)))
            return -in.args[1];

        if(equivalent(in.args[1], in.args[1].make_constant_of_type(-1.f)))
            return -in.args[0];

        if(in.args[1].type == op::DIVIDE)
        {
            //std::cout << value_to_string(in.args[0]) << " " << value_to_string(in.args[1]) << std::endl;

            if(equivalent(in.args[0], in.args[1].args[1]))
                return in.args[1].args[0];

            if(equivalent(-in.args[0], in.args[1].args[1]) || equivalent(in.args[0], -in.args[1].args[1]))
                return -in.args[1].args[0];

            //std::cout << "No op\n";
        }

        if(in.args[0].type == op::DIVIDE)
        {
            if(equivalent(in.args[1], in.args[0].args[1]))
                return in.args[0].args[0];

            if(equivalent(-in.args[1], in.args[0].args[1]) || equivalent(in.args[1], -in.args[0].args[1]))
                return -in.args[0].args[0];
        }
    }

    if(in.type == op::MINUS)
    {
        if(equivalent(in.args[0], in.args[1]))
            return in.args[0].make_constant_of_type(0.f);

        if(equivalent(in.args[1], in.args[1].make_constant_of_type(0.f)))
            return in.args[0];

        if(equivalent(in.args[0], in.args[0].make_constant_of_type(0.f)))
            return -in.args[1];

        value_base v1 = in.args[0];
        value_base v2 = in.args[1];

        if(v1.type == op::MULTIPLY && v2.type == op::MULTIPLY)
        {
            if(equivalent(v1.args[0], v1.args[1]) && equivalent(v2.args[0], v2.args[1]) && v1.args[0].type == op::PLUS && v2.args[0].type == op::MINUS)
            {
                if(equivalent(v1.args[0].args[0], v2.args[0].args[0]) && equivalent(v1.args[0].args[1], v2.args[0].args[1]))
                {
                    return v1.args[0].args[0].make_constant_of_type(4) * (v1.args[0].args[0] * v1.args[0].args[1]);
                }
            }
        }

        ///disabled because its a negative
        #if 0
        ///(a + b) - a == b
        ///(a + b) - b == a
        if(v1.type == op::PLUS)
        {
            if(equivalent(v1.args[0], v2))
                return v1.args[1];

            if(equivalent(v1.args[1], v2))
                return v1.args[0];
        }

        ///a - (a + b) == -b
        ///(b - (a + b) == -a
        if(v2.type == op::PLUS)
        {
            if(equivalent(v1, v2.args[0]))
                return -v2.args[1];

            if(equivalent(v1, v2.args[1]))
                return -v2.args[0];
        }

        ///(a - b) - a == -b
        ///(a - b) - b == n/a
        if(v1.type == op::MINUS)
        {
            if(equivalent(v1.args[0], v2))
                return -v1.args[1];
        }
        #endif
    }

    //the problem with this is it tends to elmininate common subexpressions
    if(in.type == op::UMINUS)
    {
        /*if(in.args[0].type == op::MINUS)
            return in.args[0].args[1] - in.args[0].args[0];

        if(in.args[0].type == op::DIVIDE)
        {
            if(in.args[0].args[0].is_concrete_type())
                return (-in.args[0].args[0]) / in.args[0].args[1];
        }

        if(in.args[0].type == op::MULTIPLY)
        {
            if(in.args[0].args[0].is_concrete_type())
                return (-in.args[0].args[0]) * in.args[0].args[1];

            if(in.args[0].args[1].is_concrete_type())
                return in.args[0].args[0] * (-in.args[0].args[1]);
        }*/

        if(in.args[0].type == op::UMINUS)
            return in.args[0].args[0];
    }

    if(in.type == op::DIVIDE)
    {
        if(equivalent(in.args[0], in.args[0].make_constant_of_type(0.f)))
            return in.args[0].make_constant_of_type(0.f);

        if(equivalent(in.args[0], in.args[1]))
            return in.args[0].make_constant_of_type(1.f);

        if(equivalent(in.args[1], in.args[1].make_constant_of_type(1.f)))
            return in.args[0];

        if(in.args[1].is_concrete_type() && in.args[1].is_floating_point_type())
            return in.args[0] * (in.args[1].make_constant_of_type(1.f)/in.args[1]);

        if(in.args[1].type == op::DIVIDE)
            return in.args[0] * (in.args[1].args[1] / in.args[1].args[0]);

        if(in.args[1].type == op::SQRT)
            return in.args[0] * inverse_sqrt(in.args[1].args[0]);
    }

    if(in.type == op::FMA)
    {
        if(equivalent(in.args[2], in.args[2].make_constant_of_type(0.f)))
            return in.args[0] * in.args[1];

        if(equivalent(in.args[0], in.args[0].make_constant_of_type(0.f)) || equivalent(in.args[1], in.args[1].make_constant_of_type(0.f)))
            return in.args[2];

        if(equivalent(in.args[0], in.args[0].make_constant_of_type(1.f)))
            return in.args[1] + in.args[2];

        if(equivalent(in.args[1], in.args[1].make_constant_of_type(1.f)))
            return in.args[0] + in.args[2];
    }

    return in;
}

namespace value_impl
{
template<typename U>
inline
value_base make_op_with_type_function(op::type t, const value_base& v1, U&& func);

template<typename U>
inline
value_base make_op_with_type_function(op::type t, const value_base& v1, const value_base& v2, U&& func);

///supports eg (bool) ? float : int
template<typename U>
inline
value_base make_op_with_type_function(op::type t, const value_base& v1, const value_base& v2, const value_base& v3, U&& func);

///only supports operations where all args are same type
template<typename U>
inline
value_base make_op_with_type_function_all_same(op::type t, const value_base& v1, const value_base& v2, U&& func);

template<typename U>
inline
value_base make_op_with_type_function_all_same(op::type t, const value_base& v1, const value_base& v2, const value_base& v3, U&& func);

template<typename U>
inline
value_base make_op_with_type_function(op::type t, const value_base& v1, U&& func)
{
    value_base ret;
    ret.type = t;
    ret.args = {v1};

    std::visit([&]<typename A>(const A&){
        ret.concrete = decltype(func(A()))();
    }, v1.concrete);

    return ret;
}

template<typename U>
inline
value_base make_op_with_type_function(op::type t, const value_base& v1, const value_base& v2, U&& func)
{
    value_base ret;
    ret.type = t;
    ret.args = {v1, v2};

    std::visit([&]<typename A, typename B>(const A&, const B&){
        if constexpr(std::is_invocable_v<U, A, B>) {
            ret.concrete = decltype(func(A(), B()))();
        }

    }, v1.concrete, v2.concrete);

    return ret;
}

template<typename U>
inline
value_base make_op_with_type_function(op::type t, const value_base& v1, const value_base& v2, const value_base& v3, U&& func)
{
    value_base ret;
    ret.type = t;
    ret.args = {v1, v2, v3};

    std::visit([&]<typename A, typename B, typename C>(const A&, const B&, const C&){
        if constexpr(std::is_invocable_v<U, A, B, C>) {
            ret.concrete = decltype(func(A(), B(), C()))();
        }

    }, v1.concrete, v2.concrete, v3.concrete);

    return ret;
}

template<typename U>
inline
value_base make_op_with_type_function_all_same(op::type t, const value_base& v1, const value_base& v2, U&& func)
{
    assert(v1.concrete.index() == v2.concrete.index());

    value_base ret;
    ret.type = t;
    ret.args = {v1, v2};

    std::visit([&]<typename A>(const A&){
        if constexpr(std::is_invocable_v<U, A, A>) {
            ret.concrete = decltype(func(A(), A()))();
        }
    }, v1.concrete);

    return ret;
}

template<typename U>
inline
value_base make_op_with_type_function_all_same(op::type t, const value_base& v1, const value_base& v2, const value_base& v3, U&& func)
{
    assert(v1.concrete.index() == v2.concrete.index());
    assert(v1.concrete.index() == v3.concrete.index());

    value_base ret;
    ret.type = t;
    ret.args = {v1, v2, v3};

    std::visit([&]<typename A>(const A&){
        if constexpr(std::is_invocable_v<U, A, A, A>) {
            ret.concrete = decltype(func(A(), A(), A()))();
        }
    }, v1.concrete);

    return ret;
}
}


#define IMPL_VALUE_FUNC1(type, name, func) \
value_base value_impl::name(const value_base& v1) {\
    return optimise(make_op_with_type_function(op::type, v1, func));\
}

#define IMPL_VALUE_FUNC2(type, name, func) \
value_base value_impl::name(const value_base& v1, const value_base& v2) {\
    return optimise(make_op_with_type_function_all_same(op::type, v1, v2, func));\
}

#define IMPL_VALUE_FUNC3(type, name, func) \
value_base value_impl::name(const value_base& v1, const value_base& v2, const value_base& v3) {\
    return optimise(make_op_with_type_function_all_same(op::type, v1, v2, v3, func));\
}

#define IMPL_VALUE_FUNC3_GEN(type, name, func) \
value_base value_impl::name(const value_base& v1, const value_base& v2, const value_base& v3) {\
    return optimise(make_op_with_type_function(op::type, v1, v2, v3, func));\
}

IMPL_VALUE_FUNC3(FMA, fma, stdmath::ufma);
IMPL_VALUE_FUNC1(SIN, sin, stdmath::usin);
IMPL_VALUE_FUNC1(COS, cos, stdmath::ucos);
IMPL_VALUE_FUNC1(TAN, tan, stdmath::utan);
IMPL_VALUE_FUNC1(ASIN, asin, stdmath::uasin);
IMPL_VALUE_FUNC1(ACOS, acos, stdmath::uacos);
IMPL_VALUE_FUNC1(ATAN, atan, stdmath::uatan);
IMPL_VALUE_FUNC1(SINH, sinh, stdmath::usinh);
IMPL_VALUE_FUNC1(COSH, cosh, stdmath::ucosh);
IMPL_VALUE_FUNC1(TANH, tanh, stdmath::utanh);
IMPL_VALUE_FUNC2(ATAN2, atan2, stdmath::uatan2);
IMPL_VALUE_FUNC1(SQRT, sqrt, stdmath::usqrt);
IMPL_VALUE_FUNC1(LOG, log, stdmath::ulog);
IMPL_VALUE_FUNC1(LOG2, log2, stdmath::ulog2);
IMPL_VALUE_FUNC1(FABS, fabs, stdmath::ufabs);
IMPL_VALUE_FUNC1(ISFINITE, isfinite, stdmath::uisfinite);
IMPL_VALUE_FUNC2(FMOD, fmod, stdmath::ufmod);
IMPL_VALUE_FUNC1(SIGN, sign, stdmath::usign);
IMPL_VALUE_FUNC1(FLOOR, floor, stdmath::ufloor);
IMPL_VALUE_FUNC1(CEIL, ceil, stdmath::uceil);
IMPL_VALUE_FUNC1(INVERSE_SQRT, inverse_sqrt, stdmath::uinverse_sqrt);
IMPL_VALUE_FUNC3_GEN(TERNARY, ternary, stdmath::uternary);
IMPL_VALUE_FUNC2(POW, pow, stdmath::upow);
IMPL_VALUE_FUNC1(EXP, exp, stdmath::uexp);

IMPL_VALUE_FUNC2(MIN, min, stdmath::umin);
IMPL_VALUE_FUNC2(MAX, max, stdmath::umax);
IMPL_VALUE_FUNC3(CLAMP, clamp, stdmath::uclamp);

#define IMPL_OPERATOR2(name, type, func) value_base value_impl::name(const value_base& v1, const value_base& v2) {return optimise(make_op_with_type_function_all_same(type, v1, v2, func));}
#define IMPL_OPERATOR1(name, type, func) value_base value_impl::name(const value_base& v1) {return optimise(make_op_with_type_function(type, v1, func));}

IMPL_OPERATOR2(operator%, op::MOD, stdmath::ufmod);
IMPL_OPERATOR2(operator+, op::PLUS, stdmath::op_plus);
IMPL_OPERATOR2(operator-, op::MINUS, stdmath::op_minus);
IMPL_OPERATOR2(operator*, op::MULTIPLY, stdmath::op_multiply);
IMPL_OPERATOR2(operator/, op::DIVIDE, stdmath::op_divide);
IMPL_OPERATOR1(operator-, op::UMINUS, stdmath::op_unary_minus);

IMPL_OPERATOR2(operator<, op::LT, stdmath::op_lt);
IMPL_OPERATOR2(operator<=, op::LTE, stdmath::op_lte);
IMPL_OPERATOR2(operator==, op::EQ, stdmath::op_eq);
IMPL_OPERATOR2(operator!=, op::NEQ, stdmath::op_neq);
IMPL_OPERATOR2(operator>, op::GT, stdmath::op_gt);
IMPL_OPERATOR2(operator>=, op::GTE, stdmath::op_gte);

bool value_impl::equivalent(const value_base& v1, const value_base& v2)
{
    //must be the same kind of op
    if(v1.type != v2.type)
        return false;

    //must have the same kind of abstract value name
    if(v1.abstract_value != v2.abstract_value)
        return false;

    //same concrete index
    if(v1.concrete.index() != v2.concrete.index())
        return false;

    bool invalid = true;

    //same value stored in concrete

    std::visit([&]<typename T>(const T& i1)
    {
        invalid = !(i1 == std::get<T>(v2.concrete));
    }, v1.concrete);

    if(invalid)
        return false;

    //same number of arguments
    if(v1.args.size() != v2.args.size())
        return false;

    //use the associativity of +* and check for the reverse equivalence
    if(v1.type == op::PLUS || v1.type == op::MULTIPLY)
    {
        if(equivalent(v1.args[0], v2.args[1]) && equivalent(v1.args[1], v2.args[0]))
            return true;
    }

    //check that all our arguments are equivalent
    for(int i=0; i < (int)v1.args.size(); i++)
    {
        if(!equivalent(v1.args[i], v2.args[i]))
            return false;
    }

    return true;
}

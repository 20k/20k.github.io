#ifndef VALUE_HPP_INCLUDED
#define VALUE_HPP_INCLUDED

#include <string>
#include <variant>
#include <vector>
#include <cstdint>
#include "stdmath.hpp"
#include "../common/vec/tensor.hpp"

#ifndef __clang__
#include <stdfloat>
#endif

#ifndef __clang__
using float16 = std::float16_t;
#else
using float16 = _Float16;
#endif

namespace value_impl
{
    template<typename T>
    inline
    std::string name_type(T tag);

    namespace op {
        enum type {
            NONE,
            VALUE,
            PLUS,
            MINUS,
            UMINUS,
            MULTIPLY,
            DIVIDE,
            MOD,

            ATOM_ADD,
            ATOM_XCHG,

            LT,
            LTE,
            EQ,
            NEQ,
            GT,
            GTE,
            NOT,
            LOR,
            LAND,

            FMA,
            SIN,
            COS,
            TAN,
            LOG,
            LOG2,

            ASIN,
            ACOS,
            ATAN,
            ATAN2,

            SINH,
            COSH,
            TANH,

            MIN,
            MAX,
            CLAMP,

            INVERSE_SQRT,
            SQRT,
            DOT,
            FMOD,
            ISFINITE,
            FABS,
            SIGN,
            FLOOR,
            CEIL,
            TERNARY,
            POW,
            EXP,

            GET_GLOBAL_ID,

            LAMBERT_W0,

            BRACKET,
            FOR,
            IF,
            BREAK,
            WHILE,
            RETURN,
            DECLARE,
            DECLARE_ARRAY,
            BLOCK_START,
            BLOCK_END,
            ASSIGN,
            SIDE_EFFECT,

            IMAGE_READ,
            IMAGE_READ_WITH_SAMPLER,
            IMAGE_WRITE,
            SAMPLER,

            CAST,
        };
    }

    template<typename Func, typename... Ts>
    inline
    auto change_variant_type(Func&& f, std::variant<Ts...> in)
    {
        return std::variant<decltype(f(Ts()))...>();
    }

    using supported_types = std::variant<double, float, float16, int, bool, std::uint64_t, std::int64_t>;

    struct value_base;

    template<typename T>
    inline
    T make_op(op::type t, const std::vector<value_base>& args)
    {
        T ret;
        ret.type = t;
        ret.args = args;
        return ret;
    }

    value_base optimise(const value_base& in);

    struct value_base
    {
        supported_types concrete;
        std::vector<value_base> args;
        op::type type = op::NONE;
        std::string name; //so that variables can be identified later. Clunky!
        std::string abstract_value; //if we are a named value, like eg a variable

        value_base(){}
        value_base(const std::string& str) : abstract_value(str){type = op::VALUE;}
        template<typename T>
        explicit value_base(const T& in) : concrete(in) {
            type = op::VALUE;
        }

        template<typename T>
        value_base make_constant_of_type(const T& in) const
        {
            value_base out;
            out.type = op::VALUE;

            std::visit([&]<typename U>(const U& real)
            {
                out.concrete = U(in);
            }, concrete);

            return out;
        }

        bool is_concrete_type() const
        {
            return type == op::VALUE && abstract_value.size() == 0;
        }

        bool is_floating_point_type() const
        {
            bool is_float = false;

            std::visit([&]<typename T>(const T& in){
                is_float = std::is_floating_point_v<T>;
            }, concrete);

            return is_float;
        }

        template<typename Tf, typename U>
        auto replay_impl(Tf&& type_factory, U&& handle_value) -> decltype(change_variant_type(type_factory, concrete))
        {
            if(type == op::type::VALUE)
                return handle_value(*this);

            using namespace std; //necessary for lookup

            using result_t = decltype(change_variant_type(type_factory, concrete));

            if(args.size() == 1)
            {
                auto c1 = args[0].replay_impl(type_factory, handle_value);

                return std::visit([&](auto&& v1)
                {
                    if(type == op::UMINUS)
                        return result_t(-v1);

                    if(type == op::SIN)
                        return result_t(sin(v1));

                     assert(false);
                }, c1);
            }

            if(args.size() == 2)
            {
                auto c1 = args[0].replay_impl(type_factory, handle_value);
                auto c2 = args[1].replay_impl(type_factory, handle_value);

                return std::visit([&]<typename T>(const T& v1) {
                    auto v2 = std::get<T>(c2);

                    if(type == op::PLUS)
                        return result_t(v1 + v2);

                    if(type == op::MULTIPLY)
                        return result_t(v1 * v2);

                    assert(false);
                    return result_t();
                });
            }

            assert(false);
        }

        template<typename T>
        void recurse(T&& func)
        {
            func(*this);

            for(int i=0; i < (int)args.size(); i++)
            {
                args[i].recurse(std::forward<T>(func));
            }
        }


        #define DECL_OPERATOR2(name, type, func) friend value_base name(const value_base& v1, const value_base& v2);
        #define DECL_OPERATOR1(name, type, func) friend value_base name(const value_base& v1);

        DECL_OPERATOR2(operator%, op::MOD, stdmath::ufmod);
        DECL_OPERATOR2(operator+, op::PLUS, stdmath::op_plus);
        DECL_OPERATOR2(operator-, op::MINUS, stdmath::op_minus);
        DECL_OPERATOR2(operator*, op::MULTIPLY, stdmath::op_multiply);
        DECL_OPERATOR2(operator/, op::DIVIDE, stdmath::op_divide);
        DECL_OPERATOR1(operator-, op::UMINUS, stdmath::op_unary_minus);

        DECL_OPERATOR2(operator<, op::LT, stdmath::op_lt);
        DECL_OPERATOR2(operator<=, op::LTE, stdmath::op_lte);
        DECL_OPERATOR2(operator==, op::EQ, stdmath::op_eq);
        DECL_OPERATOR2(operator!=, op::NEQ, stdmath::op_neq);
        DECL_OPERATOR2(operator>, op::GT, stdmath::op_gt);
        DECL_OPERATOR2(operator>=, op::GTE, stdmath::op_gte);

        #undef DECL_OPERATOR2
        #undef DECL_OPERATOR2
    };

    bool equivalent(const value_base& v1, const value_base& v2);

    #if 0
    template<typename U>
    inline
    std::optional<value_base> replay_constant(const value_base& v1, const value_base& v2, const value_base& v3, U&& func);

    template<typename U>
    inline
    std::optional<value_base> replay_constant(const value_base& v1, const value_base& v2, U&& func);

    template<typename U>
    inline
    std::optional<value_base> replay_constant(const value_base& v1, U&& func);
    #endif

    #define DECL_VALUE_FUNC1(type, name, func) \
    value_base name(const value_base& v1);

    #define DECL_VALUE_FUNC2(type, name, func) \
    value_base name(const value_base& v1, const value_base& v2);

    #define DECL_VALUE_FUNC3(type, name, func) \
    value_base name(const value_base& v1, const value_base& v2, const value_base& v3);

    #define DECL_VALUE_FUNC3_GEN(type, name, func) \
    value_base name(const value_base& v1, const value_base& v2, const value_base& v3);

    DECL_VALUE_FUNC3(FMA, fma, stdmath::ufma);
    DECL_VALUE_FUNC1(SIN, sin, stdmath::usin);
    DECL_VALUE_FUNC1(COS, cos, stdmath::ucos);
    DECL_VALUE_FUNC1(TAN, tan, stdmath::utan);
    DECL_VALUE_FUNC1(ASIN, asin, stdmath::uasin);
    DECL_VALUE_FUNC1(ACOS, acos, stdmath::uacos);
    DECL_VALUE_FUNC1(ATAN, atan, stdmath::uatan);
    DECL_VALUE_FUNC1(SINH, sinh, stdmath::usinh);
    DECL_VALUE_FUNC1(COSH, cosh, stdmath::ucosh);
    DECL_VALUE_FUNC1(TANH, tanh, stdmath::utanh);
    DECL_VALUE_FUNC2(ATAN2, atan2, stdmath::uatan2);
    DECL_VALUE_FUNC1(SQRT, sqrt, stdmath::usqrt);
    DECL_VALUE_FUNC1(LOG, log, stdmath::ulog);
    DECL_VALUE_FUNC1(LOG2, log2, stdmath::ulog2);
    DECL_VALUE_FUNC1(FABS, fabs, stdmath::ufabs);
    DECL_VALUE_FUNC1(ISFINITE, isfinite, stdmath::uisfinite);
    DECL_VALUE_FUNC2(FMOD, fmod, stdmath::ufmod);
    DECL_VALUE_FUNC1(SIGN, sign, stdmath::usign);
    DECL_VALUE_FUNC1(FLOOR, floor, stdmath::ufloor);
    DECL_VALUE_FUNC1(CEIL, ceil, stdmath::uceil);
    DECL_VALUE_FUNC1(INVERSE_SQRT, inverse_sqrt, stdmath::uinverse_sqrt);
    DECL_VALUE_FUNC3_GEN(TERNARY, ternary, stdmath::uternary);
    DECL_VALUE_FUNC2(POW, pow, stdmath::upow);
    DECL_VALUE_FUNC1(EXP, exp, stdmath::uexp);

    DECL_VALUE_FUNC2(MIN, min, stdmath::umin);
    DECL_VALUE_FUNC2(MAX, max, stdmath::umax);
    DECL_VALUE_FUNC3(CLAMP, clamp, stdmath::uclamp);

    std::string value_to_string(const value_base& v);

    value_base optimise(const value_base& in);

    template<typename T, typename U>
    inline
    T replay_value_base(const value_base& v, U&& handle_value)
    {
        ///returns a dual<value_base>(whatever)
        if(v.type == op::type::VALUE)
            return handle_value(v);

        using namespace stdmath;

        #define REPLAY1(func, name) if(v.type == op::func) return name(replay_value_base<T>(v.args[0], handle_value));
        #define REPLAY2(func, name) if(v.type == op::func) return name(replay_value_base<T>(v.args[0], handle_value), \
                                                                        replay_value_base<T>(v.args[1], handle_value));
        #define REPLAY3(func, name) if(v.type == op::func) return name(replay_value_base<T>(v.args[0], handle_value), \
                                                                        replay_value_base<T>(v.args[1], handle_value),\
                                                                        replay_value_base<T>(v.args[2], handle_value));

        REPLAY3(FMA, ufma);

        REPLAY1(SIN, usin);
        REPLAY1(COS, ucos);
        REPLAY1(TAN, utan);
        REPLAY1(SINH, usinh);
        REPLAY1(COSH, ucosh);
        REPLAY1(TANH, utanh);
        REPLAY1(ATAN, uatan);
        REPLAY2(ATAN2, uatan2);
        REPLAY1(LOG, ulog);
        //REPLAY1(LOG2, ulog2);
        REPLAY1(SQRT, usqrt);
        REPLAY1(FABS, ufabs);
        REPLAY1(ISFINITE, uisfinite);
        REPLAY2(FMOD, ufmod);

        REPLAY2(PLUS, op_plus);
        REPLAY2(MINUS, op_minus);
        REPLAY2(MULTIPLY, op_multiply);
        REPLAY2(DIVIDE, op_divide);
        REPLAY1(UMINUS, op_unary_minus);
        REPLAY2(MOD, ufmod);
        REPLAY3(TERNARY, uternary);

        REPLAY2(LT, op_lt);
        REPLAY2(LTE, op_lte);
        REPLAY2(EQ, op_eq);
        REPLAY2(GT, op_gt);
        REPLAY2(GTE, op_gte);

        REPLAY2(MIN, umin);
        REPLAY2(MAX, umax);
        //REPLAY3(CLAMP, uclamp);
        REPLAY2(POW, upow);

        assert(false);
    }

    template<typename T>
    struct value;

    /*#define OPTIMISE_VALUE

    #ifdef OPTIMISE_VALUE
    #define PROPAGATE2(x, y, op) if(auto it = replay_constant(x, y, [](const T& a, const T& b){using namespace stdmath; return op(a,b);})){return it.value();}
    #define PROPAGATE1(x, op)  if(auto it = replay_constant(x, [](const T& a){using namespace stdmath; return op(a);})){return it.value();}
    #else
    #define PROPAGATE2(x, y, op)
    #define PROPAGATE1(x, op)
    #endif*/

    template<typename T>
    inline
    value<T> from_base(const value_base& b);

    template<typename T>
    struct value : value_base {
        using interior_type = T;

        value() {
            value_base::type = op::VALUE;
            value_base::concrete = T{};
        }

        value(T t) {
            value_base::type = op::VALUE;
            value_base::concrete = t;
        }

        void set_from_base(const value_base& in)
        {
            static_cast<value_base&>(*this) = in;
        }

        template<typename U>
        explicit operator value<U>() const
        {
            return this->to<U>();
        }

        friend value<T> operator%(const value<T>& v1, const value<T>& v2) {
            value<T> result;
            result.type = op::MOD;
            result.args = {v1, v2};
            return from_base<T>(optimise(result));
        }

        friend value<T> operator+(const value<T>& v1, const value<T>& v2) {
            value<T> result;
            result.type = op::PLUS;
            result.args = {v1, v2};
            return from_base<T>(optimise(result));
        }

        friend value<T> operator*(const value<T>& v1, const value<T>& v2) {
            value<T> result;
            result.type = op::MULTIPLY;
            result.args = {v1, v2};
            return from_base<T>(optimise(result));
        }

        friend value<T> operator-(const value<T>& v1, const value<T>& v2) {
            value<T> result;
            result.type = op::MINUS;
            result.args = {v1, v2};
            return from_base<T>(optimise(result));
        }

        friend value<T> operator/(const value<T>& v1, const value<T>& v2) {
            value<T> result;
            result.type = op::DIVIDE;
            result.args = {v1, v2};
            return from_base<T>(optimise(result));
        }


        friend value<T> operator-(const value<T>& v1) {
            value<T> result;
            result.type = op::UMINUS;
            result.args = {v1};
            return from_base<T>(optimise(result));
        }


        friend value<bool> operator<(const value<T>& v1, const value<T>& v2) {
            value<bool> result;
            result.type = op::LT;
            result.args = {v1, v2};
            return result;
        }

        friend value<bool> operator<=(const value<T>& v1, const value<T>& v2) {
            value<bool> result;
            result.type = op::LTE;
            result.args = {v1, v2};
            return result;
        }

        friend value<bool> operator==(const value<T>& v1, const value<T>& v2) {
            value<bool> result;
            result.type = op::EQ;
            result.args = {v1, v2};
            return result;
        }

        friend value<bool> operator!=(const value<T>& v1, const value<T>& v2) {
            value<bool> result;
            result.type = op::NEQ;
            result.args = {v1, v2};
            return result;
        }

        friend value<bool> operator>(const value<T>& v1, const value<T>& v2) {
            value<bool> result;
            result.type = op::GT;
            result.args = {v1, v2};
            return result;
        }

        friend value<bool> operator>=(const value<T>& v1, const value<T>& v2) {
            value<bool> result;
            result.type = op::GTE;
            result.args = {v1, v2};
            return result;
        }

        friend value<T>& operator+=(value<T>& v1, const value<T>& v2) {
            v1 = v1 + v2;
            return v1;
        }

        friend value<T>& operator-=(value<T>& v1, const value<T>& v2) {
            v1 = v1 - v2;
            return v1;
        }

        friend value<bool> operator||(const value<T>& v1, const value<T>& v2) {
            value<bool> result;
            result.type = op::LOR;
            result.args = {v1, v2};
            return result;
        }

        friend value<bool> operator&&(const value<T>& v1, const value<T>& v2) {
            value<bool> result;
            result.type = op::LAND;
            result.args = {v1, v2};
            return result;
        }

        friend value<bool> operator!(const value<T>& v1) {
            value<bool> result;
            result.type = op::NOT;
            result.args = {v1};
            return result;
        }

        template<typename U>
        value<U> to() const
        {
            value_base out_type = name_type(U());

            value<U> ret;
            ret.type = op::CAST;
            ret.args = {out_type, *this};

            return from_base<U>(optimise(ret));
        }

        template<typename Tf, typename U>
        auto replay(Tf&& type_factory, U&& handle_value) -> decltype(type_factory(T()))
        {
            using result_t = decltype(type_factory(T()));

            return std::get<result_t>(replay_impl(type_factory, handle_value));
        }
    };

    template<typename T>
    inline
    T get_interior_type(const T&){return T();}

    template<typename T, int... N>
    inline
    T get_interior_type(const tensor<T, N...>&){return T();}

    template<typename T>
    inline
    T get_interior_type(const value<T>&){return T();}

    template<typename T, int... N>
    inline
    T get_interior_type(const tensor<value<T>, N...>&){return T();}

    template<typename T>
    inline
    value<T> from_base(const value_base& b)
    {
        value<T> ret;
        ret.set_from_base(b);
        return ret;
    }

    template<typename T, typename U>
    struct mutable_proxy
    {
        const T& v;
        U& ctx;

        mutable_proxy(const T& _v, U& _ctx) : v(_v), ctx(_ctx){}

        template<typename V>
        void operator=(const V& to_set)
        {
            assign_e(ctx, v, to_set);
        }
    };

    template<typename T>
    struct mut : T
    {
        const T& as_constant() const
        {
            return *this;
        }

        void set_from_constant(const T& in)
        {
            static_cast<T&>(*this) = in;
        }

        template<typename U>
        auto as_ref(U& executor)
        {
            return mutable_proxy(*this, executor);
        }
    };

    template<typename T>
    inline
    value_base return_b(const value<T>& in)
    {
        value<T> v;
        v.type = op::RETURN;
        v.args = {in};
        return v;
    }

    inline
    value_base return_b()
    {
        value_base v;
        v.type = op::RETURN;
        return v;
    }

    inline
    value_base break_b()
    {
        value_base v;
        v.type = op::BREAK;
        return v;
    }

    inline
    value_base block_start_b()
    {
        value_base base;
        base.type = op::BLOCK_START;
        return base;
    }

    inline
    value_base block_end_b()
    {
        value_base base;
        base.type = op::BLOCK_END;
        return base;
    }

    inline
    value_base if_b(const value<bool>& condition)
    {
        value_base base;
        base.type = op::IF;
        base.args = {condition};

        return base;
    }

    inline
    value_base while_b(const value<bool>& condition)
    {
        value_base base;
        base.type = op::WHILE;
        base.args = {condition};

        return base;
    }

    template<typename T>
    inline
    value_base assign_b(const mut<value<T>>& v1, const value<T>& v2)
    {
        value_base result;
        result.type = op::ASSIGN;
        result.args = {v1, v2};

        return result;
    }

    template<typename T>
    inline
    value_base assign_b(const mut<value<T>>& v1, const mut<value<T>>& v2)
    {
        value_base result;
        result.type = op::ASSIGN;
        result.args = {v1, v2};

        return result;
    }

    template<typename T, int... N>
    inline
    tensor<value_base, N...> assign_b(const mut<tensor<T, N...>>& v1, const tensor<T, N...>& v2)
    {
        const tensor<T, N...>& unpacked = v1.as_constant();

        return tensor_for_each_nary([&](const T& v1, const T& v2)
        {
            mut<T> packed;
            packed.set_from_constant(v1);

            return assign_b(packed, v2);
        }, unpacked, v2);
    }

    template<typename T, int... N>
    inline
    tensor<value_base, N...> assign_b(const tensor<mut<T>, N...>& v1, const tensor<T, N...>& v2)
    {
        return tensor_for_each_nary([&](const mut<T>& v1, const T& v2)
        {
            return assign_b(v1, v2);
        }, v1, v2);
    }

    template<typename T, int... N>
    inline
    tensor<value_base, N...> assign_b(const tensor<mut<T>, N...>& v1, const tensor<mut<T>, N...>& v2)
    {
        return tensor_for_each_nary([&](const mut<T>& v1, const mut<T>& v2)
        {
            return assign_b(v1, v2);
        }, v1, v2);
    }

    inline
    value_base declare_b(const std::string& name, const value_base& rhs)
    {
        value_base type;

        value_base v;

        std::visit([&]<typename T>(const T&)
        {
            v.concrete = T();
            type = name_type(T());
        }, rhs.concrete);

        v.type = op::DECLARE;
        v.args = {type, name, rhs};

        return v;
    }

    template<typename T>
    inline
    value_base declare_array_b(const std::string& name, int array_size, const std::vector<value_base>& rhs)
    {
        value_base type;

        value_base size;
        size.type = op::VALUE;
        size.concrete = (int)rhs.size();

        value_base varray_size;
        varray_size.type = op::VALUE;
        varray_size.concrete = array_size;

        value_base v;
        v.type = op::DECLARE_ARRAY;
        v.args = {name_type(T()), name, varray_size, size};
        v.concrete = get_interior_type(T());

        for(const auto& i : rhs)
        {
            v.args.push_back(i);
        }

        return v;
    }

    inline
    value_base for_b(const value<bool>& condition, const value_base& execute)
    {
        value_base type;
        type.type = op::FOR;
        type.args = {condition, execute};

        return type;
    }

    template<typename T>
    inline
    value<T> sin(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::SIN;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> cos(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::COS;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> tan(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::TAN;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> sinh(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::SINH;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> cosh(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::COSH;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> tanh(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::TANH;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> asin(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::ASIN;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> acos(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::ACOS;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> atan(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::ATAN;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> atan2(const value<T>& v1, const value<T>& v2)
    {
        value<T> ret;
        ret.type = op::ATAN2;
        ret.args = {v1, v2};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> log(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::LOG;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> log2(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::LOG2;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> sqrt(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::SQRT;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> inverse_sqrt(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::INVERSE_SQRT;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    value<T> fmod(const value<T>& v1, const value<T>& v2)
    {
        value<T> ret;
        ret.type = op::FMOD;
        ret.args = {v1, v2};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> fabs(const value<T>& v1)
    {
        value<T> ret;
        ret.type = op::FABS;
        ret.args = {v1};
        return from_base<T>(optimise(ret));
    }

    template<typename T>
    value<int> isfinite(const value<T>& v1)
    {
        value<int> ret;
        ret.type = op::ISFINITE;
        ret.args = {v1};
        return from_base<int>(optimise(ret));
    }

    template<typename T>
    inline
    value<T> sign(const value<T>& v1)
    {
        return from_base<T>(optimise(make_op<value<T>>(op::SIGN, {v1})));
    }

    template<typename T>
    inline
    value<T> floor(const value<T>& v1)
    {
        return from_base<T>(optimise(make_op<value<T>>(op::FLOOR, {v1})));
    }

    template<typename T>
    inline
    value<T> ceil(const value<T>& v1)
    {
        return from_base<T>(optimise(make_op<value<T>>(op::CEIL, {v1})));
    }

    template<typename T>
    inline
    value<T> ternary(const value<bool>& condition, const value<T>& if_true, const value<T>& if_false)
    {
        return from_base<T>(optimise(make_op<value<T>>(op::TERNARY, {condition, if_true, if_false})));
    }

    template<typename T, int N>
    inline
    tensor<value<T>, N> ternary(const value<bool>& condition, const tensor<value<T>, N>& if_true, const tensor<value<T>, N>& if_false)
    {
        tensor<value<T>, N> ret;

        for(int i=0; i < N; i++)
            ret[i] = from_base<T>(optimise(make_op<value<T>>(op::TERNARY, {condition, if_true[i], if_false[i]})));

        return ret;
    }

    template<typename T>
    inline
    value<T> min(const value<T>& v1, const value<T>& v2)
    {
        return from_base<T>(optimise(make_op<value<T>>(op::MIN, {v1, v2})));
    }

    template<typename T>
    inline
    value<T> max(const value<T>& v1, const value<T>& v2)
    {
        return from_base<T>(optimise(make_op<value<T>>(op::MAX, {v1, v2})));
    }

    template<typename T>
    inline
    value<T> clamp(const value<T>& v1, const value<T>& v2, const value<T>& v3)
    {
        return from_base<T>(optimise(make_op<value<T>>(op::CLAMP, {v1, v2, v3})));
    }

    template<typename T>
    inline
    value<T> clamp(const value<T>& v1, const T& v2, const T& v3)
    {
        return from_base<T>(optimise(make_op<value<T>>(op::CLAMP, {v1, value<T>(v2), value<T>(v3)})));
    }


    template<typename T>
    inline
    value<T> pow(const value<T>& v1, const value<T>& v2)
    {
        return from_base<T>(optimise(make_op<value<T>>(op::POW, {v1, v2})));
    }

    template<typename T>
    inline
    value<T> pow(const value<T>& v1, const T& v2)
    {
        return from_base<T>(optimise(make_op<value<T>>(op::POW, {v1, value<T>(v2)})));
    }

    template<typename T>
    inline
    value<T> exp(const value<T>& v1)
    {
        return from_base<T>(optimise(make_op<value<T>>(op::EXP, {v1})));
    }

    template<typename T>
    inline
    value<T> mix(const value<T>& v1, const value<T>& v2, const value<T>& v3)
    {
        return v1 * (1-v3) + v2 * v3;
    }

    template<typename T>
    inline
    value<T> fma(const value<T>& v1, const value<T>& v2, const value<T>& v3)
    {
         return from_base<T>(optimise(make_op<value<T>>(op::FMA, {v1, v2, v3})));
    }

    template<typename T>
    inline
    std::string name_type(T tag)
    {
        if constexpr(std::is_same_v<T, float>)
            return "float";

        else if constexpr(std::is_same_v<T, double>)
            return "double";

        else if constexpr(std::is_same_v<T, float16>)
            return "half";

        else if constexpr(std::is_same_v<T, int>)
            return "int";

        else if constexpr(std::is_same_v<T, short>)
            return "short";

        else if constexpr(std::is_same_v<T, unsigned int>)
            return "unsigned int";

        else if constexpr(std::is_same_v<T, unsigned short>)
            return "unsigned short";

        else if constexpr(std::is_same_v<T, char>)
            return "char";

        else if constexpr(std::is_same_v<T, unsigned char>)
            return "unsigned char";

        else if constexpr(std::is_same_v<T, value<float>>)
            return "float";

        else if constexpr(std::is_same_v<T, value<double>>)
            return "double";

        else if constexpr(std::is_same_v<T, value<float16>>)
            return "half";

        else if constexpr(std::is_same_v<T, value<int>>)
            return "int";

        else if constexpr(std::is_same_v<T, value<short>>)
            return "short";

        else if constexpr(std::is_same_v<T, value<unsigned int>>)
            return "unsigned int";

        else if constexpr(std::is_same_v<T, value<unsigned short>>)
            return "unsigned short";

        else if constexpr(std::is_same_v<T, value<char>>)
            return "char";

        else if constexpr(std::is_same_v<T, value<unsigned char>>)
            return "unsigned char";

        else if constexpr(std::is_same_v<T, std::monostate>)
            return "monostate##neverused";

        else if constexpr(std::is_same_v<T, tensor<value<float>, 4>>)
            return "float4";

        else if constexpr(std::is_same_v<T, tensor<value<float>, 3>>)
            return "float3";

        else if constexpr(std::is_same_v<T, tensor<value<float>, 2>>)
            return "float2";

        else if constexpr(std::is_same_v<T, tensor<value<int>, 4>>)
            return "int4";

        else if constexpr(std::is_same_v<T, tensor<value<int>, 3>>)
            return "int3";

        else if constexpr(std::is_same_v<T, tensor<value<int>, 2>>)
            return "int2";

        else if constexpr(std::is_same_v<T, tensor<value<unsigned short>, 4>>)
            return "ushort4";

        else if constexpr(std::is_same_v<T, tensor<value<unsigned short>, 3>>)
            return "ushort3";

        else if constexpr(std::is_same_v<T, tensor<value<unsigned short>, 2>>)
            return "ushort2";

        else if constexpr(std::is_same_v<T, tensor<value<short>, 4>>)
            return "short4";

        else if constexpr(std::is_same_v<T, tensor<value<short>, 3>>)
            return "short3";

        else if constexpr(std::is_same_v<T, tensor<value<short>, 2>>)
            return "short2";

        else if constexpr(std::is_same_v<T, tensor<value<float16>, 4>>)
            return "half4";

        else if constexpr(std::is_same_v<T, tensor<value<float16>, 3>>)
            return "half3";

        else if constexpr(std::is_same_v<T, tensor<value<float16>, 2>>)
            return "half2";

        else if constexpr(std::is_same_v<T, tensor<float, 4>>)
            return "float4";

        else if constexpr(std::is_same_v<T, tensor<float, 3>>)
            return "float3";

        else if constexpr(std::is_same_v<T, tensor<float, 2>>)
            return "float2";

        else if constexpr(std::is_same_v<T, tensor<int, 4>>)
            return "int4";

        else if constexpr(std::is_same_v<T, tensor<int, 3>>)
            return "int3";

        else if constexpr(std::is_same_v<T, tensor<int, 2>>)
            return "int2";

        else if constexpr(std::is_same_v<T, tensor<unsigned short, 4>>)
            return "ushort4";

        else if constexpr(std::is_same_v<T, tensor<unsigned short, 3>>)
            return "ushort3";

        else if constexpr(std::is_same_v<T, tensor<unsigned short, 2>>)
            return "ushort2";

        else if constexpr(std::is_same_v<T, tensor<short, 4>>)
            return "short4";

        else if constexpr(std::is_same_v<T, tensor<short, 3>>)
            return "short3";

        else if constexpr(std::is_same_v<T, tensor<short, 2>>)
            return "short2";

        else if constexpr(std::is_same_v<T, tensor<float16, 4>>)
            return "half4";

        else if constexpr(std::is_same_v<T, tensor<float16, 3>>)
            return "half3";

        else if constexpr(std::is_same_v<T, tensor<float16, 2>>)
            return "half2";

        else if constexpr(std::is_same_v<T, bool>)
            return "bool";

        else if constexpr(std::is_same_v<T, std::uint64_t>)
            return "ulong";

        else if constexpr(std::is_same_v<T, value<std::uint64_t>>)
            return "ulong";

        else if constexpr(std::is_same_v<T, std::int64_t>)
            return "long";

        else if constexpr(std::is_same_v<T, value<std::int64_t>>)
            return "long";

        else
        {
            #ifndef __clang__
            static_assert(false);
            #endif
        }
    }
}

using value_base = value_impl::value_base;
template<typename T>
using value = value_impl::value<T>;
template<typename T>
using mut = value_impl::mut<T>;

#endif // VALUE_HPP_INCLUDED

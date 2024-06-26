#ifndef VALUE_HPP_INCLUDED
#define VALUE_HPP_INCLUDED

#include <string>
#include <variant>
#include <vector>
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

            LT,
            LTE,
            EQ,
            GT,
            GTE,
            NEQ,
            NOT,
            LOR,

            SIN,
            COS,
            TAN,

            SQRT,
            DOT,
            FMOD,
            ISFINITE,

            GET_GLOBAL_ID,

            LAMBERT_W0,

            BRACKET,
            FOR,
            IF,
            BREAK,
            WHILE,
            RETURN,
            DECLARE,
            BLOCK_START,
            BLOCK_END,
            ASSIGN,

            IMAGE_READ,
            IMAGE_WRITE,

            CAST,
        };
    }

    template<typename Func, typename... Ts>
    inline
    auto change_variant_type(Func&& f, std::variant<Ts...> in)
    {
        return std::variant<decltype(f(Ts()))...>();
    }

    using supported_types = std::variant<double, float, float16, int, bool>;

    struct value_base
    {
        supported_types concrete;
        std::vector<value_base> args;
        op::type type = op::NONE;
        std::string name; //so that variables can be identified later. Clunky!
        std::string abstract_value; //if we are a named value, like eg a variable

        value_base(){}
        value_base(const std::string& str) : abstract_value(str){type = op::VALUE;}

        bool is_concrete_type() const
        {
            return type == op::VALUE && abstract_value.size() == 0;
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

                return std::visit([&](auto&& v1, auto&& v2) {
                    if constexpr(std::is_same_v<decltype(v1), decltype(v2)>)
                    {
                        if(type == op::PLUS)
                            return result_t(v1 + v2);

                        if(type == op::MULTIPLY)
                            return result_t(v1 * v2);
                    }

                    assert(false);
                    return result_t();
                }, c1, c2);
            }

            assert(false);
        }
    };

    template<typename T>
    struct value;

    template<typename T, typename U>
    inline
    auto replay_constant(const value<T>& v1, const value<T>& v2, U&& func) -> std::optional<value<std::remove_reference_t<decltype(func(T(), T()))>>>;

    template<typename T, typename U>
    inline
    auto replay_constant(const value<T>& v1, U&& func) -> std::optional<value<std::remove_reference_t<decltype(func(T()))>>>;

    inline
    bool equivalent(const value_base& v1, const value_base& v2);

    #define OPTIMISE_VALUE

    #ifdef OPTIMISE_VALUE
    #define PROPAGATE_INFIX(x, y, op) if(auto it = replay_constant(x, y, [](const T& a, const T& b){return a op b;})){return it.value();}
    #define PROPAGATE2(x, y, op) if(auto it = replay_constant(x, y, [](const T& a, const T& b){return op(a,b);})){return it.value();}
    #define PROPAGATE1(x, op)  if(auto it = replay_constant(x, [](const T& a){return op(a);})){return it.value();}
    #else
    #define PROPAGATE_INFIX(x, y, op)
    #define PROPAGATE2(x, y, op)
    #define PROPAGATE1(x, op)
    #endif

    template<typename T>
    struct value : value_base {
        value() {
            value_base::type = op::VALUE;
            value_base::concrete = T{};
        }

        value(T t) {
            value_base::type = op::VALUE;
            value_base::concrete = t;
        }

        friend value<T> operator%(const value<T>& v1, const value<T>& v2) {
            PROPAGATE_INFIX(v1, v2, %);

            value<T> result;
            result.type = op::MOD;
            result.args = {v1, v2};
            return result;
        }

        friend value<T> operator+(const value<T>& v1, const value<T>& v2) {
            PROPAGATE_INFIX(v1, v2, +);

            #ifdef OPTIMISE_VALUE
            if(equivalent(v1, value<T>(0)))
                return v2;

            if(equivalent(v2, value<T>(0)))
                return v1;
            #endif

            value<T> result;
            result.type = op::PLUS;
            result.args = {v1, v2};
            return result;
        }

        friend value<T> operator*(const value<T>& v1, const value<T>& v2) {
            PROPAGATE_INFIX(v1, v2, *);

            #ifdef OPTIMISE_VALUE
            if(equivalent(v1, value<T>(0)))
                return (T)0;

            if(equivalent(v2, value<T>(0)))
                return (T)0;

            if(equivalent(v1, value<T>(1)))
                return v2;

            if(equivalent(v2, value<T>(1)))
                return v1;
            #endif

            value<T> result;
            result.type = op::MULTIPLY;
            result.args = {v1, v2};
            return result;
        }

        friend value<T> operator-(const value<T>& v1, const value<T>& v2) {
            PROPAGATE_INFIX(v1, v2, -);

            #ifdef OPTIMISE_VALUE
            if(equivalent(v1, v2))
                return (T)0;

            if(equivalent(v2, value<T>(0)))
                return v1;

            ///this is a performance negative, for some reason
            if(equivalent(v1, value<T>(0)))
                return -v2;
            #endif

            value<T> result;
            result.type = op::MINUS;
            result.args = {v1, v2};
            return result;
        }

        friend value<T> operator/(const value<T>& v1, const value<T>& v2) {
            PROPAGATE_INFIX(v1, v2, /);

            #ifdef OPTIMISE_VALUE
            if(equivalent(v1, value<T>(0)))
                return (T)0;

            if(equivalent(v2, value<T>(1)))
                return v1;

            if(equivalent(v1, v2))
                return (T)1;

            if(v2.is_concrete_type() && std::is_floating_point_v<T>)
                return v1 * (1/v2);
            #endif

            value<T> result;
            result.type = op::DIVIDE;
            result.args = {v1, v2};
            return result;
        }


        friend value<T> operator-(const value<T>& v1) {
            PROPAGATE1(v1, -);

            #ifdef OPTIMISE_VALUE
            if(equivalent(v1, value<T>(0)))
                return v1;
            #endif

            value<T> result;
            result.type = op::UMINUS;
            result.args = {v1};
            return result;
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

        friend value<bool> operator!(const value<T>& v1) {
            value<bool> result;
            result.type = op::NOT;
            result.args = {v1};
            return result;
        }

        template<typename U>
        value<U> to()
        {
            PROPAGATE1(*this, (U));

            value_base out_type = name_type(U());

            value<U> ret;
            ret.type = op::CAST;
            ret.args = {out_type, *this};

            return ret;
        }

        template<typename Tf, typename U>
        auto replay(Tf&& type_factory, U&& handle_value) -> decltype(type_factory(T()))
        {
            using result_t = decltype(type_factory(T()));

            return std::get<result_t>(replay_impl(type_factory, handle_value));
        }
    };

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

    inline
    bool equivalent(const value_base& v1, const value_base& v2)
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
        std::visit([&](auto&& i1, auto&& i2)
        {
            invalid = !(i1 == i2);
        }, v1.concrete, v2.concrete);

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

    template<typename T, typename U>
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
    auto replay_constant(const value<T>& v1, U&& func) -> std::optional<value<std::remove_reference_t<decltype(func(T()))>>>
    {
        if(!v1.is_concrete_type())
            return std::nullopt;

        if(!std::holds_alternative<T>(v1.concrete))
            return std::nullopt;

        return func(std::get<T>(v1.concrete));
    }

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

    template<typename T>
    inline
    value_base declare_b(const std::string& name, const value<T>& rhs)
    {
        value_base type = name_type(T());

        value<T> v;
        v.type = op::DECLARE;
        v.args = {type, name, rhs};

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
        using std::sin;
        PROPAGATE1(v1, sin);

        value<T> ret;
        ret.type = op::SIN;
        ret.args = {v1};
        return ret;
    }

    template<typename T>
    inline
    value<T> cos(const value<T>& v1)
    {
        using std::cos;
        PROPAGATE1(v1, cos);

        value<T> ret;
        ret.type = op::COS;
        ret.args = {v1};
        return ret;
    }

    template<typename T>
    inline
    value<T> sqrt(const value<T>& v1)
    {
        using std::sqrt;
        PROPAGATE1(v1, sqrt);

        value<T> ret;
        ret.type = op::SQRT;
        ret.args = {v1};
        return ret;
    }

    template<typename T>
    value<T> fmod(const value<T>& v1, const value<T>& v2)
    {
        using std::fmod;
        PROPAGATE2(v1, v2, fmod);

        value<T> ret;
        ret.type = op::FMOD;
        ret.args = {v1, v2};
        return ret;
    }

    template<typename T>
    value<int> isfinite(const value<T>& v1)
    {
        using std::isfinite;
        PROPAGATE1(v1, (int)isfinite);

        value<int> ret;
        ret.type = op::ISFINITE;
        ret.args = {v1};
        return ret;
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

        else if constexpr(std::is_same_v<T, tensor<float16, 4>>)
            return "half4";

        else if constexpr(std::is_same_v<T, tensor<float16, 3>>)
            return "half3";

        else if constexpr(std::is_same_v<T, tensor<float16, 2>>)
            return "half2";

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

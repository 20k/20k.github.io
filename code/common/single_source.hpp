#ifndef SINGLE_SOURCE_HPP_INCLUDED
#define SINGLE_SOURCE_HPP_INCLUDED

#include "../common/vec/dual.hpp"
#include "../common/value2.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <assert.h>
#include <set>
#include <map>
#include <array>
#include <concepts>
#include <mutex>
#include <thread>

namespace value_impl
{
    struct execution_context_base
    {
        std::vector<value_base> to_execute;
        std::vector<std::pair<value_base, value_base>> aliases;
        virtual void add(value_base&& in) = 0;
        virtual void add(const value_base& in) = 0;
        virtual void alias(const value_base& check, const value_base& to) = 0;
        virtual int next_id() = 0;

        template<typename T, int... N>
        void add(tensor<T, N...>&& inout)
        {
            inout.for_each([&](auto&& v)
            {
                add(std::move(v));
            });
        }

        template<typename T>
        void pin(value<T>& inout)
        {
            if(inout.type == value_impl::op::VALUE && !inout.is_concrete_type())
                return;

            std::string name = "v" + std::to_string(next_id());

            add(declare_b(name, inout));

            value<T> pinned;
            pinned.abstract_value = name;

            inout = pinned;
        }

        template<typename T, int... N>
        void pin(tensor<T, N...>& inout)
        {
            inout.for_each([&](T& v)
            {
                pin(v);
            });
        }

        template<typename T, int... N>
        void pin(metric<T, N...>& inout)
        {
            inout.for_each([&](T& v)
            {
                pin(v);
            });
        }

        template<typename T>
        void pin(dual_types::dual_v<T>& in)
        {
            pin(in.real);
            pin(in.dual);
        }

        template<typename T, int... N>
        void pin(inverse_metric<T, N...>& inout)
        {
            inout.for_each([&](T& v)
            {
                pin(v);
            });
        }

        virtual ~execution_context_base(){}
    };

    struct execution_context : execution_context_base
    {
        int id = 0;

        void alias(const value_base& check, const value_base& to) override
        {
            aliases.push_back({check, to});
        }

        void add(value_base&& in) override
        {
            to_execute.push_back(std::move(in));
        }

        void add(const value_base& in) override
        {
            to_execute.push_back(std::move(in));
        }

        int next_id() override
        {
            return id++;
        }
    };

    inline
    std::vector<execution_context_base*>& context_stack()
    {
        static std::mutex mut;
        static std::map<std::thread::id, std::vector<execution_context_base*>> ctx;

        std::lock_guard guard(mut);
        return ctx[std::this_thread::get_id()];
    }

    template<typename T>
    inline
    T& push_context()
    {
        T* ptr = new T();

        context_stack().push_back(ptr);
        return *ptr;
    }

    inline
    void pop_context()
    {
        auto bck = context_stack().back();
        delete bck;

        context_stack().pop_back();
    }

    inline
    execution_context_base& get_context()
    {
        return *context_stack().back();
    }

    inline
    value_base declare_e(execution_context_base& ectx, const std::string& name, const value_base& rhs)
    {
        ectx.add(declare_b(name, rhs));

        value_base named;
        named.type = op::VALUE;
        named.abstract_value = name;
        named.concrete = rhs.concrete;

        return named;
    }


    inline
    value_base declare_e(execution_context_base& ectx, const value_base& rhs)
    {
        return declare_e(get_context(), "decl_" + std::to_string(get_context().next_id()), rhs);
    }

    template<typename T>
    inline
    value<T> declare_e(execution_context_base& ectx, const std::string& name, const value<T>& rhs)
    {
        ectx.add(declare_b(name, rhs));

        value<T> named;
        named.type = op::VALUE;
        named.abstract_value = name;

        return named;
    }

    template<typename T>
    inline
    value<T> declare_e(execution_context_base& ectx, const value<T>& rhs)
    {
        return declare_e(get_context(), "decl_" + std::to_string(get_context().next_id()), rhs);
    }

    template<typename T, int N>
    inline
    auto declare_e(execution_context_base& ectx, const tensor<T, N>& rhs)
    {
        tensor<decltype(declare_e(ectx, T())), N> ret;

        for(int i=0; i < N; i++)
        {
            ret[i] = declare_e(ectx, rhs[i]);
        }

        return ret;
    }

    template<typename T>
    inline
    mut<value<T>> declare_mut_e(execution_context_base& ectx, const std::string& name, const value<T>& rhs)
    {
        mut<value<T>> val;
        val.set_from_constant(declare_e(ectx, name, rhs));
        return val;
    }

    template<typename T>
    inline
    mut<value<T>> declare_mut_e(execution_context_base& ectx, const value<T>& rhs)
    {
        mut<value<T>> val;
        val.set_from_constant(declare_e(ectx, rhs));
        return val;
    }

    template<typename T, int N>
    inline
    auto declare_mut_e(execution_context_base& ectx, const tensor<T, N>& rhs)
    {
        tensor<decltype(declare_mut_e(ectx, T())), N> ret;

        for(int i=0; i < N; i++)
        {
            ret[i] = declare_mut_e(ectx, rhs[i]);
        }

        return ret;
    }

    template<typename T>
    inline
    void assign_e(execution_context_base& e, const mut<T>& v1, const T& v2)
    {
        return e.add(assign_b(v1, v2));
    }

    template<typename T, int... N>
    inline
    void assign_e(execution_context_base& e, const tensor<mut<T>, N...>& v1, const tensor<T, N...>& v2)
    {
        return e.add(assign_b(v1, v2));
    }

    template<typename T, int... N>
    inline
    void assign_e(execution_context_base& e, const tensor<mut<T>, N...>& v1, const tensor<mut<T>, N...>& v2)
    {
        return e.add(assign_b(v1, v2));
    }

    template<typename T>
    inline
    void if_e(execution_context_base& ectx, const value<bool>& condition, T&& then)
    {
        ectx.add(if_b(condition));
        ectx.add(block_start_b());

        then();

        ectx.add(block_end_b());
    }

    template<typename T>
    inline
    void while_e(execution_context_base& ectx, const value<bool>& condition, T&& then)
    {
        ectx.add(while_b(condition));
        ectx.add(block_start_b());

        then();

        ectx.add(block_end_b());
    }

    template<typename T>
    inline
    void return_e(execution_context_base& ectx, const value<T>& v)
    {
        ectx.add(return_b(v));
    }

    inline
    void return_e(execution_context_base& ectx)
    {
        ectx.add(return_b());
    }

    inline
    void break_e(execution_context_base& ectx)
    {
        ectx.add(break_b());
    }

    template<typename T>
    inline
    void for_e(execution_context_base& ectx, const value<bool>& condition, const value_base& execute, T&& then)
    {
        ectx.add(for_b(condition, execute));
        ectx.add(block_start_b());

        then();

        ectx.add(block_end_b());
    }

    namespace single_source
    {
        inline
        value_base declare_e(const value_base& rhs)
        {
            return declare_e(get_context(), rhs);
        }

        template<typename T>
        inline
        value<T> declare_e(const std::string& name, const value<T>& rhs)
        {
            return declare_e(get_context(), name, rhs);
        }

        template<typename T>
        inline
        value<T> declare_e(const value<T>& rhs)
        {
            return declare_e(get_context(), rhs);
        }

        template<typename T>
        inline
        value<T> declare_e(const mut<value<T>>& rhs)
        {
            return declare_e(get_context(), rhs);
        }

        template<typename T, int N>
        inline
        auto declare_e(const tensor<T, N>& rhs)
        {
            return declare_e(get_context(), rhs);
        }

        template<typename T>
        inline
        mut<value<T>> declare_mut_e(const std::string& name, const value<T>& rhs)
        {
            return declare_mut_e(get_context(), name, rhs);
        }

        template<typename T>
        inline
        mut<value<T>> declare_mut_e(const std::string& name, const mut<value<T>>& rhs)
        {
            return declare_mut_e(get_context(), name, rhs);
        }

        template<typename T>
        inline
        mut<value<T>> declare_mut_e(const value<T>& rhs)
        {
            return declare_mut_e(get_context(), rhs);
        }

        template<typename T>
        inline
        mut<value<T>> declare_mut_e(const mut<value<T>>& rhs)
        {
            return declare_mut_e(get_context(), rhs);
        }

        template<typename T, int N>
        inline
        tensor<mut<T>, N> declare_mut_e(const tensor<T, N>& rhs)
        {
            return declare_mut_e(get_context(), rhs);
        }

        template<typename T>
        inline
        void assign_e(const mut<value<T>>& v1, const value<T>& v2)
        {
            return assign_e(get_context(), v1, v2);
        }

        template<typename T>
        inline
        void if_e(const value<bool>& condition, T&& then)
        {
            if_e(get_context(), condition, std::forward<T>(then));
        }

        template<typename T>
        inline
        void return_e(const value<T>& v)
        {
            return_e(get_context(), v);
        }

        inline
        void return_e()
        {
            return_e(get_context());
        }

        inline
        void break_e()
        {
            break_e(get_context());
        }

        template<typename T>
        inline
        void for_e(const value<bool>& condition, const value_base& execute, T&& then)
        {
            for_e(get_context(), condition, execute, std::forward<T>(then));
        }

        template<typename T>
        inline
        void while_e(const value<bool>& condition, T&& then)
        {
            while_e(get_context(), condition, std::forward<T>(then));
        }

        template<typename T>
        inline
        auto pin(T& in)
        {
            return get_context().pin(in);
        }

        template<typename T>
        inline
        auto alias(const T& check, const T& to)
        {
            return get_context().alias(check, to);
        }

        template<typename T>
        inline
        std::string join_with_delim(std::string delim, T&& val)
        {
            return delim + value_to_string(val);
        }

        template<typename... T>
        inline
        void print(std::string fmt, T&&... args)
        {
            value_base se;
            se.type = value_impl::op::SIDE_EFFECT;

            for(int i=0; i < (int)fmt.size(); i++)
            {
                if(fmt[i] == '\n')
                {
                    fmt[i] = '\\';
                    fmt.insert(fmt.begin() + i + 1, 'n');
                    i++;
                }
            }

            std::string str = "printf(\"" + fmt + "\"";

            if constexpr(sizeof...(args) > 0)
            {
                str += (... + join_with_delim(",", args));
            }

            str += ")";

            se.abstract_value = str;

            value_impl::get_context().add(se);
        }
    }

    template<typename T>
    std::string to_string_s(T v)
    {
        if constexpr(std::is_integral_v<T>)
            return std::to_string(v);
        else
        {
            char buf[100] = {};

            auto [ptr, ec] = std::to_chars(&buf[0], buf + 100, v, std::chars_format::fixed);

            auto str = std::string(buf, ptr);

            if(auto it = str.find('.'); it == std::string::npos) {
                str += ".";
            }

            return str;
        }
    }

    inline
    value<int> get_global_id(const value<int>& idx)
    {
        value<int> v;
        v.type = op::GET_GLOBAL_ID;
        v.args = {idx};

        return v;
    }

    namespace single_source
    {
        using ::value_impl::get_global_id;
    }

    std::string value_to_string(const value_base& v);

    #define NATIVE_OPS
    #define NATIVE_DIVIDE
    //#define NATIVE_RECIP

    inline
    bool valid_for_native_op(const value_base& in)
    {
        #ifndef NATIVE_OPS
        return false;
        #endif // NATIVE_OPS

        if(in.args.size() > 2)
            return false;

        if(in.args.size() == 2 && in.args[0].concrete.index() == in.args[1].concrete.index())
            return std::holds_alternative<float>(in.args[0].concrete);

        if(in.args.size() == 1)
            return std::holds_alternative<float>(in.args[0].concrete);

        return false;
    };

    inline
    std::optional<std::string> native_op(op::type type)
    {
        using namespace op;

        std::map<op::type, std::string> table {
            {SIN, "native_sin"},
            {COS, "native_cos"},
            {TAN, "native_tan"},
            {LOG, "native_log"},
            {LOG2, "native_log2"},
            {SQRT, "native_sqrt"},
            {INVERSE_SQRT, "native_rsqrt"},
            {EXP, "native_exp"},
        };

        if(auto it = table.find(type); it != table.end())
            return it->second;

        return std::nullopt;
    }

    ///handles function calls, and infix operators
    inline
    std::string function_call_or_infix(const value_base& v)
    {
        using namespace op;

        std::map<op::type, std::string> table {
            {PLUS, "+"},
            {MINUS, "-"},
            {UMINUS, "-"},
            {MULTIPLY, "*"},
            {DIVIDE, "/"},
            {MOD, "%"},

            {LT, "<"},
            {LTE, "<="},
            {EQ, "=="},
            {NEQ, "!="},
            {GT, ">"},
            {GTE, ">="},
            {NOT, "!"},
            {LOR, "||"},
            {LAND, "&&"},

            {FMA, "fma"},
            {SIN, "sin"},
            {COS, "cos"},
            {TAN, "tan"},
            {LOG, "log"},
            {LOG2, "log2"},
            {SQRT, "sqrt"},
            {INVERSE_SQRT, "rsqrt"},
            {EXP, "exp"},
            {FMOD, "fmod"},
            {ISFINITE, "isfinite"},
            {FABS, "fabs"},
            {SIGN, "sign"},
            {FLOOR, "floor"},
            {CEIL, "ceil"},
            {SINH, "sinh"},
            {COSH, "cosh"},
            {TANH, "tanh"},
            {ASINH, "asinh"},
            {ACOSH, "acosh"},
            {ATANH, "atanh"},
            {ASIN, "asin"},
            {ACOS, "acos"},
            {ATAN, "atan"},
            {ATAN2, "atan2"},
            {MIN, "min"},
            {MAX, "max"},
            {CLAMP, "clamp"},
            {POW, "pow"},

            {GET_GLOBAL_ID, "get_global_id"},
        };

        std::set<op::type> infix{PLUS, MINUS, MULTIPLY, DIVIDE, MOD, LT, LTE, EQ, GT, GTE, NEQ, LOR, LAND};

        auto get_table_op = [&](const value_base& in)
        {
            if(valid_for_native_op(in))
                return native_op(in.type).value_or(table.at(in.type));

            return table.at(in.type);
        };

        //generate (arg[0] op arg[1]) as a string
        if(infix.count(v.type)) {
            return "(" + value_to_string(v.args.at(0)) + get_table_op(v) + value_to_string(v.args.at(1)) + ")";
        }

        //otherwise, this is a function call
        std::string args;

        //join all the function arguments together separated by a comma
        for(int i=0; i < (int)v.args.size(); i++)
        {
            args += value_to_string(v.args[i]) + ",";
        }

        if(args.size() > 0)
            args.pop_back();

        return "(" + get_table_op(v) + "(" + args + "))";
    }

    inline
    std::string value_to_string(const value_base& v)
    {
        if(v.type == op::VALUE) {
            ///variable, eg a string like "xyzw"
            if(v.abstract_value.size() != 0) {
                return v.abstract_value;
            }

            ///constant literal
            return std::visit([]<typename T>(const T& in)
            {
                std::string suffix;

                if constexpr(std::is_same_v<T, float16>)
                    suffix = "h";
                if constexpr(std::is_same_v<T, float>)
                    suffix = "f";

                if constexpr(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
                {
                    //to_string_s is implemented in terms of std::to_chars, but always ends with a "." for floating point numbers, as 1234f is invalid syntax in OpenCL
                    if(in < 0)
                        return "(" + to_string_s(in) + suffix + ")";
                    else
                        return to_string_s(in) + suffix;
                }
                else
                {
                    return to_string_s(in) + suffix;
                }

            }, v.concrete);
        }

        //v1[v2]
        if(v.type == op::BRACKET)
        {
            int N = std::get<int>(v.args.at(1).concrete);

            if(N == 1)
            {
                return "(" + value_to_string(v.args.at(0)) + "[" + value_to_string(v.args.at(2)) + "])";
            }

            if(N == 2)
            {
                value_base x = v.args.at(2);
                value_base y = v.args.at(3);

                value_base dx = v.args.at(4);
                value_base dy = v.args.at(5);

                value_base index = y * dx + x;

                return "(" + value_to_string(v.args.at(0)) + "[" + value_to_string(index) + "])";
            }

            if(N == 3)
            {
                ///name, N, x, y, z, dx, dy, dz
                value_base x = v.args.at(2);
                value_base y = v.args.at(3);
                value_base z = v.args.at(4);

                value_base dx = v.args.at(5);
                value_base dy = v.args.at(6);
                value_base dz = v.args.at(7);

                value_base index = z * dy * dx + y * dx + x;

                return "(" + value_to_string(v.args.at(0)) + "[" + value_to_string(index) + "])";
            }

            assert(false);
        }

        if(v.type == op::DECLARE){
            //v1 v2 = v3;
            if(v.args.size() == 3)
                return value_to_string(v.args.at(0)) + " " + value_to_string(v.args.at(1)) + " = " + value_to_string(v.args.at(2));
            //v1 v2;
            else if(v.args.size() == 2)
                return value_to_string(v.args.at(0)) + " " + value_to_string(v.args.at(1));
        }

        if(v.type == op::DECLARE_ARRAY) {

            std::string arr = "[" + value_to_string(v.args.at(2)) + "]";

            std::string lhs = value_to_string(v.args.at(0)) + " " + value_to_string(v.args.at(1)) + arr + " = ";

            std::string rhs = "{";

            int num = std::get<int>(v.args.at(3).concrete);

            for(int i=0; i < num; i++)
            {
                std::string str = value_to_string(v.args.at(4 + i));

                rhs += str + ",";
            }

            if(rhs.back() == ',')
                rhs.pop_back();

            rhs += "}";

            return lhs + rhs;
        }

        if(v.type == op::BLOCK_START)
            return "{";

        if(v.type == op::BLOCK_END)
            return "}";

        if(v.type == op::IF)
            return "if(" + value_to_string(v.args.at(0)) + ")";

        if(v.type == op::BREAK)
            return "break";

        if(v.type == op::WHILE)
            return "while(" + value_to_string(v.args.at(0)) + ")";

        if(v.type == op::FOR)
            return "for(;" + value_to_string(v.args.at(0)) + ";" + value_to_string(v.args.at(1)) + ")";

        if(v.type == op::RETURN)
        {
            if(v.args.size() > 0)
                return "return " + value_to_string(v.args.at(0));
            else
                return "return";
        }

        if(v.type == op::ASSIGN)
            return value_to_string(v.args.at(0)) + " = " + value_to_string(v.args.at(1));

        if(v.type == op::DOT)
            return value_to_string(v.args.at(0)) + "." + value_to_string(v.args.at(1));

        auto type_to_suffix = [](const std::string& str)
        {
            if(str == "float")
                return "f";
            else if(str == "int")
                return "i";
            else if(str == "half")
                return "h";
            else if(str == "unsigned int")
                return "ui";
            else
                assert(false);
        };

        auto join = [](const std::vector<value_base>& b)
        {
            std::string str;

            for(const auto& i : b)
            {
                str += value_to_string(i) + ",";
            }

            if(str.size() > 0)
                str.pop_back();

            return str;
        };

        if(v.type == op::SAMPLER)
        {
            std::map<std::string, std::string> sampler_mapping
            {
                {"normalized_coords_true", "CLK_NORMALIZED_COORDS_TRUE"},
                {"normalized_coords_false", "CLK_NORMALIZED_COORDS_FALSE"},
                {"address_mirrored_repeat", "CLK_ADDRESS_MIRRORED_REPEAT"},
                {"address_repeat", "CLK_ADDRESS_REPEAT"},
                {"address_clamp_to_edge", "CLK_ADDRESS_CLAMP_TO_EDGE"},
                {"address_clamp", "CLK_ADDRESS_CLAMP"},
                {"address_none", "CLK_ADDRESS_NONE"},
                {"filter_nearest", "CLK_FILTER_NEAREST"},
                {"filter_linear", "CLK_FILTER_LINEAR"},
                {"none", "0"},
            };

            if(v.args.size() == 0)
                return "0";

            std::string str;

            for(int i=0; i < v.args.size(); i++)
            {
                std::string value = sampler_mapping.at(value_to_string(v.args[i]));

                if(i == (int)v.args.size() - 1)
                    str += value;
                else
                    str += value + "|";
            }

            return "(" + str + ")";
        }

        if(v.type == op::IMAGE_READ)
        {
            std::string name = value_to_string(v.args[0]);
            int num_args = std::get<int>(v.args[1].concrete);
            std::string type = value_to_string(v.args[2]);

            std::string suffix = type_to_suffix(type);

            std::string pos_type = value_to_string(v.args.at(3));

            std::vector<value_base> pos;

            for(int i=0; i < num_args; i++)
            {
                pos.push_back(v.args.at(4 + i));
            }

            if(num_args == 3)
            {
                pos.push_back(pos.back().make_constant_of_type(0));
                num_args = 4;
            }

            return "read_image" + suffix + "(" + name + ",(" + pos_type + std::to_string(num_args) + ")(" + join(pos) + "))";
        }

        if(v.type == op::IMAGE_READ_WITH_SAMPLER)
        {
            std::string name = value_to_string(v.args[0]);
            int num_args = std::get<int>(v.args[1].concrete);
            std::string type = value_to_string(v.args[2]);

            std::string suffix = type_to_suffix(type);

            std::string pos_type = value_to_string(v.args.at(3));
            std::string sampler = value_to_string(v.args.at(4));

            std::vector<value_base> pos;

            for(int i=0; i < num_args; i++)
            {
                pos.push_back(v.args.at(5 + i));
            }

            if(num_args == 3)
            {
                pos.push_back(pos.back().make_constant_of_type(0));
                num_args = 4;
            }

            return "read_image" + suffix + "(" + name + "," + sampler + ",(" + pos_type + std::to_string(num_args) + ")(" + join(pos) + "))";
        }

        ///name, N, position[0..N], write_data_type, M, write_data[0..M]
        if(v.type == op::IMAGE_WRITE)
        {
            int last_idx = 0;

            auto next = [&]()
            {
                return v.args.at(last_idx++);
            };

            std::string name = next().abstract_value;
            int dimensions = std::get<int>(next().concrete);

            std::vector<value_base> position;

            for(int i=0; i < dimensions; i++)
            {
                position.push_back(next());
            }

            std::string datatype = next().abstract_value;

            std::string suffix = type_to_suffix(datatype);

            int data_dimensions = std::get<int>(next().concrete);
            std::vector<value_base> data;

            for(int i=0; i < data_dimensions; i++)
            {
                data.push_back(next());
            }

            //the position for opencl image writes is either an int, int2, or an int4. There is no int3
            if(dimensions > 2)
            {
                for(int i=position.size(); i < 4; i++)
                {
                    position.push_back(value<int>(0));
                }
            }

            //must write 4 components to opencl
            for(int i=data.size(); i < 4; i++)
            {
                data.push_back(value<int>(0));
            }

            int opencl_N = dimensions > 2 ? 4 : dimensions;
            int opencl_D = data_dimensions > 2 ? 4 : data_dimensions;

            std::string position_str = "(int" + std::to_string(opencl_N) + "){" + join(position) + "}";
            std::string data_str = "(" + datatype + std::to_string(opencl_D) + "){" + join(data) + "}";

            return "write_image" + suffix + "(" + name + "," + position_str + "," + data_str + ")";
        }

        if(v.type == op::CAST)
        {
            std::string type = v.args.at(0).abstract_value;

            return "(" + type + ")(" + value_to_string(v.args.at(1)) + ")";
        }

        if(v.type == op::SIDE_EFFECT)
        {
            return v.abstract_value;
        }

        #ifdef NATIVE_RECIP
        if(v.type == op::DIVIDE)
        {
            if(equivalent(v.args[0], v.args[0].make_constant_of_type(1.f)))
                return "native_recip(" + value_to_string(v.args.at(1)) + ")";
        }
        #endif

        #ifdef NATIVE_DIVIDE
        if(v.type == op::DIVIDE && v.is_floating_point_type() && std::holds_alternative<float>(v.concrete))
        {
            return "native_divide(" + value_to_string(v.args.at(0)) + "," + value_to_string(v.args.at(1)) + ")";
        }
        #endif

        if(v.type == op::TERNARY)
        {
            ///ok so. igentype has to have the same size as gentype
            ///eg we do select(float, float, int)
            ///or select(half, half, short)
            ///or select(char, char, char)
            ///or select(long, long, long)
            ///or select(double, double, long)
            ///we're modelling ternary *not* select, so i know it can fit into these. Just need to find the right type

            assert(v.args.at(2).concrete.index() == v.args.at(1).concrete.index());

            auto type_callback = []<typename T>(const T& in) -> std::string
            {
                if constexpr(std::is_same_v<T, char>)
                    return "char";
                if constexpr(std::is_same_v<T, signed char>)
                    return "char";
                if constexpr(std::is_same_v<T, unsigned char>)
                    return "char";
                if constexpr(std::is_same_v<T, short>)
                    return "short";
                if constexpr(std::is_same_v<T, unsigned short>)
                    return "short";
                if constexpr(std::is_same_v<T, int>)
                    return "int";
                if constexpr(std::is_same_v<T, unsigned int>)
                    return "int";
                if constexpr(std::is_same_v<T, long>)
                    return "long";
                if constexpr(std::is_same_v<T, unsigned long>)
                    return "long";
                if constexpr(std::is_same_v<T, float16>)
                    return "short";
                if constexpr(std::is_same_v<T, float>)
                    return "int";
                if constexpr(std::is_same_v<T, double>)
                    return "long";

                assert(false);
                return "";
            };

            std::string type_of_selection_arg = std::visit(type_callback, v.args.at(2).concrete);

            ///our select is a ? b : c
            ///opencl's select is c ? b : a
            return "select(" + value_to_string(v.args.at(2)) + "," + value_to_string(v.args.at(1)) + ",(" + type_of_selection_arg + ")(" + value_to_string(v.args.at(0)) + "))";
        }

        ///todo: 32bits use atomic_, 64bits use atom_
        if(v.type == op::ATOM_ADD)
        {
            return "atom_add(" + value_to_string(v.args.at(0)) + "+" + value_to_string(v.args.at(1)) + "," + value_to_string(v.args.at(2)) + ")";
        }

        if(v.type == op::ATOM_XCHG)
        {
            return "atomic_xchg(" + value_to_string(v.args.at(0)) + "+" + value_to_string(v.args.at(1)) + "," + value_to_string(v.args.at(2)) + ")";
        }

        return function_call_or_infix(v);
    }

    struct type_storage;

    namespace single_source {
        struct argument_pack {
            template<typename Self>
            void add_struct(this Self&& self, type_storage& result)
            {
                self.build(result);
            }
        };

        struct declare_t{};

        static constexpr declare_t declare;

        template<typename T>
        inline
        value<T> build_type(const value_base& name, const T& tag)
        {
            value<T> ret;
            ret.set_from_base(name);
            return ret;
        }

        template<typename T>
        inline
        value<T> build_type(const value_base& name, const value<T>& tag)
        {
            value<T> ret;
            ret.set_from_base(name);
            return ret;
        }

        template<typename T, int N>
        inline
        tensor<value<T>, N> build_type(const value_base& name, const tensor<T, N>& tag)
        {
            tensor<value<T>, N> ret;

            for(int i=0; i < N; i++)
            {
                ret[i].type = op::DOT;
                ret[i].args = {name, "s" + std::to_string(i)};
            }

            return ret;
        }

        template<typename T, int N>
        inline
        tensor<value<T>, N> build_type(const value_base& name, const tensor<value<T>, N>& tag)
        {
            tensor<value<T>, N> ret;

            for(int i=0; i < N; i++)
            {
                ret[i].type = op::DOT;
                ret[i].args = {name, "s" + std::to_string(i)};
            }

            return ret;
        }

        template<typename T>
        inline
        mut<value<T>> apply_mutability(const value<T>& in)
        {
            mut<value<T>> ret;
            ret.set_from_constant(in);
            return ret;
        }

        template<typename T, int N>
        inline
        tensor<mut<value<T>>, N> apply_mutability(const tensor<value<T>, N>& in)
        {
            tensor<mut<value<T>>, N> ret;

            for(int i=0; i < N; i++)
            {
                ret[i] = apply_mutability(in[i]);
            }

            return ret;
        }

        template<typename T>
        struct array {
            std::string name;

            auto operator[](const value<int>& index)
            {
                value_base op;
                op.type = op::BRACKET;
                op.args = {name, value<int>(1), index};
                op.concrete = get_interior_type(T());

                return build_type(op, T());
            }
        };

        template<typename T>
        struct array_mut : array<T> {
            auto operator[](const value<int>& index)
            {
                return apply_mutability(array<T>::operator[](index));
            }
        };

        template<typename T>
        struct buffer {
            std::string name;
            using value_type = T;

            template<typename U>
            auto operator[](const value<U>& index)
            {
                value_base op;
                op.type = op::BRACKET;
                op.args = {name, value<int>(1), index};
                op.concrete = get_interior_type(T());

                return build_type(op, T());
            }

            auto operator[](int index)
            {
                return operator[](value<int>(index));
            }

            template<typename U>
            auto operator[](const tensor<U, 3>& pos, const tensor<U, 3>& dim)
            {
                value_base op;
                op.type = op::BRACKET;
                op.args = {name, value<int>(3), pos.x(), pos.y(), pos.z(), dim.x(), dim.y(), dim.z()};
                op.concrete = get_interior_type(T());

                return build_type(op, T());
            }

            template<typename U>
            auto operator[](const tensor<U, 2>& pos, const tensor<U, 2>& dim)
            {
                value_base op;
                op.type = op::BRACKET;
                op.args = {name, value<int>(2), pos.x(), pos.y(), dim.x(), dim.y()};
                op.concrete = get_interior_type(T());

                return build_type(op, T());
            }
        };

        template<typename T>
        struct buffer_mut : buffer<T> {
            using value_type = T;

            template<typename U>
            auto operator[](const value<U>& index)
            {
                static_assert(std::is_integral_v<U>);

                return apply_mutability(buffer<T>::operator[](index));
            }

            auto operator[](int index)
            {
                return operator[](value<int>(index));
            }

            template<typename U, int N>
            auto operator[](const tensor<U, N>& pos, const tensor<U, N>& dim)
            {
                return apply_mutability(buffer<T>::operator[](pos, dim));
            }

            T atom_add_b(const value<int>& index, const T& in)
            {
                value_base op;
                op.type = op::ATOM_ADD;
                op.args = {this->name, index, in};
                op.concrete = get_interior_type(T());

                return build_type(op, T());
            }

            T atom_add_e(const value<int>& index, const T& in)
            {
                return declare_e(get_context(), atom_add_b(index, in));
            }

            T atom_xchg_b(const value<int>& index, const T& in)
            {
                value_base op;
                op.type = op::ATOM_XCHG;
                op.args = {this->name, index, in};
                op.concrete = get_interior_type(T());

                return build_type(op, T());
            }

            T atom_xchg_e(const value<int>& index, const T& in)
            {
                return declare_e(get_context(), atom_xchg_b(index, in));
            }
        };

        template<typename T>
        struct literal {
            std::string name;
            using value_type = T;

            auto get()
            {
                value_base op;
                op.type = op::VALUE;
                op.abstract_value = name;
                op.concrete = get_interior_type(T());

                return build_type(op, T());
            }
        };

        namespace sampler_flag
        {
            static std::string NORMALIZED_COORDS_TRUE = "normalized_coords_true";
            static std::string NORMALIZED_COORDS_FALSE = "normalized_coords_false";

            static std::string ADDRESS_MIRRORED_REPEAT = "address_mirrored_repeat";
            static std::string ADDRESS_REPEAT = "address_repeat";
            static std::string ADDRESS_CLAMP_TO_EDGE = "address_clamp_to_edge";
            static std::string ADDRESS_CLAMP = "address_clamp";
            static std::string ADDRESS_NONE = "address_none";
            static std::string FILTER_NEAREST = "filter_nearest";
            static std::string FILTER_LINEAR = "filter_linear";

            static std::string NONE = "none";
        }

        template<int N>
        struct image {
            std::string name;

            virtual bool is_read_only() const {
                assert(false);
            }

            virtual bool is_image_array() const {
                return false;
            }
        };

        template<int N>
        struct image_array : image<N> {
            virtual bool is_image_array() const override {
                return true;
            }
        };

        template<int CoordinateDim, typename Base>
        struct read_only_image_base : Base {
            virtual bool is_read_only() const override {
                return true;
            }

            ///this is pretty basic as read/write and doesn't encompass the full set of functionality
            template<typename T, int M, typename U>
            tensor<value<T>, M> read(execution_context_base& ectx, const tensor<value<U>, CoordinateDim>& pos, const std::vector<std::string>& sampler = {}) const
            {
                value_base type = name_type(T());
                value_base pos_type = name_type(U());

                value_base single_read;

                if(sampler.size() == 0)
                    single_read.type = op::IMAGE_READ;
                else
                    single_read.type = op::IMAGE_READ_WITH_SAMPLER;

                single_read.args = {this->name, value<int>(pos.size()), type, pos_type};

                if(sampler.size() > 0)
                {
                    value_base sam;
                    sam.type = op::SAMPLER;

                    for(auto& i : sampler)
                    {
                        value_base arg = i;
                        sam.args.push_back(arg);
                    }

                    single_read.args.push_back(sam);
                }

                for(auto& i : pos)
                    single_read.args.push_back(i);

                value_base decl_type = name_type(tensor<T, 4>());
                value_base decl_name = "iv" + std::to_string(ectx.next_id());

                value_base decl;
                decl.type = op::DECLARE;
                decl.args = {decl_type, decl_name, single_read};

                ectx.add(std::move(decl));

                tensor<value<T>, M> ret;

                std::array<std::string, 4> dots = {"x", "y", "z", "w"};

                for(int i=0; i < M; i++)
                {
                    value_base component = dots[i];

                    value<T> dot;
                    dot.type = op::DOT;
                    dot.args = {decl_name, component};

                    ret[i] = dot;
                }

                return ret;
            }

            template<typename T, int M>
            tensor<value<T>, M> read(const tensor<value<int>, CoordinateDim>& pos, const std::vector<std::string>& sampler = {}) const
            {
                return read<T, M>(get_context(), pos, sampler);
            }

            template<typename T, int M, typename U>
            tensor<value<T>, M> read(const tensor<value<U>, CoordinateDim>& pos, const std::vector<std::string>& sampler = {}) const
            {
                return read<T, M>(get_context(), pos, sampler);
            }
        };

        template<int N>
        using read_only_image = read_only_image_base<N, image<N>>;

        template<int N>
        using read_only_image_array = read_only_image_base<N+1, image_array<N>>;

        template<int N, typename Base>
        struct write_only_image_base : Base {
            virtual bool is_read_only() const override {
                return false;
            }

            template<typename T, int M>
            void write(execution_context_base& ectx, const tensor<value<int>, N>& pos, const tensor<value<T>, M>& val) const
            {
                value_base write_op;
                write_op.type = value_impl::op::IMAGE_WRITE;

                write_op.args = {this->name};

                write_op.args.push_back(value<int>(N));

                for(int i=0; i < N; i++)
                    write_op.args.push_back(pos[i]);

                value_base data_type = name_type(T());

                write_op.args.push_back(data_type);
                write_op.args.push_back(value<int>(val.size()));

                for(int i=0; i < M; i++)
                {
                    write_op.args.push_back(val[i]);
                }

                ectx.add(std::move(write_op));
            }

            template<typename T, int M>
            void write(const tensor<value<int>, N>& pos, const tensor<value<T>, M>& val) const
            {
                return write(get_context(), pos, val);
            }
        };

        template<int N>
        using write_only_image = write_only_image_base<N, image<N>>;

        template<int N>
        using write_only_image_array = write_only_image_base<N+1, image_array<N>>;
    }

    template<typename T>
    inline
    single_source::array<T> declare_array_e(execution_context_base& ectx, const std::string& name, int size, const std::vector<T>& rhs)
    {
        ectx.add(declare_array_b<T>(name, size, {}));

        assert((int)rhs.size() <= size);

        single_source::array<T> out;
        out.name = name;

        for(int i=0; i < (int)rhs.size(); i++)
        {
            single_source::array_mut<T> temp;
            temp.name = name;

            assign_e(ectx, temp[i], rhs[i]);
        }

        return out;
    }

    template<typename T>
    inline
    single_source::array<T> declare_array_e(execution_context_base& ectx, int size, const std::vector<T>& rhs)
    {
        return declare_array_e<T>(ectx, "arr_" + std::to_string(get_context().next_id()), size, rhs);
    }

    template<typename T>
    inline
    single_source::array_mut<T> declare_mut_array_e(execution_context_base& ectx, int size, const std::vector<T>& rhs)
    {
        auto lbuf = declare_array_e<T>(ectx, size, rhs);

        single_source::array_mut<T> out;
        out.name = lbuf.name;
        return out;
    }

    namespace single_source {
        template<typename T>
        inline
        array<T> declare_array_e(int size, const std::vector<T>& rhs)
        {
            return declare_array_e<T>(get_context(), size, rhs);
        }

        template<typename T>
        inline
        array_mut<T> declare_mut_array_e(int size, const std::vector<T>& rhs)
        {
            return declare_mut_array_e<T>(get_context(), size, rhs);
        }
    }

    template<typename T>
    inline
    auto as_ref(const T& in)
    {
        return value_impl::mutable_proxy(in, get_context());
    }

    template<typename T>
    inline
    auto as_constant(const mut<T>& in)
    {
        return in.as_constant();
    }

    template<typename T, int... N>
    inline
    tensor<T, N...> as_constant(const tensor<mut<T>, N...>& v1)
    {
        return tensor_for_each_nary([&](const mut<T>& v1)
        {
            return v1.as_constant();
        }, v1);
    }

    struct input;

    struct type_storage
    {
        std::string suffix;
        std::vector<input> args;
        int placeholders = 0;
    };

    struct function_context
    {
        type_storage inputs;
    };

    struct input
    {
        std::vector<type_storage> defines_structs;

        std::string type;
        bool pointer = false;
        bool is_constant = false;
        bool is_image = false;
        bool is_image_array = false;
        int image_N = 0;
        std::string name;

        std::string format()
        {
            if(is_image)
            {
                std::string tag;

                if(is_constant)
                    tag = "__read_only";
                else
                    tag = "__write_only";

                std::string array_tag;

                if(is_image_array)
                    array_tag = "_array";

                ///eg __write_only image2d_array_t my_image
                return tag + " image" + std::to_string(image_N) + "d" + array_tag + "_t " + name;
            }

            if(pointer)
            {
                std::string cst = is_constant ? "const " : "";

                return "__global " + cst + type + "* __restrict__ " + name;
            }
            else
            {
                return type + " " + name;
            }
        }
    };

    namespace builder {
        using namespace single_source;

        template<typename T>
        inline
        void add(buffer<T>& buf, type_storage& result)
        {
            input in;
            in.type = name_type(T());
            in.pointer = true;
            in.is_constant =  true;

            std::string name = "buf" + std::to_string(result.args.size()) + result.suffix;

            in.name = name;
            buf.name = name;

            result.args.push_back(in);
        }

        template<typename T>
        inline
        void add(buffer_mut<T>& buf, type_storage& result)
        {
            input in;
            in.type = name_type(T());
            in.pointer = true;
            in.is_constant =  false;

            std::string name = "buf" + std::to_string(result.args.size()) + result.suffix;

            in.name = name;
            buf.name = name;

            result.args.push_back(in);
        }

        template<typename T>
        inline
        void add(literal<T>& lit, type_storage& result)
        {
            input in;
            in.type = name_type(T());
            in.pointer = false;

            std::string name = "lit" + std::to_string(result.args.size()) + result.suffix;

            in.name = name;
            lit.name = name;

            result.args.push_back(in);
        }

        template<int N>
        inline
        void add(single_source::image<N>& img, type_storage& result)
        {
            input in;
            in.type = "error";
            in.is_image = true;
            in.image_N = N;
            in.is_constant = img.is_read_only();
            in.is_image_array = img.is_image_array();

            std::string name = "img" + std::to_string(result.args.size()) + result.suffix;

            in.name = name;
            img.name = name;

            result.args.push_back(in);
        }

        template<typename T>
        requires std::is_base_of_v<single_source::argument_pack, T>
        inline
        void add(T& pack, type_storage& result)
        {
            pack.add_struct(result);
        }

        template<typename T, std::size_t N>
        inline
        void add(std::array<T, N>& arr, type_storage& result)
        {
            for(int i=0; i < (int)N; i++)
            {
                add(arr[i], result);
            }
        }

        struct placeholder
        {
            //ideally this would be a shared_ptr, but I don't really want to bring in
            //the whole header
            type_storage* storage = nullptr;
            int placeholder_index = 0;

            template<typename T>
            void add(T&& in)
            {
                assert(storage);
                value_impl::builder::add(std::forward<T>(in), *storage);
            }
        };

        inline
        void add(placeholder& ph, type_storage& result)
        {
            ph.storage = new type_storage;
            ph.placeholder_index = result.args.size();
            ph.storage->suffix = "_" + std::to_string(result.placeholders++);
        }
    }

    template<std::size_t N, typename... Ts>
    inline
    auto grab_N_from_tuple(std::tuple<Ts...> all_elements)
    {
        return [&]<std::size_t... I>(std::index_sequence<I...>)
        {
            return std::make_tuple(std::get<I>(all_elements)...);
        }(std::make_index_sequence<N>{});
    }

    template<typename R, typename T, typename... Args>
    inline
    auto split_args(R(*func)(T&, Args...))
    {
        return std::pair<T, std::tuple<Args...>>();
    }

    template<typename R, typename Type, typename T, typename... Args>
    inline
    auto split_args_lambda(R(Type::*func)(T&, Args...))
    {
        return std::pair<T, std::tuple<Args...>>();
    }

    template<typename R, typename Type, typename T, typename... Args>
    inline
    auto split_args_lambda(R(Type::*func)(T&, Args...) const)
    {
        return std::pair<T, std::tuple<Args...>>();
    }

    template<typename T>
    concept Functor = requires
    {
        &std::remove_reference_t<T>::operator();
    };

    template<typename Lambda>
    requires Functor<Lambda>
    inline
    auto split_args(Lambda&& l)
    {
        return split_args_lambda(&l.operator());
    }

    template<typename Callable, typename... U>
    inline
    void setup_kernel(Callable&& func, function_context& ctx, U&&... concrete_args)
    {
        //todo: need to only split a certain amount... keep splitting until
        //there are concrete_args remaining
        auto [ctx_type, args] = split_args(func);

        constexpr std::size_t num = std::tuple_size_v<decltype(args)>;
        constexpr std::size_t concrete_argc = sizeof...(concrete_args);

        auto args2 = grab_N_from_tuple<num - concrete_argc>(args);

        using T = std::remove_reference_t<decltype(ctx_type)>;

        T& ectx = push_context<T>();

        std::apply([&](auto&&... expanded_args){
            (builder::add(expanded_args, ctx.inputs), ...);
        }, args2);

        std::tuple<T&> a1 = {ectx};
        std::tuple<U...> a2 = {concrete_args...};

        std::apply(func, std::tuple_cat(a1, args2, a2));

        int running_placeholder_offsets = 0;

        auto resolve_placeholder = [&]<typename T>(T&& in)
        {
            if constexpr(std::is_same_v<std::decay_t<T>, builder::placeholder>)
            {
                builder::placeholder& ph = in;
                assert(ph.storage);

                printf("Adding %i\n", ph.storage->args.size());

                ctx.inputs.args.insert(ctx.inputs.args.begin() + ph.placeholder_index + running_placeholder_offsets, ph.storage->args.begin(), ph.storage->args.end());
                running_placeholder_offsets += ph.storage->args.size();

                delete ph.storage;
                ph.storage = nullptr;
            }
        };

        std::apply([&](auto&&... expanded_args){
            (resolve_placeholder(expanded_args), ...);
        }, args2);
    }

    inline
    void substitute(const value_base& what, const value_base& with, value_base& modify)
    {
        if(equivalent(what, modify))
        {
            modify = with;
            return;
        }

        for(int i=0; i < (int)modify.args.size(); i++)
        {
            substitute(what, with, modify.args[i]);
        }
    }

    inline
    bool expression_present_in(const value_base& expression, const value_base& in)
    {
        if(equivalent(expression, in))
            return true;

        for(int i=0; i < (int)in.args.size(); i++)
        {
            if(expression_present_in(expression, in.args[i]))
                return true;
        }

        return false;
    }

    ///issue is that we're peeking throuhg expressions which should not be peeked through, ie operator.
    inline
    void build_common_subexpressions_between(const value_base& v1, const value_base& v2, std::vector<value_base>& out)
    {
        auto invalid_type = [](op::type t) {
            return t == op::VALUE || t == op::DOT || t == op::BRACKET ||
            t == op::FOR || t == op::IF || t == op::BREAK || t == op::WHILE ||
            t == op::RETURN || t == op::BLOCK_START || t == op::BLOCK_END ||
            t == op::ASSIGN || t == op::SIDE_EFFECT || t == op::IMAGE_READ || t == op::IMAGE_WRITE || t == op::CAST;
        };

        if(invalid_type(v1.type))
            return;

        if(v1.type != op::DECLARE)
        {
            if(expression_present_in(v1, v2))
            {
                for(auto& j : out)
                {
                    if(equivalent(v1, j))
                        return;
                }

                out.push_back(v1);
                return;
            }
        }

        int start = 0;

        if(v1.type == op::DECLARE)
            start = 2;

        for(int i=start; i < (int)v1.args.size(); i++)
        {
            if(invalid_type(v1.args[i].type))
                continue;

            if(expression_present_in(v1.args[i], v2))
            {
                bool already_found = false;

                for(auto& j : out)
                {
                    if(equivalent(v1.args[i], j))
                    {
                        already_found = true;
                        break;
                    }
                }

                if(!already_found)
                    out.push_back(v1.args[i]);
            }
            else
            {
                build_common_subexpressions_between(v1.args[i], v2, out);
            }
        }
    }

    ///so we want to go top down in our tree, and for each node, check all the other expressions to see
    ///if they contain that same expression
    ///if they do, we want to declare a new variable, and then yank us out

    inline
    std::vector<value_base> expression_eliminate(execution_context_base& ectx, const std::vector<value_base>& in)
    {
        std::vector<value_base> current_expression_list = in;

        std::vector<value_base> ret;

        for(int i=0; i < (int)current_expression_list.size(); i++)
        {
            std::vector<value_base> exprs;

            for(int j=i+1; j < (int)current_expression_list.size(); j++)
            {
                build_common_subexpressions_between(current_expression_list[i], current_expression_list[j], exprs);
            }

            ///so, at i, its safe to common expression to a declaration just before i

            for(int kk=0; kk < (int)exprs.size(); kk++)
            {
                std::string name = "d" + std::to_string(ectx.next_id());

                value_base decl = value_impl::declare_b(name, exprs[kk]);

                value_base var = name;
                var.concrete = decl.concrete;

                current_expression_list.insert(current_expression_list.begin() + i + kk, decl);

                for(int jj = i + kk + 1; jj < current_expression_list.size(); jj++)
                {
                    substitute(exprs[kk], var, current_expression_list[jj]);
                }
            }

            i += exprs.size();

            ///so need to be mindful of the following pattern
            ///float v1 = whatever;
            ///float v2 = whatever * v1;
            ///float v3 = whatever * v1;
            ///if we pull out an expr involving v1, then we have to be careful with where we declare it
            ///i mean hey, inherently v1 can't contain a reference to itself in a declaration (unlike mutability), so we know if we pull out a common subexpression
            ///involving it where it can be placed
        }

        return current_expression_list;

        /*std::vector<value_base> ret = in;
        std::vector<value_base> exprs;
        std::vector<value_base> decls;

        for(int i=0; i < (int)in.size(); i++)
        {
            for(int j=i+1; j < (int)in.size(); j++)
            {
                build_common_subexpressions_between(in[i], in[j], exprs);
            }
        }

        for(auto& e : exprs)
        {
            std::string name = "d" + std::to_string(ectx.next_id());

            value_base decl = value_impl::declare_b(name, e);

            std::cout <<" Fdecl " << value_to_string(decl) << std::endl;

            value_base var = name;
            var.concrete = decl.concrete;

            decls.push_back(decl);

            for(auto& kk : ret)
            {
                substitute(e, var, kk);
            }
        }

        return {ret, decls};*/
    }

    inline
    std::string generate_kernel_string(function_context& kctx, const std::string& kernel_name)
    {
        execution_context_base& ctx = get_context();

        std::string base;

        base += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n#pragma OPENCL FP_CONTRACT ON\n\n";

        base += "__kernel void " + kernel_name + "(";

        for(int i=0; i < (int)kctx.inputs.args.size(); i++)
        {
            base += kctx.inputs.args[i].format();

            if(i != (int)kctx.inputs.args.size() - 1)
                base += ",";
        }

        base += ")\n{\n";

        auto is_semicolon_terminated = [](op::type t)
        {
            return t == op::ASSIGN || t == op::DECLARE || t == op::DECLARE_ARRAY || t == op::RETURN || t == op::BREAK || t == op::IMAGE_READ || t == op::IMAGE_WRITE || t == op::SIDE_EFFECT;
        };

        std::vector<std::vector<value_base>> blocks;
        blocks.emplace_back();

        #define ALIASES
        #ifdef ALIASES
        for(auto& v : ctx.to_execute)
        {
            for(auto& [check, sub] : ctx.aliases)
            {
                v.recurse([&](value_base& in)
                {
                    if(equivalent(in, check))
                    {
                        in = sub;
                    }
                });
            }
        }
        #endif

        for(auto& v : ctx.to_execute)
        {
            auto introduces_block = [](op::type t) {
                return t == op::BLOCK_START || t == op::BLOCK_END || t == op::IF || t == op::WHILE || t == op::FOR || t == op::BREAK || t == op::RETURN || t == op::SIDE_EFFECT ||
                       t == op::BRACKET || t == op::ASSIGN;
            };

            if(introduces_block(v.type))
            {
                blocks.emplace_back();
            }

            blocks.back().push_back(v);

            if(introduces_block(v.type))
            {
                blocks.emplace_back();
            }
        }

        int indentation = 1;

        int bid = 0;

        for(auto& block : blocks)
        {
            //#define ELIMINATE_SUBEXPRESSIONS
            #ifdef ELIMINATE_SUBEXPRESSIONS
            auto next_block = expression_eliminate(get_context(), block);
            #else
            auto next_block = block;
            #endif

            for(const value_base& v : next_block)
            {
                if(v.type == op::BLOCK_END)
                    indentation--;

                std::string prefix(indentation * 4, ' ');

                base += prefix + "//" + std::to_string(bid) + "\n";

                if(is_semicolon_terminated(v.type))
                    base += prefix + value_to_string(v) + ";\n";
                else
                    base += prefix + value_to_string(v) + "\n";

                if(v.type == op::BLOCK_START)
                    indentation++;
            }

            bid++;
        }

        base += "\n}\n";

        return base;
    }


    template<typename T, typename... U>
    inline
    std::string make_function(T&& in, const std::string& kernel_name, U&&... args)
    {
        function_context kctx;
        setup_kernel(in, kctx, std::forward<U>(args)...);

        std::string str = generate_kernel_string(kctx, kernel_name);

        pop_context();
        return str;
    }

    using namespace single_source;
}

using execution_context = value_impl::execution_context;
namespace single_source = value_impl::single_source;

template<typename T>
using buffer = value_impl::buffer<T>;

template<typename T>
using buffer_mut = value_impl::buffer_mut<T>;

template<typename T>
using literal = value_impl::literal<T>;

template<int N>
using image = value_impl::image<N>;

template<int N>
using read_only_image = value_impl::single_source::read_only_image<N>;

template<int N>
using write_only_image = value_impl::single_source::write_only_image<N>;

template<int N>
using read_only_image_array = value_impl::single_source::read_only_image_array<N>;

template<int N>
using write_only_image_array = value_impl::single_source::write_only_image_array<N>;

#endif // SINGLE_SOURCE_HPP_INCLUDED

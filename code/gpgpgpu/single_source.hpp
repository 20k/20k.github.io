#ifndef SINGLE_SOURCE_HPP_INCLUDED
#define SINGLE_SOURCE_HPP_INCLUDED

#include "value.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <assert.h>
#include <set>
#include <map>

namespace value_impl
{
    struct execution_context_base
    {
        std::vector<value_base> to_execute;
        virtual void add(const value_base& in) = 0;
        virtual int next_id() = 0;

        template<typename T, int... N>
        void add(const tensor<T, N...>& inout)
        {
            inout.for_each([&](const T& v)
            {
                add(v);
            });
        }

        template<typename T>
        void pin(value<T>& inout)
        {
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

        virtual ~execution_context_base(){}
    };

    struct execution_context : execution_context_base
    {
        int id = 0;

        void add(const value_base& in) override
        {
            to_execute.push_back(in);
        }

        int next_id() override
        {
            return id++;
        }
    };

    inline
    std::vector<execution_context_base*>& context_stack()
    {
        static thread_local std::vector<execution_context_base*> ctx;
        return ctx;
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
        mut<value<T>> declare_mut_e(const value<T>& rhs)
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
        auto pin(T& in)
        {
            return get_context().pin(in);
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

    std::string value_to_string(const value_base& v);

    //#define NATIVE_OPS
    //#define NATIVE_DIVIDE

    ///handles function calls, and infix operators
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
            {GT, ">"},
            {GTE, ">="},
            {NEQ, "!="},
            {NOT, "!"},
            {LOR, "||"},

            #ifdef NATIVE_OPS
            {SIN, "native_sin"},
            {COS, "native_cos"},
            {TAN, "native_tan"},
            {SQRT, "native_sqrt"},
            #else
            {SIN, "sin"},
            {COS, "cos"},
            {TAN, "tan"},
            {SQRT, "sqrt"},
            #endif // NATIVE_OPS
            {FMOD, "fmod"},
            {ISFINITE, "isfinite"},

            {GET_GLOBAL_ID, "get_global_id"},
        };

        std::set<op::type> infix{PLUS, MINUS, MULTIPLY, DIVIDE, MOD, LT, LTE, EQ, GT, GTE, NEQ, LOR};

        //generate (arg[0] op arg[1]) as a string
        if(infix.count(v.type)) {
            return "(" + value_to_string(v.args.at(0)) + table.at(v.type) + value_to_string(v.args.at(1)) + ")";
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

        return "(" + table.at(v.type) + "(" + args + "))";
    }

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

                //to_string_s is implemented in terms of std::to_chars, but always ends with a "." for floating point numbers, as 1234f is invalid syntax in OpenCL
                if(in < 0)
                    return "(" + to_string_s(in) + suffix + ")";
                else
                    return to_string_s(in) + suffix;
            }, v.concrete);
        }

        //v1[v2]
        if(v.type == op::BRACKET)
            return "(" + value_to_string(v.args.at(0)) + "[" + value_to_string(v.args.at(1)) + "])";

        if(v.type == op::DECLARE){
            //v1 v2 = v3;
            if(v.args.size() == 3)
                return value_to_string(v.args.at(0)) + " " + value_to_string(v.args.at(1)) + " = " + value_to_string(v.args.at(2));
            //v1 v2;
            else if(v.args.size() == 2)
                return value_to_string(v.args.at(0)) + " " + value_to_string(v.args.at(1));
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

        if(v.type == op::IMAGE_READ)
        {
            std::string type = value_to_string(v.args[2]);
            std::string name = value_to_string(v.args[0]);

            std::string suffix = type_to_suffix(type);

            int num_args = std::get<int>(v.args[1].concrete);

            std::string pos_type = value_to_string(v.args.at(3));

            std::vector<value_base> pos;

            for(int i=0; i < num_args; i++)
            {
                pos.push_back(v.args.at(4 + i));
            }

            return "read_image" + suffix + "(" + name + ",(" + pos_type + std::to_string(num_args) + ")(" + join(pos) + "))";
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

        #ifdef NATIVE_DIVIDE
        if(v.type == op::DIVIDE)
        {
            return "native_divide(" + value_to_string(v.args.at(0)) + "," + value_to_string(v.args.at(1)) + ")";
        }
        #endif

        return function_call_or_infix(v);
    }

    namespace single_source {
        template<typename T>
        struct buffer {
            std::string name;
            using value_type = T;

            value<T> operator[](const value<int>& index)
            {
                value<T> op;
                op.type = op::BRACKET;
                op.args = {name, index};

                return op;
            }
        };

        template<typename T>
        struct buffer_mut : buffer<T> {
            std::string name;
            using value_type = T;

            mut<value<T>> operator[](const value<int>& index)
            {
                mut<value<T>> res;
                res.set_from_constant(buffer<T>::operator[](index));
                return res;
            }
        };

        template<typename T>
        struct literal {
            std::string name;
            using value_type = T;

            value<T> get()
            {
                value<T> val;
                val.abstract_value = name;
                return val;
            }
        };

        template<int N>
        struct image {
            std::string name;
        };

        template<int N>
        struct read_only_image : image<N> {
            ///this is pretty basic as read/write and doesn't encompass the full set of functionality
            template<typename T, int M>
            tensor<value<T>, M> read(execution_context_base& ectx, const tensor<value<int>, N>& pos) const
            {
                value_base type = name_type(T());
                value_base pos_type = std::string("int");

                value_base single_read;
                single_read.type = op::IMAGE_READ;
                single_read.args = {this->name, value<int>(N), type, pos_type};

                for(auto& i : pos)
                    single_read.args.push_back(i);

                value_base decl_type = name_type(tensor<T, 4>());
                value_base decl_name = "iv" + std::to_string(ectx.next_id());

                value_base decl;
                decl.type = op::DECLARE;
                decl.args = {decl_type, decl_name, single_read};

                ectx.add(decl);

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
            tensor<value<T>, M> read(const tensor<value<int>, N>& pos) const
            {
                return read<T, M>(get_context(), pos);
            }
        };

        template<int N>
        struct write_only_image : image<N> {
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

                ectx.add(write_op);
            }

            template<typename T, int M>
            void write(const tensor<value<int>, N>& pos, const tensor<value<T>, M>& val) const
            {
                return write(get_context(), pos, val);
            }
        };
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

    template<typename T>
    using buffer = single_source::buffer<T>;
    template<typename T>
    using buffer_mut = single_source::buffer_mut<T>;
    template<typename T>
    using literal = single_source::literal<T>;
    template<int N>
    using image = single_source::image<N>;
    template<int N>
    using read_only_image = single_source::read_only_image<N>;
    template<int N>
    using write_only_image = single_source::write_only_image<N>;

    struct input;

    struct type_storage
    {
        std::vector<input> args;
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
        int image_N = 0;
        std::string name;

        std::string format()
        {
            if(is_image)
            {
                if(is_constant)
                    return "__read_only image" + std::to_string(image_N) + "d_t " + name;
                else
                    return "__write_only image" + std::to_string(image_N) + "d_t " + name;
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

    namespace impl {
        template<typename T>
        void add(buffer<T>& buf, type_storage& result)
        {
            input in;
            in.type = name_type(T());
            in.pointer = true;
            in.is_constant =  true;

            std::string name = "buf" + std::to_string(result.args.size());

            in.name = name;
            buf.name = name;

            result.args.push_back(in);
        }

        template<typename T>
        void add(buffer_mut<T>& buf, type_storage& result)
        {
            input in;
            in.type = name_type(T());
            in.pointer = true;
            in.is_constant =  false;

            std::string name = "buf" + std::to_string(result.args.size());

            in.name = name;
            buf.name = name;

            result.args.push_back(in);
        }

        template<typename T>
        void add(literal<T>& lit, type_storage& result)
        {
            input in;
            in.type = name_type(T());
            in.pointer = false;

            std::string name = "lit" + std::to_string(result.args.size());

            in.name = name;
            lit.name = name;

            result.args.push_back(in);
        }

        template<int N>
        void add(single_source::read_only_image<N>& img, type_storage& result)
        {
            input in;
            in.type = "error";
            in.is_image = true;
            in.is_constant = true;
            in.image_N = N;

            std::string name = "img" + std::to_string(result.args.size());

            in.name = name;
            img.name = name;

            result.args.push_back(in);
        }

        template<int N>
        void add(single_source::write_only_image<N>& img, type_storage& result)
        {
            input in;
            in.type = "error";
            in.is_image = true;
            in.image_N = N;

            std::string name = "img" + std::to_string(result.args.size());

            in.name = name;
            img.name = name;

            result.args.push_back(in);
        }
    }

    template<typename T, typename R, typename... Args>
    void setup_kernel(R(*func)(T&, Args...), function_context& ctx)
    {
        T& ectx = push_context<T>();

        std::tuple<std::remove_reference_t<Args>...> args;

        std::apply([&](auto&&... expanded_args){
            (impl::add(expanded_args, ctx.inputs), ...);
        }, args);

        std::tuple<T&> a1 = {ectx};

        std::apply(func, std::tuple_cat(a1, args));
    }

    std::string generate_kernel_string(function_context& kctx, const std::string& kernel_name)
    {
        execution_context_base& ctx = get_context();

        std::string base;

        base += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n#pragma OPENCL FP_CONTRACT OFF\n\n";

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
            return t == op::ASSIGN || t == op::DECLARE || t == op::RETURN || t == op::BREAK || t == op::IMAGE_READ || t == op::IMAGE_WRITE;
        };

        int indentation = 1;

        int bid = 0;

        for(const value_base& v : ctx.to_execute)
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

            bid++;
        }

        base += "\n}\n";

        return base;
    }


    template<typename T>
    std::string make_function(T&& in, const std::string& kernel_name)
    {
        function_context kctx;
        setup_kernel(in, kctx);

        std::string str = generate_kernel_string(kctx, kernel_name);

        pop_context();
        return str;
    }
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

#endif // SINGLE_SOURCE_HPP_INCLUDED

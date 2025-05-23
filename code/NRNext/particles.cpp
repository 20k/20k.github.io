#include "particles.hpp"
#include "integration.hpp"
#include "bssn.hpp"
#include "init_general.hpp"
#include "formalisms.hpp"
#include "raytrace_init.hpp"
#include "interpolation.hpp"
#include "../common/vec/dual.hpp"
#include <vec/stdmath.hpp>

template<typename T>
using dual = dual_types::dual_v<T>;

///https://arxiv.org/pdf/1611.07906.pdf (20)
template<typename T>
T dirac_delta(const T& frac_0_1, const T& full_world_radius)
{
    //frac is 0, 1
    //but the dirac delta function really takes 0 -> 2
    T frac = frac_0_1 * 2;

    T branch_1 = (1.f/4.f) * pow(2.f - frac, 3.f);
    T branch_2 = 1.f - (3.f/2.f) * pow(frac, 2.f) + (3.f/4.f) * pow(frac, 3.f);

    T mult = 1/(M_PI * pow(full_world_radius/2, 3.f));

    T result = stdmath::uternary(frac <= 2, branch_1, T{0.f});
    result = stdmath::uternary(frac <= 1, branch_2, result);
    return mult * result;
}

template<typename T>
inline
T get_dirac(auto&& func, tensor<T, 3> cell_pos, tensor<T, 3> dirac_location, T radius_cells, T scale)
{
    //#define GET_DIRAC1_STANDARD
    #ifdef GET_DIRAC1_STANDARD
    return func((cell_pos - dirac_location).length() / radius_cells, radius_cells * scale);
    #endif // GET_DIRAC_STANDARD

    #define GET_DIRAC1_CORRECTED
    #ifdef GET_DIRAC1_CORRECTED
    tensor<T, 3> scale3 = {1, 1, 1};

    auto im1 = cell_pos - scale3 / 2;
    auto ip1 = cell_pos + scale3 / 2;

    return integrate_3d_trapezoidal([&](T x, T y, T z)
    {
        tensor<T, 3> pos = {x, y, z};

        T frac = (pos - dirac_location).length() / radius_cells;

        return func(frac, radius_cells * scale);
    }, 2, ip1, im1);
    #endif // GET_DIRAC_CORRECTED
}

inline
valuef get_dirac3(auto&& func, const v3f& cell_pos, const v3f& dirac_location, const valuef& radius_cells, const valuef& scale)
{
    using namespace single_source;

    //#define GET_DIRAC_STANDARD
    #ifdef GET_DIRAC_STANDARD
    valuef r = (cell_pos - dirac_location).length();
    //pin(r);
    return func(r / radius_cells, radius_cells * scale);
    #endif // GET_DIRAC_STANDARD

    #define GET_DIRAC_CORRECTED
    #ifdef GET_DIRAC_CORRECTED
    tensor<valuef, 3> scale3 = {1, 1, 1};

    auto im1 = cell_pos - scale3 / 2;
    auto ip1 = cell_pos + scale3 / 2;
    pin(im1);
    pin(ip1);

    return integrate_3d_trapezoidal([&](const valuef& x, const valuef& y, const valuef& z)
    {
        tensor<valuef, 3> pos = {x, y, z};
        pin(pos);

        valuef r = (pos - dirac_location).length();
        pin(r);

        valuef frac = r / radius_cells;

        valuef out = func(frac, radius_cells * scale);
        pin(out);
        return out;
    }, 2, ip1, im1);
    #endif // GET_DIRAC_CORRECTED
}

//so. det(cA) = c^n det(A). For us, c^3
//in a conformally flat spacetime, det(cfl) = 1
//we have the quantity E = mass dirac lorentz / sqrt(det(Y))
//we're calculating the quantity E * sqrt(det(Y))
//need to calculate what sqrt(det(Y)) is in terms of phis
//Y_ij = phi^4 cfl_flat
//det(Y_ij) = det(phi^4 cfl_flat)
//det(Y_ij) = (phi^4)^3 = phi^12?
//sqrt(det(Y_ij)) = phi^12^0.5 = phi^6
//E = non_cfl_E  phi^-6

//ok I was just wrong
//cY = W^2 Yij
//det(Yij) = det(cY W^-2)
//det(Yij) = det(cY) (W^-2)^3
//= W^-6
//sqrt(det(Yij)) = W^-3

//det(cA) = c^n det(A). For us, c^3
//Yij = W^2 cY
//det(Yij) = det(W^2 cY)
//det(Yij) = W^6 det(cY)
//sqrt(det(Yij)) = W^6^0.5
//= W^3

void for_each_dirac(v3i cell, v3i dim, valuef scale, v3f dirac_pos, auto&& func)
{
    v3f fpos = world_to_grid(dirac_pos, dim, scale);

    using namespace single_source;

    ///minimum perf floor is 190, and that's achieved with radius_cells = 0
    int radius_cells = 3;

    if(radius_cells > 0)
    {
        //The appropriate modification is rightwards + 1, leftwards + 0
        int spread = radius_cells + 1;

        mut<valuei> z = declare_mut_e(valuei(-radius_cells));

        for_e(z <= spread, assign_b(z, z+1), [&]{
            mut<valuei> y = declare_mut_e(valuei(-radius_cells));

            for_e(y <= spread, assign_b(y, y+1), [&]{
                mut<valuei> x = declare_mut_e(valuei(-radius_cells));

                for_e(x <= spread, assign_b(x, x+1), [&]{
                    v3i offset = {declare_e(x), declare_e(y), declare_e(z)};
                    offset += cell;
                    pin(offset);

                    valuef dirac = get_dirac3(dirac_delta<valuef>, (v3f)offset, fpos, radius_cells, scale);
                    pin(dirac);

                    if_e(dirac > 0, [&]{
                        func(offset, dirac);
                    });
                });
            });
        });
    }
    else
    {
        func((v3i)floor(fpos), 1.f);
    }
}

//https://arxiv.org/pdf/1611.07906 16
void calculate_particle_nonconformal_E(execution_context& ectx, particle_base_args<buffer<valuef>> particles_in,
                                       buffer_mut<valuei64> nonconformal_E_out,
                                       literal<v3i> dim, literal<valuef> scale, literal<value<size_t>> particle_count,
                                       literal<valued> fixed_scale)
{
    using namespace single_source;

    value<size_t> id = value_impl::get_global_id_us(0);
    pin(id);

    if_e(id >= particle_count.get(), [&]{
        return_e();
    });

    valuef lorentz = 1;
    valuef mass = particles_in.get_mass(id);
    v3f pos = particles_in.get_position(id);
    //v3f vel = particles_in.get_velocity(id);

    pin(mass);
    pin(pos);
    //pin(vel);

    v3i cell = (v3i)floor(world_to_grid(pos, dim.get(), scale.get()));
    pin(cell);

    for_each_dirac(cell, dim.get(), scale.get(), pos, [&](v3i offset, valuef dirac)
    {
        valuef fin_E = mass * lorentz * dirac;

        auto scale = [&](valuef in)
        {
            valued in_d = (valued)in;
            valued in_scaled = in_d * fixed_scale.get();

            return (valuei64)round((valuef)in_scaled);
        };

        valuei64 as_i64 = scale(fin_E);

        valuei idx = offset.z() * dim.get().y() * dim.get().x() + offset.y() * dim.get().x() + offset.x();

        nonconformal_E_out.atom_add_e(idx, as_i64);
    });
}

void sum_E(execution_context& ectx,
            literal<v3i> idim,
            buffer<valuef> E_in,
            literal<valuei> positions_length,
            literal<valuef> scale, buffer_mut<value<std::int64_t>> sum)
{
    using namespace single_source;

    valuei lid = value_impl::get_global_id(0);

    pin(lid);

    v3i dim = idim.get();

    if_e(lid >= positions_length.get(), []{
        return_e();
    });

    v3i pos = get_coordinate_including_boundary(lid, dim, 0);
    pin(pos);

    valuef E = E_in[pos, dim];

    /*if_e(E != 0, [&]{
        print("Hi E %f\n", E);
    });*/

    valued as_double = (valued)E * pow(10., 12.) * (valued)pow(scale.get(), 3.f);

    value<std::int64_t> as_uint = (value<std::int64_t>)as_double;

    sum.atom_add_e(0, as_uint);
}

void calculate_particle_intermediates(execution_context& ectx,
                                      bssn_args_mem<buffer<valuef>> in,
                                      particle_base_args<buffer<valuef>> particles_in,
                                      buffer<valuef> lorentz_in,
                                      particle_utility_args<buffer_mut<valuei64>> out,
                                      literal<v3i> dim, literal<valuef> scale, literal<value<size_t>> particle_count,
                                      literal<valued> fixed_scale)
{
    using namespace single_source;

    value<size_t> id = value_impl::get_global_id_us(0);
    pin(id);

    if_e(id >= particle_count.get(), [&]{
        return_e();
    });

    valuef lorentz = lorentz_in[id];
    valuef mass = particles_in.get_mass(id);
    v3f pos = particles_in.get_position(id);
    v3f vel = particles_in.get_velocity(id);

    pin(pos);
    pin(vel);
    pin(mass);
    pin(lorentz);

    if_e(!isfinite(mass) || mass == 0, [&]{
        return_e();
    });

    v3f fcell = world_to_grid(pos, dim.get(), scale.get());
    v3i cell = (v3i)floor(fcell);
    pin(cell);

    for_each_dirac(cell, dim.get(), scale.get(), pos, [&](v3i offset, valuef dirac) {
        bssn_args args(offset, dim.get(), in);

        valuef E = mass * lorentz * args.gA * pow(args.W, 3) * dirac;
        v3f Ji = mass * vel * pow(args.W, 3) * dirac;

        tensor<valuef, 3, 3> Sij;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                Sij[i, j] = (mass * vel[i] * vel[j] / lorentz) * pow(args.W, 3) * dirac / max(args.gA, 0.01f);
            }
        }

        //print("Mass by scale %f\n", (valued)mass * fixed_scale.get());

        std::array<valuef, 6> Sij_sym = extract_symmetry(Sij);

        auto scale = [&](valuef in)
        {
            valued in_d = (valued)in;
            valued in_scaled = in_d * fixed_scale.get();

            return (valuei64)round((valuef)in_scaled);
        };

        valuei64 E_scaled = scale(E);

        tensor<valuei64, 3> Si_scaled;

        for(int i=0; i < 3; i++)
            Si_scaled[i] = scale(Ji[i]);

        std::array<valuei64, 6> Sij_scaled;

        for(int i=0; i < 6; i++)
            Sij_scaled[i] = scale(Sij_sym[i]);

        ///[offset, dim.get()]

        valuei idx = offset.z() * dim.get().y() * dim.get().x() + offset.y() * dim.get().x() + offset.x();

        valuei64 val = out.E.atom_add_e(idx, E_scaled);

        if_e(val < 0, [&]{
            print("Error in particle dynamics\n");
        });

        for(int i=0; i < 3; i++)
            out.Si_raised[i].atom_add_e(idx, Si_scaled[i]);

        for(int i=0; i < 6; i++)
            out.Sij_raised[i].atom_add_e(idx, Sij_scaled[i]);
    });
}

void fixed_to_float(execution_context& ectx, buffer<valuei64> in, buffer_mut<valuef> out, literal<valued> fixed_scale, literal<valuei> count)
{
    using namespace single_source;

    valuei id = value_impl::get_global_id(0);
    pin(id);

    if_e(id >= count.get(), [&]{
        return_e();
    });

    valued as_double = ((valued)in[id]) / fixed_scale.get();

    as_ref(out[id]) = (valuef)as_double;
}

//https://arxiv.org/pdf/0904.4184.pdf 1.4.18
v4f get_timelike_vector(v3f speed, tetrad tet)
{
    valuef v2 = dot(speed, speed);

    valuef Y = 1 / sqrt(1 - v2);

    v4f bT = Y * tet.v[0];
    v4f bX = Y * speed.x() * tet.v[1];
    v4f bY = Y * speed.y() * tet.v[2];
    v4f bZ = Y * speed.z() * tet.v[3];

    return bT + bX + bY + bZ;
}

struct evolve_vars
{
    valuef gA;
    v3f gB;
    valuef W;

    //interpolation no longer guarantees unit determinant
    metric<valuef, 3, 3> cY;
    tensor<valuef, 3, 3> cA;
    valuef K;

    v3f dgA;
    v3f dW;
    tensor<valuef, 3, 3> dgB;
    tensor<valuef, 3, 3, 3> dcY;

    evolve_vars(bssn_args_mem<buffer<valuef>> in, v3f fpos, v3i dim, valuef scale)
    {
        using namespace single_source;
        pin(fpos);

        fpos = clamp(fpos, (v3f){3,3,3}, (v3f)dim - 4.);
        pin(fpos);

        auto gA_at = [&](v3i pos)
        {
            bssn_args args(pos, dim, in);
            pin(args.gA);
            return args.gA;
        };

        auto gB_at = [&](v3i pos)
        {
            bssn_args args(pos, dim, in);
            pin(args.gA);
            return args.gB;
        };

        auto W_at = [&](v3i pos)
        {
            bssn_args args(pos, dim, in);
            pin(args.W);
            return args.W;
        };

        auto cY_at = [&](v3i pos)
        {
            bssn_args args(pos, dim, in);
            pin(args.cY);

            //print("cY %.23f %.23f %.23f %.23f %.23f %.23f pos %i %i %i\n", args.cY[0, 0], args.cY[1, 1], args.cY[2, 2], args.cY[0, 1], args.cY[0, 2], args.cY[1, 2], pos.x(), pos.y(), pos.z());

            return args.cY;
        };

        auto K_at = [&](v3i pos)
        {
            bssn_args args(pos, dim, in);
            pin(args.K);
            return args.K;
        };

        auto cA_at = [&](v3i pos)
        {
            bssn_args args(pos, dim, in);
            pin(args.cA);
            return args.cA;
        };

        auto dgA_at = [&](v3i pos)
        {
            derivative_data d;
            d.pos = pos;
            d.dim = dim;
            d.scale = scale;

            bssn_args args(pos, dim, in, true);

            v3f dgA = (v3f){diff1(args.gA, 0, d), diff1(args.gA, 1, d), diff1(args.gA, 2, d)};
            pin(dgA);

            //print("dgA %.23f %.23f %.23f pos %i %i %i\n", dgA[0], dgA[1], dgA[2], pos.x(), pos.y(), pos.z());

            return dgA;
        };

        auto dW_at = [&](v3i pos)
        {
            derivative_data d;
            d.pos = pos;
            d.dim = dim;
            d.scale = scale;

            bssn_args args(pos, dim, in, true);

            v3f dW = (v3f){diff1(args.W, 0, d), diff1(args.W, 1, d), diff1(args.W, 2, d)};
            pin(dW);

            return dW;
        };

        auto dgB_at = [&](v3i pos)
        {
            derivative_data d;
            d.pos = pos;
            d.dim = dim;
            d.scale = scale;

            bssn_args args(pos, dim, in);
            tensor<valuef, 3, 3> dgB;

            for(int i=0; i < 3; i++)
                for(int j=0; j < 3; j++)
                    dgB[i, j] = diff1(args.gB[j], i, d);

            pin(dgB);
            return dgB;
        };

        auto dcY_at = [&](v3i pos)
        {
            derivative_data d;
            d.pos = pos;
            d.dim = dim;
            d.scale = scale;

            bssn_args args(pos, dim, in, true);
            tensor<valuef, 3, 3, 3> dcY;

            for(int i=0; i < 3; i++)
                for(int j=0; j < 3; j++)
                    for(int k=0; k < 3; k++)
                        dcY[i, j, k] = diff1(args.cY[j, k], i, d);

            pin(dcY);
            return dcY;
        };


        gA = function_trilinear_particles(gA_at, fpos);
        gB = function_trilinear_particles(gB_at, fpos);
        cY = function_trilinear_particles(cY_at, fpos);
        cA = function_trilinear_particles(cA_at, fpos);
        K = function_trilinear_particles(K_at, fpos);
        W = function_trilinear_particles(W_at, fpos);

        pin(gA);
        pin(gB);
        pin(cY);
        pin(cA);
        pin(K);
        pin(W);

        dgA = function_trilinear_particles(dgA_at, fpos);
        dgB = function_trilinear_particles(dgB_at, fpos);
        dcY = function_trilinear_particles(dcY_at, fpos);
        dW = function_trilinear_particles(dW_at, fpos);

        pin(dgA);
        pin(dgB);
        pin(dcY);
        pin(dW);
    }
};

//screw it. Do the whole tetrad spiel from raytrace_init, I've already done it. Return a tetrad
void calculate_particle_properties(execution_context& ectx, bssn_args_mem<buffer<valuef>> in, std::array<buffer<valuef>, 3> pos_in, std::array<buffer<valuef>, 3> vel_in, buffer<valuef> mass_in, std::array<buffer_mut<valuef>, 3> vel_out, buffer_mut<valuef> lorentz_out, literal<value<size_t>> count, literal<v3i> dim, literal<valuef> scale)
{
    using namespace single_source;

    value<size_t> id = value_impl::get_global_id_us(0);

    if_e(id >= count.get(), [&]{
        return_e();
    });

    v3f world_pos = {pos_in[0][id], pos_in[1][id], pos_in[2][id]};

    v3f cell_pos = world_to_grid(world_pos, dim.get(), scale.get());
    pin(cell_pos);

    evolve_vars vars(in, cell_pos, dim.get(), scale.get());

    auto Yij = vars.cY / pow(max(vars.W, 0.01f), 2);

    pin(Yij);
    pin(vars.gA);
    pin(vars.gB);

    m44f metric = calculate_real_metric(Yij, vars.gA, vars.gB);
    pin(metric);

    tetrad tet = calculate_tetrad(metric, {0,0,0}, false);

    v3f speed_in = {vel_in[0][id], vel_in[1][id], vel_in[2][id]};

    v4f velocity4 = get_timelike_vector(speed_in, tet);

    valuef lorentz = 1 / sqrt(1 - dot(speed_in, speed_in));

    v4f velocity_lo = metric.lower(velocity4);

    as_ref(vel_out[0][id]) = velocity_lo[1];
    as_ref(vel_out[1][id]) = velocity_lo[2];
    as_ref(vel_out[2][id]) = velocity_lo[3];
    as_ref(lorentz_out[id]) = velocity4[0];
}

template<bool FirstStep>
void evolve_particles(execution_context& ctx,
                      bssn_args_mem<buffer<valuef>> base,
                      bssn_args_mem<buffer<valuef>> in,
                      particle_base_args<buffer<valuef>> p_base, particle_base_args<buffer<valuef>> p_in, particle_base_args<buffer_mut<valuef>> p_out,
                      buffer_mut<valuef> lorentz_out,
                      literal<value<size_t>> count,
                      literal<v3i> dim,
                      literal<valuef> scale,
                      literal<valuef> timestep)
{
    using namespace single_source;

    value<size_t> id = value_impl::get_global_id_us(0);
    pin(id);

    if_e(id >= count.get(), [&]{
        return_e();
    });

    v3f pos_base = p_base.get_position(id);
    v3f pos_next = p_in.get_position(id);

    v3f vel_base = p_base.get_velocity(id);
    v3f vel_next = p_in.get_velocity(id);

    v3f grid_next = world_to_grid(pos_next, dim.get(), scale.get());

    valuef mass = p_in.get_mass(id);
    pin(mass);

    if_e(!isfinite(mass) || mass == 0, [&]{
        for(int i=0; i < 3; i++)
            as_ref(p_out.positions[i][id]) = pos_base[i];

        for(int i=0; i < 3; i++)
            as_ref(p_out.velocities[i][id]) = vel_base[i];

        as_ref(lorentz_out[id]) = valuef(1.f);
        as_ref(p_out.masses[id]) = mass;

        return_e();
    });

    v3f vel;
    v3f pos;

    metric<valuef, 3, 3> cY;
    valuef W;
    valuef gA;
    v3f gB;

    v3f dW;
    v3f dgA;
    tensor<valuef, 3, 3> dgB;
    tensor<valuef, 3, 3, 3> dcY;

    if(!FirstStep)
    {
        v3f grid_base = world_to_grid(pos_base, dim.get(), scale.get());

        vel = (vel_base + vel_next) * 0.5f;

        evolve_vars b_evolve(base, grid_base, dim.get(), scale.get());
        evolve_vars i_evolve(in, grid_next, dim.get(), scale.get());

        cY = (b_evolve.cY + i_evolve.cY) * 0.5f;
        W = (b_evolve.W + i_evolve.W) * 0.5f;
        gA = (b_evolve.gA + i_evolve.gA) * 0.5f;
        gB = (b_evolve.gB + i_evolve.gB) * 0.5f;

        dW = (b_evolve.dW + i_evolve.dW) * 0.5f;
        dgA = (b_evolve.dgA + i_evolve.dgA) * 0.5f;
        dgB = (b_evolve.dgB + i_evolve.dgB) * 0.5f;
        dcY = (b_evolve.dcY + i_evolve.dcY) * 0.5f;
    }
    else
    {
        evolve_vars i_evolve(in, grid_next, dim.get(), scale.get());

        vel = vel_next;

        cY = i_evolve.cY;
        W = i_evolve.W;
        gA = i_evolve.gA;
        gB = i_evolve.gB;

        dW = i_evolve.dW;
        dgA = i_evolve.dgA;
        dgB = i_evolve.dgB;
        dcY = i_evolve.dcY;
    }

    auto icY = cY.invert();
    auto iYij = icY * (W*W);

    valuef au0_sq = 1 + iYij.dot(vel, vel);
    valuef u0 = sqrt(au0_sq) / gA;
    pin(u0);

    v3f dX = -gB + iYij.raise(vel) / u0;

    v3f dV;

    {
        v3f p1 = -gA * u0 * dgA;

        v3f p2;

        for(int i=0; i < 3; i++)
        {
            valuef sum = 0;

            for(int j=0; j < 3; j++)
            {
                sum += vel[j] * dgB[i, j];
            }

            p2[i] = sum;
        }

        v3f p3;

        //Y^jk,i = u dvdx + v dudx
        //(W^2 cY^jk),i = W^2 (cY^jk,i) + cY^jk (W^2),i
        //= W^2 (cY^jk,i) + cY^jk 2 W dW[i]

        for(int i=0; i < 3; i++)
        {
            metric<dual<valuef>, 3, 3> d_cYij;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    d_cYij[j, k].real = cY[j, k];
                    d_cYij[j, k].dual = dcY[i, j, k];
                }
            }

            pin(d_cYij);

            auto dicY = d_cYij.invert();
            pin(dicY);

            valuef sum = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    valuef licy = pow(W, 2) * dicY[j, k].dual + cY[j, k] * 2 * W * dW[i];

                    sum += -(vel[j] * vel[k] / (2 * u0)) * licy;
                }
            }

            p3[i] = sum;
        }

        dV = p1 + p2 + p3;
    }

    for(int i=0; i < 3; i++)
        as_ref(p_out.positions[i][id]) = pos_base[i] + timestep.get() * dX[i];

    for(int i=0; i < 3; i++)
        as_ref(p_out.velocities[i][id]) = vel_base[i] + timestep.get() * dV[i];

    as_ref(lorentz_out[id]) = u0;

    valuef sim_width = (valuef)(dim.get().x() - 1) * scale.get();

    as_ref(p_out.masses[id]) = p_in.masses[id];

    valuei dist = distance_to_boundary((v3i)round(grid_next), dim.get());

    if_e(dist <= 10 || gA < 0.15f, [&]{
        as_ref(p_out.masses[id]) = valuef(0.f);
    });
}

void boot_particle_kernels(cl::context ctx)
{
    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(calculate_particle_nonconformal_E, "calculate_particle_nonconformal_E");
    }, {"calculate_particle_nonconformal_E"});

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(fixed_to_float, "fixed_to_float");
    }, {"fixed_to_float"});

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(calculate_particle_properties, "calculate_particle_properties");
    }, {"calculate_particle_properties"});

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(calculate_particle_intermediates, "calculate_particle_intermediates");
    }, {"calculate_particle_intermediates"});

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(evolve_particles<false>, "evolve_particles");
    }, {"evolve_particles"});

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(evolve_particles<true>, "evolve_particles_base");
    }, {"evolve_particles_base"});

    cl::async_build_and_cache(ctx, [&]{
        return value_impl::make_function(sum_E, "sum_E");
    }, {"sum_E"});
}

double get_fixed_scale(double total_mass, int64_t particle_count)
{
    double approx_total_mass = total_mass;
    double fixed_scale = ((double)particle_count / approx_total_mass) * pow(10., 7.);
    return fixed_scale;
}

//so. I need to calculate E, without the conformal factor
//https://arxiv.org/pdf/1611.07906 16
void particle_initial_conditions(cl::context& ctx, cl::command_queue& cqueue, discretised_initial_data& to_fill, particle_data& data, t3i dim, float scale)
{
    cl::buffer intermediate(ctx);
    intermediate.alloc(sizeof(cl_long) * dim.x() * dim.y() * dim.z());
    intermediate.set_to_zero(cqueue);

    ///assume that our total mass is K
    ///to correctly sum it, we want the scale to be.. well, each particle's mass is K/N
    ///and we want.. 3 digits ( = log2(10^3) = 10 bits) of precision? as many as possible?

    double fixed_scale = get_fixed_scale(data.total_mass, data.count);

    {
        cl_ulong count = data.count;

        cl::args args;
        args.push_back(data.positions[0], data.positions[1], data.positions[2]);
        args.push_back(data.velocities[0], data.velocities[1], data.velocities[2]);
        args.push_back(data.masses);
        args.push_back(intermediate);
        args.push_back(dim);
        args.push_back(scale);
        args.push_back(count);
        args.push_back(fixed_scale);

        cqueue.exec("calculate_particle_nonconformal_E", args, {count}, {128});
    }

    {
        int size = dim.x() * dim.y() * dim.z();

        cl::args args;
        args.push_back(intermediate);
        args.push_back(to_fill.particles_contrib);
        args.push_back(fixed_scale);
        args.push_back(size);

        cqueue.exec("fixed_to_float", args, {size}, {128});
    }
}

void dirac_test()
{
    t3f dirac_location = {0, 0, 0.215f};

    int grid_size = 14;
    float world_width = 5;
    float scale = (world_width / (grid_size - 1));

    std::vector<float> values;
    values.resize(grid_size * grid_size * grid_size);

    int centre = (grid_size - 1)/2;

    auto w2g = [&](t3f world)
    {
        return (world / scale) + (t3f){centre, centre, centre};
    };

    auto g2w = [&](t3f grid)
    {
        return (grid - (t3f){centre, centre, centre}) * scale;
    };

    for(int z=0; z < grid_size; z++)
    {
        for(int y=0; y < grid_size; y++)
        {
            for(int x=0; x < grid_size; x++)
            {
                t3i gpos = {x, y, z};
                t3f wpos = g2w((t3f)gpos);

                float d1 = (wpos - dirac_location).length();

                #ifdef WORLD

                float dirac = get_dirac(dirac_delta_f, wpos, dirac_location, 1.f, scale);
                #else
                t3f dirac_grid = w2g(dirac_location);
                float radius_cells = 1/scale;

                float d2 = ((t3f)gpos - dirac_grid).length();

                //printf("Dpos %f %f\n", d1 / 1.f, d2 / radius_cells);

                float dirac = get_dirac(dirac_delta<float>, (t3f)gpos, dirac_grid, radius_cells, scale);

                #endif

                values[z * grid_size * grid_size + y * grid_size + x] = dirac;
            }
        }
    }

    float integrated = 0.f;

    for(auto& i : values)
    {
        integrated += i * scale * scale * scale;
    }

    std::cout << "Integrated " << integrated << std::endl;

    //assert(false);

    #ifdef DIRAC_1D
    float dirac_location = 0.215f;

    int grid_size = 5;
    float world_width = 5;
    float scale = (world_width / (grid_size - 1));

    std::vector<float> values;
    values.resize(grid_size);

    int centre = (grid_size - 1)/2;

    auto w2g = [&](float world)
    {
        return (world / scale) + centre;
    };

    auto g2w = [&](float grid)
    {
        return (grid - centre) * scale;
    };

    for(int i=0; i < values.size(); i++)
    {
        float world = g2w(i);

        float im1 = world - scale / 2.f;
        float ip1 = world + scale / 2.f;

        //float dirac = dirac_delta2(fabs(world - dirac_location));

        /*float dirac = integrate_1d_trapezoidal([&](float in)
        {
            return dirac_delta2(fabs(in - dirac_location));
        }, 10, ip1, im1) / scale;*/

        float dirac = test_dirac1(fabs(world - dirac_location));

        printf("Dirac %f\n", dirac);

        /*float dirac = integrate_1d_trapezoidal([&](float in)
        {
            return test_dirac1(fabs(in - dirac_location));
        }, 100, ip1, im1) / scale;*/

        values[i] = dirac;
    }

    float integrated = 0.f;

    for(auto& i : values)
    {
        integrated += i * scale;
    }

    std::cout << "Integrated " << integrated << std::endl;
    #endif
}

std::vector<buffer_descriptor> particle_buffers::get_description()
{
    buffer_descriptor p0;
    p0.name = "p0";
    p0.sommerfeld_enabled = false;

    buffer_descriptor p1;
    p1.name = "p1";
    p1.sommerfeld_enabled = false;

    buffer_descriptor p2;
    p2.name = "p2";
    p2.sommerfeld_enabled = false;

    buffer_descriptor v0;
    v0.name = "v0";
    v0.sommerfeld_enabled = false;

    buffer_descriptor v1;
    v1.name = "v1";
    v1.sommerfeld_enabled = false;

    buffer_descriptor v2;
    v2.name = "v2";
    v2.sommerfeld_enabled = false;

    buffer_descriptor mass;
    mass.name = "mass";
    mass.sommerfeld_enabled = false;

    return {p0, p1, p2, v0, v1, v2, mass};
}

std::vector<cl::buffer> particle_buffers::get_buffers()
{
    return {positions[0], positions[1], positions[2], velocities[0], velocities[1], velocities[2], masses};
}

void particle_buffers::allocate(cl::context ctx, cl::command_queue cqueue, t3i size)
{
    for(int i=0; i < 3; i++)
    {
        positions[i].alloc(sizeof(cl_float) * particle_count);
        velocities[i].alloc(sizeof(cl_float) * particle_count);
    }

    masses.alloc(sizeof(cl_float) * particle_count);
};

void particle_plugin::add_args_provider(all_adm_args_mem& mem)
{
    mem.add(full_particle_args<buffer<valuef>>());
}

buffer_provider* particle_plugin::get_buffer_factory(cl::context ctx)
{
    return new particle_buffers(ctx, particle_count);
}

buffer_provider* particle_plugin::get_utility_buffer_factory(cl::context ctx)
{
    return new particle_utility_buffers(ctx);
}

particle_plugin::particle_plugin(cl::context ctx, uint64_t _particle_count) : lorentz_storage(ctx), particle_count(_particle_count)
{
    for(int i=0; i < 10; i++)
        particle_temp.emplace_back(ctx);

    boot_particle_kernels(ctx);
}

//consider implementing 3.2 https://arxiv.org/pdf/1905.08890

template struct full_particle_args<buffer<valuef>>;
template struct full_particle_args<buffer_mut<valuef>>;

//going to evaluate the metric at the cell centre
//so: to do this, we need to discretise everything onto a grid, which means: fixed point
//may or may not benefit from downscaling to floats
template<typename T>
valuef full_particle_args<T>::adm_p(bssn_args& args, const derivative_data& d)
{
    return this->E[d.pos, d.dim];
}

template<typename T>
tensor<valuef, 3> full_particle_args<T>::adm_Si(bssn_args& args, const derivative_data& d)
{
    return this->get_Si(d.pos, d.dim);
}

template<typename T>
tensor<valuef, 3, 3> full_particle_args<T>::adm_W2_Sij(bssn_args& args, const derivative_data& d)
{
    return args.W * args.W * this->get_Sij(d.pos, d.dim);
}

void particle_utility_buffers::allocate(cl::context ctx, cl::command_queue cqueue, t3i size)
{
    E.alloc(sizeof(cl_float) * size.x() * size.y() * size.z());
    E.set_to_zero(cqueue);

    for(auto& i : Si_raised)
    {
        i.alloc(sizeof(cl_float) * size.x() * size.y() * size.z());
        i.set_to_zero(cqueue);
    }

    for(auto& i : Sij_raised)
    {
        i.alloc(sizeof(cl_float) * size.x() * size.y() * size.z());
        i.set_to_zero(cqueue);
    }
}

//hmm
std::vector<buffer_descriptor> particle_utility_buffers::get_description()
{
    buffer_descriptor bE;
    bE.name = "E";

    buffer_descriptor bs0, bs1, bs2;
    bs0.name = "s0";
    bs1.name = "s1";
    bs2.name = "s2";

    buffer_descriptor bss0, bss1, bss2, bss3, bss4, bss5;
    bss0.name = "ss0";
    bss1.name = "ss1";
    bss2.name = "ss2";
    bss3.name = "ss3";
    bss4.name = "ss4";
    bss5.name = "ss5";

    return {bE, bs0, bs1, bs2, bss0, bss1, bss2, bss3, bss4, bss5};
}

std::vector<cl::buffer> particle_utility_buffers::get_buffers()
{
    std::vector<cl::buffer> out;

    out.push_back(E);

    for(auto& i : Si_raised)
        out.push_back(i);

    for(auto& i : Sij_raised)
        out.push_back(i);

    return out;
}


void particle_plugin::calculate_intermediates(cl::context ctx, cl::command_queue cqueue, std::vector<cl::buffer> bssn_in, particle_buffers& p_in, particle_utility_buffers& util_out, t3i dim, float scale)
{
    for(auto& i : particle_temp)
        i.set_to_zero(cqueue);

    cl_ulong count = particle_count;

    double fixed_scale = get_fixed_scale(total_mass, count);

    {
        cl::args args;

        for(auto& i : bssn_in)
            args.push_back(i);

        args.push_back(p_in.positions[0], p_in.positions[1], p_in.positions[2]);
        args.push_back(p_in.velocities[0], p_in.velocities[1], p_in.velocities[2]);
        args.push_back(p_in.masses);
        args.push_back(lorentz_storage);

        for(auto& i : particle_temp)
            args.push_back(i);

        args.push_back(dim);
        args.push_back(scale);
        args.push_back(count);
        args.push_back(fixed_scale);

        cqueue.exec("calculate_particle_intermediates", args, {count}, {128});
    }

    //void fixed_to_float(execution_context& ectx, buffer<valuei64> in, buffer<valuef> out, literal<valued> fixed_scale, literal<valuei> count)
    {
        auto fix = [&](cl::buffer b1, cl::buffer b2)
        {
            int linear_cnt = dim.x() * dim.y() * dim.z();

            cl::args args;
            args.push_back(b1);
            args.push_back(b2);
            args.push_back(fixed_scale);
            args.push_back(linear_cnt);

            cqueue.exec("fixed_to_float", args, {linear_cnt}, {128});
        };

        fix(particle_temp[0], util_out.E);
        fix(particle_temp[1], util_out.Si_raised[0]);
        fix(particle_temp[2], util_out.Si_raised[1]);
        fix(particle_temp[3], util_out.Si_raised[2]);
        fix(particle_temp[4], util_out.Sij_raised[0]);
        fix(particle_temp[5], util_out.Sij_raised[1]);
        fix(particle_temp[6], util_out.Sij_raised[2]);
        fix(particle_temp[7], util_out.Sij_raised[3]);
        fix(particle_temp[8], util_out.Sij_raised[4]);
        fix(particle_temp[9], util_out.Sij_raised[5]);
    }
}

void particle_plugin::init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, cl::buffer u, buffer_provider* to_init, buffer_provider* to_init_utility)
{
    assert(pack.gpu_particles);
    total_mass = pack.gpu_particles->total_mass;

    lorentz_storage.alloc(sizeof(cl_float) * particle_count);
    lorentz_storage.set_to_zero(cqueue);

    for(auto& i : particle_temp)
    {
        i.alloc(sizeof(cl_ulong) * int64_t{pack.dim.x()} * pack.dim.y() * pack.dim.z());
        i.set_to_zero(cqueue);
    }

    particle_buffers& p_out = *dynamic_cast<particle_buffers*>(to_init);
    particle_utility_buffers& util_out = *dynamic_cast<particle_utility_buffers*>(to_init_utility);

    particle_data& p_in = pack.gpu_particles.value();
    cl_ulong count = particle_count;

    {

        cl::args args;
        in.append_to(args);
        args.push_back(p_in.positions[0], p_in.positions[1], p_in.positions[2]);
        args.push_back(p_in.velocities[0], p_in.velocities[1], p_in.velocities[2]);
        args.push_back(p_in.masses);
        args.push_back(p_out.velocities[0], p_out.velocities[1], p_out.velocities[2]);
        args.push_back(lorentz_storage);
        args.push_back(count);
        args.push_back(pack.dim);
        args.push_back(pack.scale);

        cqueue.exec("calculate_particle_properties", args, {count}, {128});
    }

    cl::copy(cqueue, p_in.positions[0], p_out.positions[0]);
    cl::copy(cqueue, p_in.positions[1], p_out.positions[1]);
    cl::copy(cqueue, p_in.positions[2], p_out.positions[2]);
    cl::copy(cqueue, p_in.masses, p_out.masses);

    std::vector<cl::buffer> bssn;

    in.for_each([&](cl::buffer buf)
    {
        bssn.push_back(buf);
    });

    calculate_intermediates(ctx, cqueue, bssn, p_out, util_out, pack.dim, pack.scale);
}

void particle_plugin::step(cl::context ctx, cl::command_queue cqueue, const plugin_step_data& sdata)
{
    /*void evolve_particles(execution_context& ctx,
                      bssn_args_mem<buffer<valuef>> base,
                      bssn_args_mem<buffer<valuef>> in,
                      particle_base_args<buffer<valuef>> p_base, particle_base_args<buffer<valuef>> p_in, particle_base_args<buffer_mut<valuef>> p_out,
                      literal<value<size_t>> count,
                      literal<v3i> dim,
                      literal<valuef> scale,
                      literal<valuef> timestep)*/

    particle_buffers& base = *dynamic_cast<particle_buffers*>(sdata.buffers[sdata.base_idx]);
    particle_buffers& in = *dynamic_cast<particle_buffers*>(sdata.buffers[sdata.in_idx]);
    particle_buffers& out = *dynamic_cast<particle_buffers*>(sdata.buffers[sdata.out_idx]);

    cl_ulong count = particle_count;

    {
        cl::args args;

        for(auto& i : sdata.base_bssn_buffers)
            args.push_back(i);

        for(auto& i : sdata.bssn_buffers)
            args.push_back(i);

        auto base_bufs = base.get_buffers();
        auto in_bufs = in.get_buffers();
        auto out_bufs = out.get_buffers();

        for(auto& i : base_bufs)
            args.push_back(i);
        for(auto& i : in_bufs)
            args.push_back(i);
        for(auto& i : out_bufs)
            args.push_back(i);

        args.push_back(lorentz_storage);

        args.push_back(count);
        args.push_back(sdata.dim);
        args.push_back(sdata.scale);
        args.push_back(sdata.timestep);

        if(sdata.in_idx == sdata.base_idx)
            cqueue.exec("evolve_particles_base", args, {count}, {128});
        else
            cqueue.exec("evolve_particles", args, {count}, {128});
    }

    particle_utility_buffers& util_out = *dynamic_cast<particle_utility_buffers*>(sdata.utility_buffers);

    calculate_intermediates(ctx, cqueue, sdata.bssn_buffers, in, util_out, sdata.dim, sdata.scale);

    //#define CHECK_E
    #ifdef CHECK_E
    {
        cl::buffer buf(ctx);
        buf.alloc(sizeof(cl_long));
        buf.set_to_zero(cqueue);

        cl_int len = sdata.dim.x() * sdata.dim.y() * sdata.dim.z();

        cl::args args;
        args.push_back(sdata.dim);
        args.push_back(util_out.E);
        args.push_back(len);
        args.push_back(sdata.scale);
        args.push_back(buf);

        cqueue.exec("sum_E", args, {len}, {128});

        cl_long found = buf.read<cl_long>(cqueue).at(0);

        std::cout << "TOTAL E " << (double)found / pow(10., 12.) << std::endl;
    }

    #endif // CHECK_E
}

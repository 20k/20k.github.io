#include "particles.hpp"
#include "integration.hpp"
#include "bssn.hpp"
#include "init_general.hpp"
#include "formalisms.hpp"
#include "raytrace_init.hpp"
#include "interpolation.hpp"

///https://arxiv.org/pdf/1611.07906.pdf (20)
//3d
valuef dirac_delta_v(const valuef& r, const valuef& radius)
{
    valuef frac = r / radius;

    valuef mult = 1/(M_PI * pow(radius, 3.f));

    valuef result = 0;

    valuef branch_1 = (1.f/4.f) * pow(2.f - frac, 3.f);
    valuef branch_2 = 1.f - (3.f/2.f) * pow(frac, 2.f) + (3.f/4.f) * pow(frac, 3.f);

    result = ternary(frac <= 2, mult * branch_1, 0.f);
    result = ternary(frac <= 1, mult * branch_2, result);

    return result;
}

//3d
float dirac_delta_f(const float& r, const float& radius)
{
    float frac = r / radius;

    float mult = 1/(M_PI * pow(radius, 3.f));

    float result = 0;

    float branch_1 = (1.f/4.f) * pow(2.f - frac, 3.f);
    float branch_2 = 1.f - (3.f/2.f) * pow(frac, 2.f) + (3.f/4.f) * pow(frac, 3.f);

    if(frac <= 1)
        return mult * branch_2;
    if(frac <= 2)
        return mult * branch_1;

    return 0.f;
}

//1d
float dirac_delta_1d(const float& r)
{
    if(r >= 1)
        return 0.f;

    return 1 - r;
}

template<typename T>
inline
T get_dirac(auto&& func, tensor<T, 3> world_pos, tensor<T, 3> dirac_location, T radius, T scale)
{
    T r = (world_pos - dirac_location).length();

    #ifdef GET_DIRAC_STANDARD
    return func(r, radius);
    #endif // GET_DIRAC_STANDARD

    #define GET_DIRAC_CORRECTED
    #ifdef GET_DIRAC_CORRECTED
    tensor<T, 3> scale3 = {scale, scale, scale};

    auto im1 = world_pos - scale3 / 2;
    auto ip1 = world_pos + scale3 / 2;

    return integrate_3d_trapezoidal([&](T x, T y, T z)
    {
        tensor<T, 3> pos = {x, y, z};

        return func((pos - dirac_location).length(), radius);
    }, 10, ip1, im1) / (scale*scale*scale);
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

//det(cA) = c^n det(A). For us, c^3
//Yij = W^2 cY
//det(Yij) = det(W^2 cY)
//det(Yij) = W^6 det(cY)
//sqrt(det(Yij)) = W^6^0.5
//= W^3

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

    int radius_cells = 3;
    valuef radius_world = radius_cells * scale.get();

    //valuef dirac_prefix = 1/(M_PI * pow(radius_world, 3.f));

    valuef lorentz = 1;
    valuef mass = particles_in.get_mass(id);
    v3f pos = particles_in.get_position(id);
    //v3f vel = particles_in.get_velocity(id);

    pin(mass);
    pin(pos);
    //pin(vel);

    v3i cell = (v3i)floor(world_to_grid(pos, dim.get(), scale.get()));
    pin(cell);

    int spread = radius_cells + 1;

    for(int z = -spread; z <= spread; z++)
    {
        for(int y = -spread; y <= spread; y++)
        {
            for(int x = -spread; x <= spread; x++)
            {
                v3i offset = {x, y, z};
                offset += cell;

                offset = clamp(offset, (v3i){0,0,0}, dim.get() - 1);
                pin(offset);

                v3f world_pos = grid_to_world((v3f)offset, dim.get(), scale.get());
                pin(world_pos);

                valuef dirac = dirac_delta_v((world_pos - pos).length(), radius_world);
                pin(dirac);

                if_e(dirac > 0, [&]{
                    valuef fin_E = mass * lorentz * dirac;

                    auto scale = [&](valuef in)
                    {
                        valued in_d = (valued)in;
                        valued in_scaled = in_d * fixed_scale.get();

                        return (valuei64)in_scaled;
                    };

                    valuei64 as_i64 = scale(fin_E);

                    valuei idx = offset.z() * dim.get().y() * dim.get().x() + offset.y() * dim.get().x() + offset.x();

                    nonconformal_E_out.atom_add_e(idx, as_i64);
                });
            }
        }
    }
}

void calculate_particle_intermediates(execution_context& ectx,
                                      bssn_args_mem<buffer<valuef>> in,
                                      particle_base_args<buffer<valuef>> particles_in,
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

    int radius_cells = 3;
    valuef radius_world = radius_cells * scale.get();

    valuef lorentz = particles_in.get_lorentz(id) + 1;
    valuef mass = particles_in.get_mass(id);
    v3f pos = particles_in.get_position(id);
    v3f vel = particles_in.get_velocity(id);

    pin(pos);
    pin(vel);
    pin(mass);
    pin(lorentz);

    v3i cell = (v3i)floor(world_to_grid(pos, dim.get(), scale.get()));
    pin(cell);

    int spread = radius_cells + 1;

    for(int z = -spread; z <= spread; z++)
    {
        for(int y = -spread; y <= spread; y++)
        {
            for(int x = -spread; x <= spread; x++)
            {
                v3i offset = {x, y, z};
                offset += cell;

                offset = clamp(offset, (v3i){0,0,0}, dim.get() - 1);
                pin(offset);

                bssn_args args(offset, dim.get(), in);

                v3f world_pos = grid_to_world((v3f)offset, dim.get(), scale.get());
                pin(world_pos);

                valuef dirac = dirac_delta_v((world_pos - pos).length(), radius_world);
                pin(dirac);

                if_e(dirac > 0, [&] {
                    valuef sqrt_det_Gamma = pow(max(args.W, 0.1f), 3);
                    pin(sqrt_det_Gamma);

                    valuef fin_E = mass * lorentz * dirac / sqrt_det_Gamma;
                    v3f Si_raised = (mass * lorentz * dirac / sqrt_det_Gamma) * vel;

                    tensor<valuef, 3, 3> Sij_raised;

                    for(int i=0; i < 3; i++)
                    {
                        for(int j=0; j < 3; j++)
                        {
                            Sij_raised[i, j] = (mass * lorentz * dirac / sqrt_det_Gamma) * vel[i] * vel[j];
                        }
                    }

                    std::array<valuef, 6> Sij_sym = extract_symmetry(Sij_raised);

                    auto scale = [&](valuef in)
                    {
                        valued in_d = (valued)in;
                        valued in_scaled = in_d * fixed_scale.get();

                        return (valuei64)in_scaled;
                    };

                    valuei64 E_scaled = scale(fin_E);

                    tensor<valuei64, 3> Si_scaled;

                    for(int i=0; i < 3; i++)
                        Si_scaled[i] = scale(Si_raised[i]);

                    std::array<valuei64, 6> Sij_scaled;

                    for(int i=0; i < 6; i++)
                        Sij_scaled[i] = scale(Sij_sym[i]);

                    ///[offset, dim.get()]

                    valuei idx = offset.z() * dim.get().y() * dim.get().x() + offset.y() * dim.get().x() + offset.x();

                    out.E.atom_add_e(idx, E_scaled);

                    for(int i=0; i < 3; i++)
                        out.Si_raised[i].atom_add_e(idx, Si_scaled[i]);

                    for(int i=0; i < 6; i++)
                        out.Sij_raised[i].atom_add_e(idx, Sij_scaled[i]);
                });
            }
        }
    }
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

//screw it. Do the whole tetrad spiel from raytrace_init, I've already done it. Return a tetrad
void calculate_particle_properties(execution_context& ectx, bssn_args_mem<buffer<valuef>> in, std::array<buffer<valuef>, 3> pos_in, std::array<buffer<valuef>, 3> vel_in, std::array<buffer_mut<valuef>, 3> vel_out, buffer_mut<valuef> lorentz_out, literal<value<size_t>> count, literal<v3i> dim, literal<valuef> scale)
{
    using namespace single_source;

    value<size_t> id = value_impl::get_global_id_us(0);

    if_e(id >= count.get(), [&]{
        return_e();
    });

    v3f world_pos = {pos_in[0][id], pos_in[1][id], pos_in[2][id]};

    v3f cell_pos = world_to_grid(world_pos, dim.get(), scale.get());

    adm_variables vars = admf_at(cell_pos, dim.get(), in);

    m44f metric = calculate_real_metric(vars.Yij, vars.gA, vars.gB);

    tetrad tet = calculate_tetrad(metric, {0,0,0}, false);

    /*print("Tet %f %f %f %f b %f %f %f %f c %f %f %f %f d %f %f %f %f\n",
          tet.v[0][0], tet.v[0][1], tet.v[0][2], tet.v[0][3],
          tet.v[1][0], tet.v[1][1], tet.v[1][2], tet.v[1][3],
          tet.v[2][0], tet.v[2][1], tet.v[2][2], tet.v[2][3],
          tet.v[3][0], tet.v[3][1], tet.v[3][2], tet.v[3][3]
          );*/

    v3f speed_in = {vel_in[0][id], vel_in[1][id], vel_in[2][id]};

    v4f velocity4 = get_timelike_vector(speed_in, tet);

    valuef lorentz = 1 / sqrt(1 - dot(speed_in, speed_in));

    v4f projected = (velocity4 / lorentz) - get_adm_hypersurface_normal_raised(vars.gA, vars.gB);

    as_ref(vel_out[0][id]) = projected[1];
    as_ref(vel_out[1][id]) = projected[2];
    as_ref(vel_out[2][id]) = projected[3];
    //always store the lorentz factor as lorentz - 1
    as_ref(lorentz_out[id]) = lorentz - 1;
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

        auto gA_at = [&](v3i pos)
        {
            pos = clamp(pos, (v3i){1,1,1}, dim - 2);

            bssn_args args(pos, dim, in);
            //pin(args.gA);
            return args.gA;
        };

        auto gB_at = [&](v3i pos)
        {
            pos = clamp(pos, (v3i){1,1,1}, dim - 2);

            bssn_args args(pos, dim, in);
            //pin(args.gA);
            return args.gB;
        };

        auto W_at = [&](v3i pos)
        {
            pos = clamp(pos, (v3i){1,1,1}, dim - 2);

            bssn_args args(pos, dim, in);
            //pin(args.W);
            return args.W;
        };

        auto cY_at = [&](v3i pos)
        {
            pos = clamp(pos, (v3i){1,1,1}, dim - 2);

            bssn_args args(pos, dim, in);
            //pin(args.cY);
            return args.cY;
        };

        auto K_at = [&](v3i pos)
        {
            pos = clamp(pos, (v3i){1,1,1}, dim - 2);

            bssn_args args(pos, dim, in);
            return args.K;
        };

        auto cA_at = [&](v3i pos)
        {
            pos = clamp(pos, (v3i){1,1,1}, dim - 2);

            bssn_args args(pos, dim, in);
            return args.cA;
        };

        auto dgA_at = [&](v3i pos)
        {
            pos = clamp(pos, (v3i){1,1,1}, dim - 2);

            derivative_data d;
            d.pos = pos;
            d.dim = dim;
            d.scale = scale;

            bssn_args args(pos, dim, in);

            v3f dgA = (v3f){diff1(args.gA, 0, d), diff1(args.gA, 1, d), diff1(args.gA, 2, d)};
            //pin(dgA);

            return dgA;
        };

        auto dW_at = [&](v3i pos)
        {
            pos = clamp(pos, (v3i){1,1,1}, dim - 2);

            derivative_data d;
            d.pos = pos;
            d.dim = dim;
            d.scale = scale;

            bssn_args args(pos, dim, in);

            v3f dW = (v3f){diff1(args.W, 0, d), diff1(args.W, 1, d), diff1(args.W, 2, d)};
            //pin(dW);

            return dW;
        };

        auto dgB_at = [&](v3i pos)
        {
            pos = clamp(pos, (v3i){1,1,1}, dim - 2);

            derivative_data d;
            d.pos = pos;
            d.dim = dim;
            d.scale = scale;

            bssn_args args(pos, dim, in);
            tensor<valuef, 3, 3> dgB;

            for(int i=0; i < 3; i++)
                for(int j=0; j < 3; j++)
                    dgB[i, j] = diff1(args.gB[j], i, d);

            //pin(dgB);
            return dgB;
        };

        auto dcY_at = [&](v3i pos)
        {
            pos = clamp(pos, (v3i){1,1,1}, dim - 2);

            derivative_data d;
            d.pos = pos;
            d.dim = dim;
            d.scale = scale;

            bssn_args args(pos, dim, in);
            tensor<valuef, 3, 3, 3> dcY;

            for(int i=0; i < 3; i++)
                for(int j=0; j < 3; j++)
                    for(int k=0; k < 3; k++)
                        dcY[i, j, k] = diff1(args.cY[j, k], i, d);

            //pin(dcY);
            return dcY;
        };


        gA = function_trilinear(gA_at, fpos);
        gB = function_trilinear(gB_at, fpos);
        cY = function_trilinear(cY_at, fpos);
        cA = function_trilinear(cA_at, fpos);
        K = function_trilinear(K_at, fpos);
        W = function_trilinear(W_at, fpos);

        pin(gA);
        pin(gB);
        pin(cY);
        pin(cA);
        pin(K);
        pin(W);

        dgA = function_trilinear(dgA_at, fpos);
        dgB = function_trilinear(dgB_at, fpos);
        dcY = function_trilinear(dcY_at, fpos);
        dW = function_trilinear(dW_at, fpos);

        pin(dgA);
        pin(dgB);
        pin(dcY);
        pin(dW);
    }
};

void evolve_particles(execution_context& ctx,
                      bssn_args_mem<buffer<valuef>> base,
                      bssn_args_mem<buffer<valuef>> in,
                      particle_base_args<buffer<valuef>> p_base, particle_base_args<buffer<valuef>> p_in, particle_base_args<buffer_mut<valuef>> p_out,
                      literal<value<size_t>> count,
                      literal<v3i> dim,
                      literal<valuef> scale,
                      literal<valuef> timestep)
{
    using namespace single_source;

    value<size_t> id = value_impl::get_global_id_us(0);

    if_e(id >= count.get(), [&]{
        return_e();
    });

    v3f pos_base = p_base.get_position(id);
    v3f pos_next = p_in.get_position(id);
    v3f vel_base = p_base.get_velocity(id);
    v3f vel_next = p_in.get_velocity(id);
    valuef lorentz_base = p_base.get_lorentz(id);
    valuef lorentz_next = p_in.get_lorentz(id);

    valuef lorentz = (lorentz_base + lorentz_next) * 0.5f + 1;
    v3f vel = (vel_base + vel_next) * 0.5f;

    evolve_vars b_evolve(base, pos_base, dim.get(), scale.get());
    evolve_vars i_evolve(in, pos_next, dim.get(), scale.get());

    auto cY = (b_evolve.cY + i_evolve.cY) * 0.5f;
    auto W = (b_evolve.W + i_evolve.W) * 0.5f;
    auto cA = (b_evolve.cA + i_evolve.cA) * 0.5f;
    auto gA = (b_evolve.gA + i_evolve.gA) * 0.5f;
    auto gB = (b_evolve.gB + i_evolve.gB) * 0.5f;
    auto K = (b_evolve.K + i_evolve.K) * 0.5f;

    auto dW = (b_evolve.dW + i_evolve.dW) * 0.5f;
    auto dgA = (b_evolve.dgA + i_evolve.dgA) * 0.5f;
    auto dgB = (b_evolve.dgB + i_evolve.dgB) * 0.5f;
    auto dcY = (b_evolve.dcY + i_evolve.dcY) * 0.5f;

    auto icY = cY.invert();

    auto iYij = icY * (W*W);
    tensor<valuef, 3, 3> Kij = (cA + cY.to_tensor() * (K/3.f)) / pow(max(W, 0.01f), 2.f);
    pin(Kij);
    pin(iYij);

    auto christoff2_cfl = christoffel_symbols_2(icY, dcY);
    pin(christoff2_cfl);

    auto christoff2 = get_full_christoffel2(W, dW, cY, icY, christoff2_cfl);
    pin(christoff2);

    v3f dX = gA * vel - gB;

    v3f dV;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            valuef kjvk = 0;

            for(int k=0; k < 3; k++)
            {
                kjvk += Kij[j, k] * vel[k];
            }

            valuef christoffel_sum = 0;

            for(int k=0; k < 3; k++)
            {
                christoffel_sum += christoff2[i, j, k] * vel[k];
            }

            valuef dlog_gA = dgA[j] / max(gA, 0.01f);

            dV[i] += gA * vel[j] * (vel[i] * (dlog_gA - kjvk) + 2 * iYij.raise(Kij, 0)[i, j] - christoffel_sum)
                    - iYij[i, j] * dgA[j] - vel[j] * dgB[j, i];
        }
    }

    valuef dlorentz = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            dlorentz += lorentz * vel[i] * (gA * Kij[i, j] * vel[j] - dgA[i]);
        }
    }

    for(int i=0; i < 3; i++)
        as_ref(p_out.positions[i][id]) = pos_base[i] + timestep.get() * dX[i];

    for(int i=0; i < 3; i++)
        as_ref(p_out.velocities[i][id]) = vel_base[i] + timestep.get() * dV[i];

    as_ref(p_out.lorentzs[id]) = lorentz_base + timestep.get() * dlorentz;
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
        return value_impl::make_function(evolve_particles, "evolve_particles");
    }, {"evolve_particles"});
}

double get_fixed_scale(int64_t particle_count)
{
    double approx_total_mass = 1;
    double fixed_scale = ((double)particle_count / approx_total_mass) * pow(10., 5.);
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

    double fixed_scale = get_fixed_scale(data.count);

    {
        cl_ulong count = data.count;

        cl::args args;
        args.push_back(data.positions[0], data.positions[1], data.positions[2]);
        args.push_back(data.velocities[0], data.velocities[1], data.velocities[2]);
        args.push_back(data.masses);
        args.push_back(nullptr);
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

    int grid_size = 5;
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

                float dirac = get_dirac(dirac_delta_f, wpos, dirac_location, 1.f, scale);

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

    buffer_descriptor lorentz;
    lorentz.name = "lorentz";
    lorentz.sommerfeld_enabled = false;

    return {p0, p1, p2, v0, v1, v2, mass, lorentz};
}

std::vector<cl::buffer> particle_buffers::get_buffers()
{
    return {positions[0], positions[1], positions[2], velocities[0], velocities[1], velocities[2], masses, lorentzs};
}

void particle_buffers::allocate(cl::context ctx, cl::command_queue cqueue, t3i size)
{
    for(int i=0; i < 3; i++)
    {
        positions[i].alloc(sizeof(cl_float) * particle_count);
        velocities[i].alloc(sizeof(cl_float) * particle_count);
    }

    masses.alloc(sizeof(cl_float) * particle_count);
    lorentzs.alloc(sizeof(cl_float) * particle_count);
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

particle_plugin::particle_plugin(cl::context ctx, uint64_t _particle_count) : particle_count(_particle_count)
{
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
    //todo: fixme
    auto Yij = args.cY / pow(max(args.W, 0.1f), 2.f);

    v3f Ji = this->get_Si(d.pos, d.dim);

    return Yij.lower(Ji);
}

template<typename T>
tensor<valuef, 3, 3> full_particle_args<T>::adm_W2_Sij(bssn_args& args, const derivative_data& d)
{
    auto Yij = args.cY / pow(max(args.W, 0.1f), 2.f);

    tensor<valuef, 3, 3> Sij = this->get_Sij(d.pos, d.dim);

    return args.W * args.W * Yij.lower(Yij.lower(Sij, 0), 1);
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

struct particle_temp
{
    std::vector<cl::buffer> bufs;

    particle_temp(cl::context ctx, cl::command_queue cqueue, t3i size)
    {
        for(int i=0; i < 10; i++)
        {
            bufs.emplace_back(ctx).alloc(sizeof(cl_ulong) * size.x() * size.y() * size.z());
            bufs.back().set_to_zero(cqueue);
        }
    }
};

void calculate_intermediates(cl::context ctx, cl::command_queue cqueue, std::vector<cl::buffer> bssn_in, particle_buffers& p_in, particle_utility_buffers& util_out, t3i dim, float scale, cl_ulong count)
{
    particle_temp tmp(ctx, cqueue, dim);

    double fixed_scale = get_fixed_scale(count);

    {
        cl::args args;

        for(auto& i : bssn_in)
            args.push_back(i);

        args.push_back(p_in.positions[0], p_in.positions[1], p_in.positions[2]);
        args.push_back(p_in.velocities[0], p_in.velocities[1], p_in.velocities[2]);
        args.push_back(p_in.masses);
        args.push_back(p_in.lorentzs);

        for(auto& i : tmp.bufs)
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

        fix(tmp.bufs[0], util_out.E);
        fix(tmp.bufs[1], util_out.Si_raised[0]);
        fix(tmp.bufs[2], util_out.Si_raised[1]);
        fix(tmp.bufs[3], util_out.Si_raised[2]);
        fix(tmp.bufs[4], util_out.Sij_raised[0]);
        fix(tmp.bufs[5], util_out.Sij_raised[1]);
        fix(tmp.bufs[6], util_out.Sij_raised[2]);
        fix(tmp.bufs[7], util_out.Sij_raised[3]);
        fix(tmp.bufs[8], util_out.Sij_raised[4]);
        fix(tmp.bufs[9], util_out.Sij_raised[5]);
    }
}

void particle_plugin::init(cl::context ctx, cl::command_queue cqueue, bssn_buffer_pack& in, initial_pack& pack, cl::buffer u, buffer_provider* to_init, buffer_provider* to_init_utility)
{
    assert(pack.gpu_particles);

    particle_buffers& p_out = *dynamic_cast<particle_buffers*>(to_init);
    particle_utility_buffers& util_out = *dynamic_cast<particle_utility_buffers*>(to_init_utility);

    particle_data& p_in = pack.gpu_particles.value();
    cl_ulong count = particle_count;

    {

        cl::args args;
        in.append_to(args);
        args.push_back(p_in.positions[0], p_in.positions[1], p_in.positions[2]);
        args.push_back(p_in.velocities[0], p_in.velocities[1], p_in.velocities[2]);
        args.push_back(p_out.velocities[0], p_out.velocities[1], p_out.velocities[2]);
        args.push_back(p_out.lorentzs);
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

    calculate_intermediates(ctx, cqueue, bssn, p_out, util_out, pack.dim, pack.scale, count);
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

        args.push_back(count);
        args.push_back(sdata.dim);
        args.push_back(sdata.scale);
        args.push_back(sdata.timestep);

        cqueue.exec("evolve_particles", args, {count}, {128});
    }

    particle_utility_buffers& util_out = *dynamic_cast<particle_utility_buffers*>(sdata.utility_buffers);

    calculate_intermediates(ctx, cqueue, sdata.bssn_buffers, in, util_out, sdata.dim, sdata.scale, count);
}

#include "init_general.hpp"

void discretised_initial_data::init(cl::command_queue& cqueue, t3i dim)
{
    int64_t cells = int64_t{dim.x()} * dim.y() * dim.z();

    mu_h_cfl.alloc(sizeof(cl_float) * cells);
    cfl.alloc(sizeof(cl_float) * cells);
    star_indices.alloc(sizeof(cl_int) * cells);

    mu_h_cfl.set_to_zero(cqueue);
    cfl.fill(cqueue, cl_float{1});
    star_indices.fill(cqueue, cl_int{-1});

    for(auto& i : AIJ_cfl)
    {
        i.alloc(sizeof(cl_float) * cells);
        i.set_to_zero(cqueue);
    }

    for(auto& i : Si_cfl)
    {
        i.alloc(sizeof(cl_float) * cells);
        i.set_to_zero(cqueue);
    }

    for(auto& i : col)
    {
        i.alloc(sizeof(cl_float) * cells);
        i.set_to_zero(cqueue);
    }

    particles_contrib.alloc(sizeof(cl_float) * cells);
    particles_contrib.set_to_zero(cqueue);
}

struct all_laplace_args : value_impl::single_source::argument_pack
{
    buffer<valuef> cfl;
    buffer<valuef> aij_aIJ;
    buffer<valuef> mu_h_cfl;
    buffer<valuef> particles_contrib;

    void build(auto& in)
    {
        using namespace value_impl::builder;

        add(cfl, in);
        add(aij_aIJ, in);
        add(mu_h_cfl, in);
        add(particles_contrib, in);
    }
};

//https://arxiv.org/pdf/1905.08890 2.11 particle initial conditions, 2.28 and 2.26
//for the last term, I convert a 1/sqrt(det(Y)) -> phi^-6
laplace_solver get_laplace_solver_impl(cl::context ctx)
{
    laplace_solver laplace;

    laplace.boot(ctx, [](laplace_params params, all_laplace_args args)
    {
        v3i pos = params.pos;
        v3i dim = params.dim;
        auto cfl = args.cfl[pos, dim] + params.u[pos, dim];
        auto mu_h = args.mu_h_cfl[pos, dim];
        auto particles = args.particles_contrib[pos, dim];
        //for some reason, the hamiltonian constraint isn't really working

        /*single_source::if_e(pos.x() == (dim.x() - 1)/2 && pos.y() == (dim.y() - 1)/2 && pos.z() == ((dim.z() - 1)/2), [&]{
            single_source::print("Part %f\n", -2 * M_PI * pow(cfl, -6) * particles);
        });*/

        return -(1.f/8.f) * pow(cfl, -7) * args.aij_aIJ[pos, dim] - 2 * M_PI * pow(cfl, -3) * mu_h - 2 * M_PI * pow(cfl, -1) * particles;
    }, all_laplace_args(), "laplace_mg");

    return laplace;
}

laplace_solver& get_laplace_solver(cl::context ctx)
{
    static laplace_solver laplace = get_laplace_solver_impl(ctx);
    return laplace;
}

void boot_initial_kernels(cl::context ctx)
{
    auto sum_buffers = [](execution_context& ctx, buffer_mut<valuef> inout, buffer<valuef> in, literal<v3i> dim)
    {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        if_e(lid >= dim.get().x() * dim.get().y() * dim.get().z(), [&]{
            return_e();
        });

        as_ref(inout[lid]) = as_constant(inout[lid]) + in[lid];
    };

    cl::async_build_and_cache(ctx, [=]{
        return value_impl::make_function(sum_buffers, "sum_buffers");
    }, {"sum_buffers"});

    auto calculate_aijaIJ = [](execution_context& ectx, buffer_mut<valuef> aij_aIJ_out, std::array<buffer<valuef>, 6> aIJ_packed, literal<v3i> dim)
    {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        if_e(lid >= dim.get().x() * dim.get().y() * dim.get().z(), [&]{
            return_e();
        });

        metric<valuef, 3, 3> met;

        for(int i=0; i < 3; i++)
            met[i, i] = 1;

        int index_table[3][3] = {{0, 1, 2},
                                 {1, 3, 4},
                                 {2, 4, 5}};

        tensor<valuef, 3, 3> aIJ;

        for(int i=0; i < 3; i++)
            for(int j=0; j < 3; j++)
                aIJ[i, j] = aIJ_packed[index_table[i][j]][lid];

        tensor<valuef, 3, 3> aij = lower_both(aIJ, met);

        as_ref(aij_aIJ_out[lid]) = sum_multiply(aij, aIJ);
    };

    cl::async_build_and_cache(ctx, [=]{
        return value_impl::make_function(calculate_aijaIJ, "aijaij");
    }, {"aijaij"});

    auto calculate_bssn_variables = [](execution_context& ectx,
                                       bssn_args_mem<buffer_mut<valuef>> out,
                                       buffer<valuef> cfl_reg, buffer<valuef> u,
                                       std::array<buffer<valuef>, 6> aIJ_summed,
                                       literal<v3i> dim) {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        pin(lid);

        if_e(lid >= dim.get().x() * dim.get().y() * dim.get().z(), [&] {
            return_e();
        });

        valuef cfl = cfl_reg[lid] + u[lid];

        metric<valuef, 3, 3> flat;

        for(int i=0; i < 3; i++)
            flat[i, i] = 1;

        metric<valuef, 3, 3> Yij = flat * pow(cfl, 4.f);

        int index_table[3][3] = {{0, 1, 2},
                                 {1, 3, 4},
                                 {2, 4, 5}};

        tensor<valuef, 3, 3> baIJ;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                baIJ[i, j] = aIJ_summed[index_table[i][j]][lid];
            }
        }

        tensor<valuef, 3, 3> Kij = lower_both(baIJ, flat) * pow(cfl, -2.f);

        //valuef gA = 1/(pow(cfl, 2));
        valuef gA = 1;
        tensor<valuef, 3> gB = {0,0,0};
        tensor<valuef, 3> cG = {0,0,0};

        valuef W = pow(Yij.det(), -1/6.f);
        metric<valuef, 3, 3> cY = W*W * Yij;
        //valuef K = trace(Kij, Yij.invert()); // 0
        valuef K = 0;

        cY = flat;

        tensor<valuef, 3, 3> cA = W*W * (Kij - (1.f/3.f) * Yij.to_tensor() * K);

        std::array<valuef, 6> packed_cA = extract_symmetry(cA);
        std::array<valuef, 6> packed_cY = extract_symmetry(cY.to_tensor());

        packed_cY[0] = packed_cY[0] - 1;
        packed_cY[3] = packed_cY[3] - 1;
        packed_cY[5] = packed_cY[5] - 1;
        gA = gA - 1;
        W = W - 1;

        for(int i=0; i < 6; i++)
        {
            as_ref(out.cY[i][lid]) = packed_cY[i];
            as_ref(out.cA[i][lid]) = packed_cA[i];
        }

        as_ref(out.K[lid]) = K;
        as_ref(out.W[lid]) = W;

        for(int i=0; i < 3; i++)
            as_ref(out.cG[i][lid]) = cG[i];

        as_ref(out.gA[lid]) = gA;

        for(int i=0; i < 3; i++)
            as_ref(out.gB[i][lid]) = gB[i];
    };

    cl::async_build_and_cache(ctx, [=]{
        return value_impl::make_function(calculate_bssn_variables, "calculate_bssn_variables");
    }, {"calculate_bssn_variables"});

    auto fetch_linear = [](execution_context& ectx, buffer<valuef> buf, literal<v3f> pos, literal<v3i> dim, buffer_mut<valuef> out)
    {
        using namespace single_source;

        valuei lid = value_impl::get_global_id(0);

        if_e(lid != 0, [&]{
            return_e();
        });

        auto get = [&](v3i pos)
        {
            pos = clamp(pos, (v3i){0,0,0}, dim.get() - 1);

            return buf[pos, dim.get()];
        };

        as_ref(out[0]) = function_trilinear(get, pos.get());
    };

    cl::async_build_and_cache(ctx, [=]{
        return value_impl::make_function(fetch_linear, "fetch_linear");
    }, {"fetch_linear"});
}

std::vector<float> extract_adm_masses(cl::context& ctx, cl::command_queue& cqueue, cl::buffer u_buf, t3i u_dim, float scale, const std::vector<black_hole_params>& params_bh)
{
    std::vector<float> ret;

    ///https://arxiv.org/pdf/gr-qc/0610128 6
    for(const black_hole_params& bh : params_bh)
    {
        ///Mi = mi(1 + ui + sum(m_j / 2d_ij) i != j
        t3f pos = world_to_grid(bh.position, u_dim, scale);

        cl::buffer u_read(ctx);
        u_read.alloc(sizeof(cl_float));

        cl::args args;
        args.push_back(u_buf, pos, u_dim, u_read);

        cqueue.exec("fetch_linear", args, {1}, {1});

        float u = u_read.read<float>(cqueue).at(0);

        float sum = 0;

        for(const black_hole_params& bh2 : params_bh)
        {
            if(&bh == &bh2)
                continue;

            sum += bh2.bare_mass / (2 * (bh2.position - bh.position).length());
        }

        float adm_mass = bh.bare_mass * (1 + u + sum);

        ret.push_back(adm_mass);
    }

    return ret;
}

float get_scale(float simulation_width, t3i dim);

std::pair<cl::buffer, initial_pack> initial_params::build(cl::context& ctx, cl::command_queue& cqueue, float simulation_width, bssn_buffer_pack& to_fill)
{
    std::optional<particle_data> gpu_particles;

    if(particles.size() > 0)
    {
        particle_data& dat = gpu_particles.emplace(ctx);
        dat.add(cqueue, particles);
    }

    auto [u_found, pack] = get_laplace_solver(ctx).solve(ctx, cqueue, simulation_width, dim,
                                                         [&ctx, &cqueue, this, &gpu_particles](t3i idim, float iscale)
    {
        initial_pack pack(ctx, cqueue, idim, iscale);

        for(auto& i : params_bh)
            pack.add(ctx, cqueue, i);

        for(auto& i : params_ns)
            pack.add(ctx, cqueue, i);

        if(gpu_particles)
            particle_initial_conditions(ctx, cqueue, pack.disc, gpu_particles.value(), idim, iscale);

        pack.finalise(cqueue);
        return pack;
    });

    for(neutron_star::data& dat : params_ns)
    {
        dat.finalise(ctx, cqueue, pack.disc, dim, get_scale(simulation_width, dim));
    }

    if(particles.size() > 0)
    {
        pack.gpu_particles = std::move(gpu_particles);
    }

    {
        cl::args args;

        to_fill.for_each([&](cl::buffer& buf) {
            args.push_back(buf);
        });

        args.push_back(pack.disc.cfl);
        args.push_back(u_found);

        for(int i=0; i < 6; i++)
            args.push_back(pack.disc.AIJ_cfl[i]);

        args.push_back(dim);

        cqueue.exec("calculate_bssn_variables", args, {dim.x() * dim.y() * dim.z()}, {128});
    }

    return {u_found, pack};
}

bool initial_params::hydrodynamics_wants_colour()
{
    for(const neutron_star::data& i : params_ns)
    {
        if(i.params.colour_aux.has_value())
            return true;
    }

    return false;
}

bool initial_params::hydrodynamics_enabled()
{
    return params_ns.size() > 0;
}

bool initial_params::particles_enabled()
{
    return particles.size() > 0;
}

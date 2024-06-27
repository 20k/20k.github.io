#ifndef METRICS_HPP_INCLUDED
#define METRICS_HPP_INCLUDED

#ifdef WORMHOLE
inline
metric<valuef, 4, 4> get_metric(const tensor<valuef, 4>& position) {
    using namespace single_source;

    valuef M = 0.01;
    valuef p = 1;
    valuef a = 0.001f;

    valuef l = position[1];

    valuef x = 2 * (fabs(l) - a) / (M_PI * M);

    valuef r = ternary(fabs(l) <= a,
                       p,
                       p + M * (x * atan(x) - 0.5f * log(1 + x*x)));


    valuef theta = position[2];

    metric<valuef, 4, 4> m;
    m[0, 0] = -1;
    m[1, 1] = 1;

    m[2, 2] = r*r;
    m[3, 3] = r*r * sin(theta)*sin(theta);

    return m;
}
#endif // WORMHOLE

#define BH_MASS 0
#define BH_SPIN 0

//#define SCHWARZSCHILD_EF
#ifdef SCHWARZSCHILD_EF
#define HAS_ACCRETION_DISK
#define BH_MASS 1
#define BH_SPIN 0

inline
metric<valuef, 4, 4> get_metric(const tensor<valuef, 4>& position) {
    using namespace single_source;
    valuef rs = 2 * BH_MASS;
    valuef r = position[1];
    valuef theta = position[2];

    metric<valuef, 4, 4> m;
    /*m[0, 0] = -(1-rs/r);
    m[1, 1] = 1/(1-rs/r);*/

    m[0, 0] = -(1-rs/r);
    m[1, 0] = 1;
    m[0, 1] = 1;

    m[2, 2] = r*r;
    m[3, 3] = r*r * sin(theta)*sin(theta);

    return m;
}
#endif

#define KERR
#ifdef KERR

#define HAS_ACCRETION_DISK
#define BH_MASS 1
#define BH_SPIN 0.9

inline
metric<valuef, 4, 4> get_metric(const tensor<valuef, 4>& position) {
    using namespace single_source;

    valuef r = position[1];
    valuef theta = position[2];

    metric<valuef, 4, 4> m;

    float M = BH_MASS;
    float a = BH_SPIN;

    valuef rs = 2 * M;

	valuef ct = cos(theta);
	valuef st = sin(theta);

	valuef R2 = r*r + a * a * ct * ct;
	valuef D = r*r + a * a - rs * r;

	valuef dv = (1 - (rs * r) / R2);
	valuef dv_dr = -2;
	valuef dv_dphi = (2 * a * st * st / R2) * (rs * r);
	valuef dr_dphi = 2 * a * st * st;
	valuef dtheta = -R2;
	valuef dphi = (st * st / R2) * (D * a * a * st * st - pow(a * a + r*r, 2.f));

	///v, r, theta, phi
	m[0, 0] = -dv;
	m[1, 0] = -0.5f * dv_dr;
	m[0, 1] = -0.5f * dv_dr;

	m[3, 0] = -0.5f * dv_dphi;
	m[0, 3] = -0.5f * dv_dphi;

	m[1, 3] = -0.5f * dr_dphi;
	m[3, 1] = -0.5f * dr_dphi;

	m[2, 2] = -dtheta;
	m[3, 3] = -dphi;

    return m;
}
#endif

auto metric_to_spherical = [](auto generic)
{
    return generic;
};

auto spherical_to_metric = [](auto spherical)
{
    return spherical;
};


#endif // METRICS_HPP_INCLUDED

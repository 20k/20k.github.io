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

#ifdef SCHWARZCHILD_EF
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
}
#endif

#define KERR
#ifdef KERR

#define HAS_ACCRETION_DISK
#define BH_MASS 1
#define BH_SPIN 0.9999

inline
metric<valuef, 4, 4> get_metric(const tensor<valuef, 4>& position) {
    using namespace single_source;

    valuef r = position[1];
    valuef theta = position[2];

    metric<valuef, 4, 4> m;

    float M = BH_MASS;
    float a = BH_SPIN;

    valuef rs = 2 * M;

    valuef E = r * r + a * a * cos(theta) * cos(theta);
    valuef D = r * r  - rs * r + a * a;

    m[0, 0] = -(1 - rs * r / E);
    m[1, 1] = E / D;
    m[2, 2] = E;
    m[3, 3] = (r * r + a * a + (rs * r * a * a / E) * sin(theta) * sin(theta)) * sin(theta) * sin(theta);
    m[0, 3] = 0.5f * -2 * rs * r * a * sin(theta) * sin(theta) / E;
    m[3, 0] = m[0, 3];

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

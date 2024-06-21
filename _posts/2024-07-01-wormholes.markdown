---
layout: post
title:  "Taking a trip through Interstellar's wormholes"
date:   2024-06-19 19:33:23 +0000
categories: C++
---

Hiyas! We're going to tie up some loose ends today, and complete the steps you need to render arbitrary metric tensors in general relativity. This is the last tutorial article in this segment - after this we'll be moving onto numerical relativity, so its time to clear up a few straggler topics:

1. A dynamic timestep
2. Workable camera controls/consistently orienting tetrads
3. Observers with velocity
4. Redshift
5. Accretion disk?

# The interstellar wormhole

The paper which describes interstellars wormhole is [this](https://arxiv.org/pdf/1502.03809) one. We want the fully configurable version, which are equations (5a-c)

Given a coordinate system $(t, l, \theta, \phi)$, and the parameters $M$ = mass, $a$ = wormhole length and $p$ = throat radius:

$$
\begin{align}
r &= p + M(x \; atan(x) - \frac{1}{2}ln(1 + x^2)) \;\; &where \;\; &|l| > a\\
r &= p \;\; &where \;\; &|l| < a\\
x &= \frac{2(|l| - a)}{\pi M}\\
\\
ds^2 &= -(1-2\frac{M}{r}) dt^2 + \frac{dr^2}{1-2\frac{M}{r}} + r^2 (d\theta^2 + sin^2 \theta d\phi^2)
\end{align}
$$

Note that there's a discontinuity in these equations at $|l|=a$ as given, so I swap (2) for a $<=$ instead. Using the raytracer we've produced, we can translate this to code:

```c++
metric<valuef, 4, 4> get_metric(const tensor<valuef, 4>& position) {
    using namespace single_source;

    valuef M = 0.01;
    valuef p = 1;
    valuef a = 0.001f;

    valuef l = position[1];

    valuef x = 2 * (fabs(l) - a) / (M_PI * M);

    //the analytic differentiator I've got can't handle multiple statements, so this becomes a ternary
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
```

With almost no changes to our code (other than updating our termination conditions, as $l$ can be $< 0$), we can immediately produce a rendering of this:

![wormhole](/assets/wormhole_1.png)

While the performance isn't too bad at ~70ms/frame, its definitely time to fix our timestepping - particularly to chop down on polar artifacts

# Dynamic Timestepping

One of the simplest and most effective strategies for dynamic timestepping is to ensure that the distance[^dist] a lightray moves is limited to some constant. The simplest version of this looks something like this:

[^dist]: We're using manhatten coordinate distance. You could use more advanced schemes - and these would work well, but bear in mind that calculating the timestep has a cost in itself and can end up having a significant overhead

```c++
valuef get_timestep(v4f position, v4f velocity)
{
    v4f avelocity = fabs(velocity);
    return 0.01f/max(max(avelocity.x(), avelocity.y()), max(avelocity.z(), avelocity.w()));
}
```

The works reasonably well, but it still gives a fairly low universal timestep, whereas we want to concentrate our timestepping more in areas that are likely to be a bit more visually interesting, ie near the object we're simulating in question. So something like this is a bit more interesting:

```c++
valuef get_timestep(v4f position, v4f velocity)
{
    valuef divisor = max(max(avelocity.x(), avelocity.y()), max(avelocity.z(), avelocity.w()));

    v4f avelocity = fabs(velocity);
    valuef normal_precision = 0.1f/divisor;

    valuef high_precision = 0.02f/divisor;

    return ternary(fabs(position[1]) < 3.f, high_precision, normal_precision);
}
```

This assumes that `position[1]` is a radial coordinate, which is not always true - and you'll need a generic system for calculating the distance from your object in question for this to work - but its worth it for the extra performance

With this in place, we get this result, which looks pretty great

![wormhole2](/assets/wormhole_2.png)

The singularities are barely noticable, and our performance is 60ms/frame @ 1080p

## Watch out for your integrator!

Be aware, not all integrators work with variable timesteps. For example, in this article we're using the leapfrog integrator:

```c++
        as_ref(velocity) = cvelocity + acceleration * dt;
        as_ref(position) = cposition + velocity.as<valuef>() * dt;
```

Where the velocity is updated, and then the position is updated with that new velocity. This integrator does not work with a dynamic timestep, and so you must swap to a different scheme - like euler

```c++
        as_ref(position) = cposition + cvelocity * dt;
        as_ref(velocity) = cvelocity + acceleration * dt;
```
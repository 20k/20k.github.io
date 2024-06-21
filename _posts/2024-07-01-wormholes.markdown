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
metric<valuef, 4, 4> metric(const tensor<valuef, 4>& position) {
    valuef M = 0.01;
    valuef a = 0.001;
    valuef p = 1;

    valuef l = position[1];

    valuef x = 2 * (fabs(l) - a) / (M_PI * M);

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
```
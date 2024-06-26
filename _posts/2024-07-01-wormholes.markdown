---
layout: post
title:  "Taking a trip through Interstellar's wormholes"
date:   2024-06-19 19:33:23 +0000
categories: C++
---

Hiyas! We're going to tie up some loose ends today, and complete the steps you need to render arbitrary metric tensors in general relativity. This is the last jumbo tutorial article I'm doing in this series - after this we'll be moving onto numerical relativity, so its time to clear up a few straggler topics:

1. A dynamic timestep
2. Workable camera controls/consistently orienting tetrads
3. Observers with velocity
4. Redshift
5. Accretion disk

Todo: Do I need to split this up into chapters?

# The interstellar wormhole

The paper which describes interstellar's wormhole is [this](https://arxiv.org/pdf/1502.03809) one. We want the fully configurable smooth version, which is equations (5a-c)

Given a coordinate system $(t, l, \theta, \phi)$, and the parameters $M$ = mass, $a$ = wormhole length and $p$ = throat radius:

$$
\begin{align}
r &= p + M(x \; atan(x) - \frac{1}{2}ln(1 + x^2)) \;\; &where \;\; &\mid l\mid > a \\
r &= p \;\; &where \;\; &\mid l\mid < a \\
x &= \frac{2(\mid l\mid - a)}{\pi M} \\
\\
ds^2 &= -(1-2\frac{M}{r}) dt^2 + \frac{dr^2}{1-2\frac{M}{r}} + r^2 (d\theta^2 + sin^2 \theta d\phi^2) \\
\end{align}
$$

Note that there's a discontinuity in these equations at $\mid l\mid = a$ as given, so I swap (2) for a $ <= $ instead. Using the raytracer we've produced, we can translate this to code:

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
    v4f avelocity = fabs(velocity);
    valuef divisor = max(max(avelocity.x(), avelocity.y()), max(avelocity.z(), avelocity.w()));

    valuef normal_precision = 0.1f/divisor;
    valuef high_precision = 0.02f/divisor;

    return ternary(fabs(position[1]) < 3.f, high_precision, normal_precision);
}
```

This assumes that `position[1]` is a radial coordinate, which is not always true - and you'll need a generic system for calculating the distance from your object in question for this to work - but its worth it for the extra performance

With this in place, we get this result, which looks pretty great:

![wormhole2](/assets/wormhole_2.png)

The singularities are barely noticable, and our performance is 60ms/frame @ 1080p. Truly incredible. I'm joking, but interactive framerates here are pretty decent - with a better integrator, some performance improvements, and general work this can be improved singificantly

## Watch out for your integrator!

Be aware, not all integrators work with variable timesteps. For example, in this article we've previously been using the leapfrog integrator:

```c++
        as_ref(velocity) = cvelocity + acceleration * dt;
        as_ref(position) = cposition + velocity.as<valuef>() * dt;
```

Where the velocity is updated, and then the position is updated with that new velocity. This integrator does not work with a dynamic timestep, and so you must swap to a different scheme - like euler

```c++
        as_ref(position) = cposition + cvelocity * dt;
        as_ref(velocity) = cvelocity + acceleration * dt;
```

# Camera Controls / Orienting Tetrad

Todo: This segment needs to be broken up

At the moment, we're constructing a tetrad directly from the underying metric. This works great, but results in a tetrad that - in polar metrics - tends to point directly at our object. This leads to very unintuitive camera controls as we move our camera around. In this segment we're going to commit some crimes against coordinate systems to get them to point roughly where we want them to

<iframe width="560" height="315" src="https://www.youtube.com/embed/L-sXQdiCkCY?si=4Hu52YdR1hoJZBUd" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

Fixing this is a bit of a pain. What we'd like to do is have our tetrads consistently point in a specific direction - any direction would do. While this isn't possible to do in the general case (because the tetrads and coordinates are inherently arbitrary), as it turns out - we actually can do this in a pretty wide variety of cases. Most coordinate systems aren't out to get you, so lets examine how to do this. This process is unfortunately going to be very coordinate dependent

First off, lets define 3 vectors in cartesian, where it'd sure be ideal if our coordinate basis vectors/tetrads pointed in that direction

$$
b_x = (1, 0, 0)\\
b_y = (0, 1, 0)\\
b_z = (0, 0, 1)
$$

Now, our coordinate system isn't in cartesian, so we need to convert these ideal vectors into our actual coordinate system, whatever that may be. Here we have polar coordinates, so lets use that illustratively

$$
\begin{align}
r &= \sqrt{x^2 + y^2 + z^2}\\
\theta &= acos(\frac{z}{r});\\
\phi &= atan2(y, x)
\end{align}
$$

We are dealing with *tangent* vectors, so we need to treat $b$ as a velocity - for that reason we want the total derivative[^howtodifferentiate] of the above set of equations:

[^howtodifferentiate]: If you don't know what this is: partially differentiate with respect to each variable (while holding the others constant, as per usual), and then sum all the resulting derivative. You should absolutely be automating this with dual numbers though!

$$
\begin{align}
dr &= \frac{x dx + y dy + z dz}{r}\\
d\theta &= \frac{z dr}{r^2 \sqrt{1 - \frac{z^2}{r^2}}} - \frac{dz}{r \sqrt{1 - \frac{z^2}{r^2}}}\\
d\phi &= \frac{x dy - y dx)}{x^2 y^2}\\
\end{align}
$$

And we'll call these new vectors $c_k$, for our new coordinate vectors[^inpractice] after plugging our cartesian coordinates into this equation. Note that we're only working with 3d vectors at the moment, not 4d vectors. We'll need to construct 4d vectors for our next step, and we do that as following:

[^inpractice]: In practice I always take a trip spherical coordinates first, to fix velocity vectors when $r < 0$ (inverting dr)

$$d_k^\mu = (0, c_k^0, c_k^1, c_k^2)$$

Where we hope that the 0th component is a timelike[^untested] coordinate. We now have 3 ($_k$) spatial vectors $d_k^\mu$, which have 4 components ($^\mu$) each. Now, calculate your tetrads as per the usual fashion, via gram-schmidt to get a valid tetrad frame $e_i^\mu$, followed by projecting your vectors $d_k^\mu$ into the local frame of reference (via the inverse tetrads). We'll call these projected local vectors $l_k^\mu$

$$l_i^\mu = e^i_\mu d^\mu_i$$

Note that $i$ ranges over 1-3, instead of 0-3. We now have 3 vectors - which we *hope* are spacelike. If they are - which they often will be - you can proceed

We can now orthonormalise these vectors to produce a new set of spacelike basis vectors for the frame of reference. Because we're in a minkowski metric - we can use gram schmidt with a trivial 3x3 identity matrix as the metric tensor

As you may have spotted, orthornormalising these vectors changes them - we start from a vector, which means that you can only preserve one of the original vectors after orthonormalisation. The correct vector to preserve is the 'up' vector that you use for your fixed mouse vertical axis - this means that as you spin the camera left and right, the axis you rotate around remains consistent, and this is a reasonable compromise

One key thing to note is that we only orient the *initial* tetrad if we're parallel transporting tetrads - this is one of the reasons why the technique works, often the camera ends up parallel transported into poorly behaved areas from well behaved nearly flat ones, so we only need it to work at our starting point

In code, this looks like this:

```c++
if(should_orient)
{
    v4f spher = GenericToSpherical(position.get());
    v3f cart = spherical_to_cartesian(spher.yzw());

    v3f cx = (v3f){1, 0, 0};
    v3f cy = (v3f){0, 1, 0};
    v3f cz = (v3f){0, 0, 1};

    //in practice we always convert to spherical coordinates first
    v3f sx = convert_velocity(cartesian_to_spherical, cart, cx);
    v3f sy = convert_velocity(cartesian_to_spherical, cart, cy);
    v3f sz = convert_velocity(cartesian_to_spherical, cart, cz);

    //This is why we convert to spherical
    //This is a specific correction for wormhole-typed metrics, to fix the camera's direction when the radial coordinate is negative, so that it lines up with +r
    sx.x() = ternary(spher.y() < 0, -sx.x(), sx.x());
    sy.x() = ternary(spher.y() < 0, -sy.x(), sy.x());
    sz.x() = ternary(spher.y() < 0, -sz.x(), sz.x());

    //now we convert our basis vectors into the metrics actual coordinate system. You'll need to provide SphericalToGeneric
    v4f gx = convert_velocity(SphericalToGeneric, spher, (v4f){0.f, sx.x(), sx.y(), sx.z()});
    v4f gy = convert_velocity(SphericalToGeneric, spher, (v4f){0.f, sy.x(), sy.y(), sy.z()});
    v4f gz = convert_velocity(SphericalToGeneric, spher, (v4f){0.f, sz.x(), sz.y(), sz.z()});

    inverse_tetrad itet = tetrads.invert();

    //only here for the compile times
    pin(gx);
    pin(gy);
    pin(gz);

    v4f lx = itet.into_frame_of_reference(gx);
    v4f ly = itet.into_frame_of_reference(gy);
    v4f lz = itet.into_frame_of_reference(gz);

    //only here for the compile times
    pin(lx);
    pin(ly);
    pin(lz);

    //start orthonormalisation off with our 'up' vector
    std::array<v3f, 3> ortho = orthonormalise(ly.yzw(), lx.yzw(), lz.yzw());

    ///after we orthonormalise, our axis are permuted, so undo that permutation
    v4f x_basis = {0, ortho[1].x(), ortho[1].y(), ortho[1].z()};
    v4f y_basis = {0, ortho[0].x(), ortho[0].y(), ortho[0].z()};
    v4f z_basis = {0, ortho[2].x(), ortho[2].y(), ortho[2].z()};

    //we now have our new basis vectors
    v4f x_out = tetrads.into_coordinate_space(x_basis);
    v4f y_out = tetrads.into_coordinate_space(y_basis);
    v4f z_out = tetrads.into_coordinate_space(z_basis);

    boosted.v[1] = x_out;
    boosted.v[2] = y_out;
    boosted.v[3] = z_out;
}
```

Which results in your metric looking like this as you fly around, making it way easier to implement camera controls, as now you have a consistent usable tetrad basis

[^untested]: You could potentially calculate which coordinate is timelike, and improve this step by setting the timelike coordinate to be 0 instead. However, your tetrad orientation would no longer be consistent - this is a basic problem with this method - we cannot truly globally orient our tetrad vectors

<iframe width="560" height="315" src="https://www.youtube.com/embed/lyfOMHNaLyw?si=n2Zrmubk-ux-wmzZ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

# Observers with velocity / Lorentz boosts

We'd like to give our observer a shove in a particular direction, instead of letting them drift completely freely. Amusingly enough, the interstellar wormhole has neglegible gravity, so we'll simply sit still forever if we can't represent a moving observer (instead of whatever we get out of the metric from Gram-Scmidt[^notethat])

[^notethat]: Note that boosting our observer in a direction *still* doesn't give us any more physical information as to where we're going, as our initial tetrad is still entirely arbitrary. We do however construct our boosts in the local frame of reference - so if we know where our tetrad is pointing (which is not always certain), we know the direction we boost towards. We have no information on our absolute velocity - as there is no absolute velocity

In a previous article, we learnt that given a set of tetrad vectors $e_i^\mu$, the 0th vector $e_0^\mu$ represents the velocity of our observer. If we want to represent an observer with a different speed, sadly we can't just modify that component - to transform the entire tetrad is a little more involved. The standard method for this is something called a lorentz boost (or more generally, a lorentz transform - which may include rotations), which - in special relativity - relates two observers moving at different speeds. A lorentz boost (or transform) in general relativity is often denoted by the symbol $\Lambda^i_{\;j}$, or $B^i_{\;j}$ for a lorentz boost specifically here. [This](https://arxiv.org/pdf/1106.2037) paper and [this](https://arxiv.org/pdf/2404.05744) paper contain more information in general

For us, we're looking to perform a lorentz boost in an arbitrary direction, and apply it to our basis vectors. Before we get there, we need to know what a 4-velocity is - something we've skimmed over a bit

## 4-velocities are not quite like 3-velocities

Lets imagine you have a regular good ol' velocity in 3 dimensional space. We'll normalise our velocities so that $1$ represents the speed of light, and $0$ represents stationary relative to our observer. We're going to examine the difference between:

1. 3-velocities parameterised by coordinate time $\frac{dx^i}{dt}$, your regular everyday concept of velocity. We will call this $v$, and its euclidian magnitude is $|v|$. $|v| = 1$ represents light
2. 4-velocities parameterised by coordinate time $\frac{dx^\mu}{dt}$
3. 4-velocities parameterised by an affine parameter $\frac{dx^\mu}{d\lambda}$
4. 4-velocities parameterised by proper time $\frac{dx^\mu}{d\tau}$
5. Timelike vs lightlike geodesics, with all of the different parameterisations

It is common to define 4-velocities as only being those velocities which are parameterised by proper time, but we're going with the generalised understanding here. Note that this segment is dealing with line elements in minkowski for simplicity, and as such is special relativity. That said, the language of special relativity, and general relativity are often not very similar at all, so this is very useful to write down

### Lightlike Geodesics

A lightlike geodesic is defined has having $ds^2 = 0$. If you've forgotten, remember that we evaluate this quantity with the metric tensor, or directly with the [line element](https://en.wikipedia.org/wiki/Line_element)

#### Coordinate time parameterisation

Constructing a 4-velocity for a lightlike geodesic which is parameterised by coordinate time is easy. Definitionally for a lightlike geodesic:

$$ds^2 = 0$$

We can use the line element for minkowski as such, plugging in our 3-velocity to get a 4-velocity

$$
ds^2 = 0 = -dt^2 + dx^2 + dy^2 + dz^2\\
-dt^2 = |v|^2\\
$$

We know a ray of light moves with a speed of $|v|=1$, therefore

$$
dt = \pm 1
$$

Giving us a final coordinate parameterised 4-velocity of $\frac{dx^\mu}{dt} = (\pm 1, v^0, v^1, v^2)$

#### Affine parameterisation

This is the most common parameterisation for a geodesic, and the one we will be using. Luckily, because the parameter for a geodesic has no particularly useful interpretation, we simply set $\lambda = t$ at the moment of construction

In minkowski, because spacetime is trivially flat, this parameterisation will always hold. In curved spacetime, this relation will only hold at the moment of construction, and then will diverge. This is because we use [different geodesic equations](https://en.wikipedia.org/wiki/Geodesics_in_general_relativity#Equivalent_mathematical_expression_using_coordinate_time_as_parameter) depending on the parameterisation we pick

#### Proper time parameterisation

The full line element reads:

$$ds^2 = -d\tau^2 = g_{\mu\nu}$$

Because for a lightlike geodesic, $ds^2 = 0$, $d\tau^2 = 0$. There is therefore no proper time parameterisation of a lightlike geodesic. For this reason, it is common to state that the velocity of a lightlike geodesic is not a 4-velocity, as it can never be parameterised by proper time

### Timelike Geodesics

A timelike geodesic is defined as when $ds^2 < 0$

#### Coordinate time parameterisation

Lets first up construct a timelike 4-velocity $\frac{dx^\mu}{dt}$ parameterised by coordinate time. We know that the $dx^0/dt$ component must be $1$, as $dx^0 = dt$. Lets check if $(1, v^0, v^1, v^2)$ is timelike:

$$
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2\\
\\
= -1 + dx^2 + dy^2 + dz^2 < 0\\
$$

Therefore, if $|v| < 1$, we do get a valid timelike geodesic $\frac{dx^\mu}{dt} = (1, v^0, v^1, v^2)$ parameterised by coordinate time

#### Proper time parameterisation

To transform from a coordinate time paramterisation, to a proper time parameterisation, we must multiply $\frac{dx^\mu}{dt}$ by $\frac{dt}{d\tau}$. How do we calculate $d\tau$? Easy, we use the line element for minkowski, and plug in our coordinate time velocity

$$\begin{align}
ds^2 &= -d\tau^2 = -dt^2 + dx^2 + dy^2 + dz^2\\
-d\tau^2 &= -1 + |v|^2\\
d\tau^2 &= 1 - |v|^2\\
d\tau &= \sqrt{1 - |v|^2}\\
\frac{dt}{d\tau} &= \frac{1}{\sqrt{1 - |v|^2}}\\
\end{align}
$$

You may recognise this as the formula for the lorentz factor $\gamma = \frac{dt}{d\tau}$. So to construct a 4-velocity parameterised by coordinate time from the 3-velocity $v^i$, we do:

$$\begin{align}
\frac{dx^\mu}{d\tau} &= \frac{dt}{d\tau} \frac{dx^\mu}{dt}\\
\frac{dx^\mu}{d\tau} &= \frac{1}{\sqrt{1 - |v^i|^2}}  \frac{dx^\mu}{dt}\\
\frac{dx^\mu}{d\tau} &= \gamma  \frac{dx^\mu}{dt}\\
\frac{dx^\mu}{d\tau} &= \gamma  (1, v^0, v^1, v^2)\\
\frac{dx^\mu}{d\tau} &= \gamma  (1, v)\\
\frac{dx^\mu}{d\tau} &=  (\gamma, \gamma v^0, \gamma v^1, \gamma v^2)\\
\end{align}
$$

Part of the reason why I'm spelling this out so explicitly is because all this notation is used to mean the same thing, so hopefully you can come back to this in the future

#### Affine parameterisation

Like with lightlike geodesics, we can construct 'an' affine parameterisation by setting $\lambda = t$ at the moment of construction, after which the two parameters diverge. This however is very uncommon, and is only mentioned for completeness. When you do this, the parameterisation has no physical interpretation

We can also construct an affine time parameterisation by setting $\lambda = \tau$, where $d\tau = ds^2 = -1$ (which is true in any proper time parameterisation). One very neat fact of proper time is that it *is* a general affine parameterisation, and so if we use a proper time parameterised geodesic and plug it through the geodesic equation specialised for the affine parameter (which is the one we use), this is perfectly valid

## Calculating a lorentz boost

We now know how to make an observer with a (timelike) 4-velocity in minkowski, by constructing it from a 3-velocity. To be very explicit, given a velocity in cartesian coordinates $d^i = (dx, dy, dz)$, where $\mid d\mid < 1$

$$
\frac{dx^\mu}{d\tau} = v_{local} = \frac{1}{\sqrt{1 - |d|^2}} (1, d^0, d^1, d^2)
$$

We then want to convert $v_{local}$ to $v$ by transforming it with the tetrad, as $v^\mu = e_i^\mu v^i_{local}$, to get our new observer velocity in our curved spacetime

If we have an initial 4-velocity $u^\mu = e_0^\mu$ of our tetrad, and we want to boost the tetrad to represent an observer with a 4-velocity $v$, the formula for the lorentz boost is this[^form]:

[^form]: [https://arxiv.org/pdf/2404.05744](https://arxiv.org/pdf/2404.05744) (18)

$$
\begin{align}
B^i_{\;\;j} &= \delta^i_{\;\;j} + \frac{(v^i + u^i)(v_j + u_j)}{1 + \gamma} - 2 v^i u_j\\
\gamma &= -v_i u^i
\end{align}
$$

$\delta^i_{\;\;j}$ is known as the kronecker delta. It is $1$ when $i==j$, and $0$ otherwise

Next up, we need to apply this lorentz boost to our tetrad vectors. Lets say our original tetrads are $e_i^\mu$, and our boosted tetrads are $\hat{e}_i^\mu$. We already know what the new value of $\hat{e}_0^\mu$ will be, as it must be $v$, but the general transform is:

$$\hat{e}^i_a = B^i_{\;\;j} e_a^j$$

In code:

```c++
//gets our new observer
v4f get_timelike_vector(v3f velocity, const tetrad& tetrads)
{
    v4f coordinate_time = {1, velocity.x(), velocity.y(), velocity.z()};

    valuef lorentz_factor = 1/sqrt(1 - (velocity.x() * velocity.x() + velocity.y() * velocity.y() + velocity.z() * velocity.z()));

    v4f proper_time = lorentz_factor * coordinate_time;

    ///put into curved spacetime
    return proper_time.x() * tetrads.v[0] + proper_time.y() * tetrads.v[1] + proper_time.z() * tetrads.v[2] + proper_time.w() * tetrads.v[3];
}

//does the actual tetrad boosting
tetrad boost_tetrad(v3f velocity, const tetrad& tetrads, const metric<valuef, 4, 4>& m)
{
    using namespace single_source;

    v4f u = tetrads.v[0];
    v4f v = get_timelike_vector(velocity, tetrads);

    v4f u_l = m.lower(u);
    v4f v_l = m.lower(v);

    valuef Y = -dot(v_l, u);

    ///https://arxiv.org/pdf/2404.05744 18
    tensor<valuef, 4, 4> B;

    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            valuef kronecker = (i == j) ? 1 : 0;

            B[i, j] = kronecker + ((v[i] + u[i]) * (v_l[j] + u_l[j]) / (1 + Y)) - 2 * v[i] * u_l[j];
        }
    }

    tetrad next;

    //multiply our old tetrads by our lorentz boost to get our new tetrad
    for(int a=0; a < 4; a++)
    {
        for(int i=0; i < 4; i++)
        {
            valuef sum = 0;

            for(int j=0; j < 4; j++)
            {
                sum += B[i, j] * tetrads.v[a][j];
            }

            next.v[a][i] = sum;
        }
    }

    return next;
}
```

## Wormhole Results

With a wormhole with params M = 0.01, p = 1, a = 1, and an observer of $(-0.2, 0, 0)$

<iframe width="560" height="315" src="https://www.youtube.com/embed/86528MigRMg?si=WF8o3tEySnRpykT2" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

Of course the foreground and background texture do not have to be the same, and can be distinguished by $l < 0$

M = 0.01, p = 1, a = 0.001

<iframe width="560" height="315" src="https://www.youtube.com/embed/tgIBbN5_q1E?si=1tkzCCEcvLnIDKjn" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

# Redshift

The equations for redshift in general relativity are pretty simple. First off, we define the redshift $z$, as follows[^derivation]:

$$z+1 = \frac{g_{\mu\nu} k^\mu_{emit} u^\mu_{emit}}{g_{\mu\nu} k^\mu_{obs} u^\mu_{obs}}$$

$k^\mu$ represents our geodesic's velocity, and u is the observer velocity. $u^\mu_{emit}$ specifically will be where our ray terminates, and $u^\mu_{obs} = e_0$ is our initial observer's velocity (after boosting!). Do note that the metric tensors are evaluated at different locations

Next up, we need to work out how our light changes, from our end frame of reference (defined by u^\mu_{emit}), to our initial frame of reference. Light has two properties - frequency/wavelength, and intensity. The equation for wavelength looks like this[^linky]

[^linky]: [https://www.astro.ljmu.ac.uk/~ikb/research/zeta/node1.html](https://www.astro.ljmu.ac.uk/~ikb/research/zeta/node1.html)

$$z+1 = \frac{\lambda_{obs}}{\lambda_{em}}$$

Once we have our new wavelength, we need to calculate the intensity. We can do this by calculating the lorentz invariant (constant in every frame of reference): $\frac{I_\nu}{\nu^3}$, where $\nu$ is your frequency. See [here](https://www.astro.princeton.edu/~jeremy/heap.pdf) 1.26 for details. Note that the quantity $I \lambda^3$ is also therefore lorentz invariant

So, to calculate our observed intensity, we say

$$I_{obs} \lambda_{obs}^3 = I_{emit} \lambda_{emit}^3$$

$$
I_{obs} = I_{emit} \frac{\lambda_{emit}^3}{\lambda_{obs}^3}
$$

Note that this equation is linear in terms of intensity, and only depends on the ratio of our wavelengths

Once we have our new frequency $\lambda_{obs}$, and $I_{obs}$, in theory we have everything we need to render our our final colour. We just have two unknowns, which are our initial intensity, and the initial wavelength

## Where do $I_{emit}$ and $\lambda_{emit}$ come from?

It depends what we're simulating. For our use case - redshifting a galaxy background, you'd need frequency and intensity data across the entire sky. A good starting point is over [here](http://aladin.cds.unistra.fr/hips/list), luckily we live in 2024 and a significant amount of this information is simply public - unfortunately these skymaps do not come with what units their intensity data is in, making them unusable[^digging] . Still, you can go find the original surveys - although it requires significant digging which I'm not going to do in this article

If you have a blackbody radiator, it becomes fairly straightforward, as given a temperature we can redshift that directly, via the equation:

$$T_{obs} = \frac{T_{emit}}{1+z}$$

[^digging]: This step is the bottleneck for actually achieving what we're trying to do here. Try as I might, I cannot find any standardised way to obtain anything corresponding to physical units (instead of raw data in unknown units). If you know, please contact me! It looks like [adadin](https://aladin.cds.unistra.fr/hips/HipsIn10Steps.gml) may be able to do what we want, but its certainly not straightforward. Apparently the 'default' unit is ADU, which is the raw CCD readout data, but its not even vaguely clear how to go about converting this into a calibrated physical unit

For our galaxy background, we're instead going to implement *illustrative* redshift, rather than attempting to find calibration data for hundreds of surveys

## Illustrative redshift

The key here is that we're going to discard physicality, and just show a measure of redshift. To do this, we first pick a fairly arbitrary wavelength - in my case I use $555nm$[^dontdoit], to represent green light. We then carry on as normal, and calculate $z+1$. Our intensity data is defined as the $Y$ component of the $XYZ$ colour space which represents power, see [here](https://en.wikipedia.org/wiki/SRGB#From_sRGB_to_CIE_XYZ). Put more simply: we convert to linear sRGB, and then calculate Y as:

[^dontdoit]: You might be tempted to try and do something more fancy like dominant colours or whatever, but the reality is when you're dealing only with a narrow range of visible light it makes 0 difference

$$Y = 0.2126 r + 0.7152 g + 0.0722 b$$

Once we've calculated our new intensity via the intensity equation, its then time to recolour our texture. We don't actually want to use our new wavelength - because it contains no useful colour information, but instead interpolate between red and blue depending on the value of $z$. $z$ has a range of $[-1, +inf]$, so we split into two branches

### Redshift Only

Redshift (z > 0):

```c++
new_colour = mix(old_colour, pure_red / 0.2126, tanh(z));
```

Redshift naturally fades to black as the intensity drops. The choice of `tanh` to map the infinite range to $[0, 1]$ is fairly arbitrary. The division by the constant is to ensure that our brightness remains the sam

### Blueshift Only

Blueshift (z > 1):

```c++
iv1pz = (1/(1 + z)) - 1;

interpolating_fraction = tanh(iv1pz);

new_colour = mix(old_colour, pure_blue / 0.0722, interpolating_fraction);
```

The mapping here is more complicated to replicate the same falloff as redshift. One question you might have is why we're dividing our colours by the constants: notice that they're the same constants we use to calculate $Y$. This ensures that our new colour is equivalent in power/brightness to the old one

One problem specific to blueshift is that our energy is unbounded, and our pixels can become infinitely bright. Its therefore much more aesthetically pleasing to spill over the extra energy into white once we max out the blue colour - which is shown in the full code sample

## Physically accurate wavelength rendering

Because our colour is artificial, this doesn't matter for us, but if you wanted to physically accurately render a wavelength, here's how you'd do it:

Human colour response to a frequency spectrum is defined by the LMS (long medium short - your eyes cone response) colour system. First up, you need to download the cie 1931 2 degree fov data from [here](http://www.cvrl.org/cmfs.htm).This gives you a table of colour matching functions which we convolve against our frequency data. This convolution returns a new set of tristimulus values in the LMS colour space, which represents how much each eye cone responds to a particular frequency

Once you have an LMS triplet, you convert that to the XYZ colour space, by calculating the inverse of [this](https://en.wikipedia.org/wiki/LMS_color_space#Hunt,_RLAB) matrix, which you use under D65 lighting

Then, convert that to the sRGB' linear colour space, via this [matrix](https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB), before finally using the CsRGB conversion below it

There will likely be a future article about accurately rendering black body radiators, but this one is long enough as it is

### Code

This is one of those things that's a lot easier to express in code, rather than equations:

```c++
//calculate Y of XYZ
valuef energy_of(v3f v)
{
    return v.x()*0.2125f + v.y()*0.7154f + v.z()*0.0721f;
}

v3f redshift(v3f v, valuef z)
{
    using namespace single_source;

    {
        valuef iemit = energy_of(v);

        ///z+1 = lobs / lemit
        ///lobs = lemit * (z+1)
        valuef test_wavelength = 555;
        valuef lobs = test_wavelength * (z + 1);

        ///Iobs lobs^3 = Iemit lemit^3
        valuef iobs = iemit * pow(test_wavelength, 3.f) / pow(lobs, 3.f);

        v = (iobs / iemit) * v;

        pin(v);
    }

    valuef radiant_energy = energy_of(v);

    v3f red = {1/0.2125f, 0.f, 0.f};
    v3f green = {0, 1/0.7154, 0.f};
    v3f blue = {0.f, 0.f, 1/0.0721};

    mut_v3f result = declare_mut_e((v3f){0,0,0});

    //redshift
    if_e(z >= 0, [&]{
        as_ref(result) = mix(v, radiant_energy * red, tanh(z));
    });

    //blueshift
    if_e(z < 0, [&]{
        //map red and blueshift to the same sacles
        valuef iv1pz = (1/(1 + z)) - 1;

        valuef interpolating_fraction = tanh(iv1pz);

        v3f col = mix(v, radiant_energy * blue, interpolating_fraction);

        //calculate spilling into white. This works out how much energy we are unable to represent
        //and spills the rest into the other colours
        {
            valuef final_energy = energy_of(clamp(col, 0.f, 1.f));
            valuef real_energy = energy_of(col);

            valuef remaining_energy = real_energy - final_energy;

            col.x() += remaining_energy * red.x();
            col.y() += remaining_energy * green.y();
        }

        as_ref(result) = col;
    });

    as_ref(result) = clamp(result, 0.f, 1.f);

    return declare_e(result);
}
```

For a schwarzschild black hole, a velocity boosted 0.5 towards the black hole, and a starting position of $(0, 5, 0, 0)$, you end up with this

![Towards](/assets/blueshift_bh.png)

![Away](/assets/redshift_bh.png)

From the front and back. Notice that we blueshift in the direction of travel, and see a redshift in the opposite direction

# Rendering Kerr

## Accretion Disks

No black hole is complete without an accretion disk, so today we're going to be looking at implementing a reasonable accretion disk model. We're going to look at whats called a [thin-disk model](https://www.emis.de/journals/LRG/Articles/lrr-2013-1/articlese5.html) (98-100), because it has a straightforward algebraic solution. In the future, we'll be simulating accretion disks directly

One thing we should examine first is the concept of an innermost stable orbit, or ISCO for short

### ISCO

Outside of the ISCO, circular trajectories around the rotational plane of a black hole are stable[^stable]. Within the ISCO, orbits are unstable (as circular orbits don't exist) - and matter spirals into the black hole quickly. This is known as the 'plunging' region, and will crop up a lot in future articles

[^stable]: No orbit in general relativity is ever truly stable, as everything emits gravitational waves and orbits decay. Near a black hole, this effect is particularly intense. Inwards accretion for a disk I believe is primarily driven by fluid dynamics, but this is outside of my area of knowledge

For a schwarzschild black hole, the isco is defined as $r_{isco} = 3rs = 6M$. For a kerr black hole, the ISCO is defined as follows[^iscoeq] for a black hole with mass $M$, and spin $a$:

$$
\begin{align}
X &= a/M\\
Z_1 &= 1 + (1-X^2)^{\frac{1}{3}} ((1+X)^{\frac{1}{3}}(1-X)^{\frac{1}{3}})\\
Z_2 &= (3 X^2 + Z_1^2)^{\frac{1}{2}}\\
r_{isco} &= M (3 + Z_2 \pm ((3 - Z_1)(3 + Z_1 + 2Z_2))^{\frac{1}{2}})\\
\end{align}
$$

The retrograde ISCO is higher than the prograde ISCO, so flip the sign appropriately when calculating

[^iscoeq]: See [wikipedia](https://en.wikipedia.org/wiki/Innermost_stable_circular_orbit#Rotating_black_holes), or 2.21 in [this](https://articles.adsabs.harvard.edu/pdf/1972ApJ...178..347B) paper

## Accretion Disk Regions

Because orbits within the ISCO are unstable, matter depletes from this region very quickly. For this reason, accretion disks are often modelled as having a gap between the event horizon, and the ISCO - which we will follow[^alert]. For the model we're looking at, [here](https://www.emis.de/journals/LRG/Articles/lrr-2013-1/articlese5.html) equations 98-100, we have three regions for us to work with, which are called:

[^alert]: Note that one of the papers linked, [this](https://arxiv.org/pdf/1110.6556) one, was originally what this article implemented, and claims to model the plunging region. This ended up being a significantly delay, as as far as I can tell the equations for the plunging region fundamentally do not work - you can straightforwardly prove that the radial velocity profile is imaginary, and tends to infinity, simultaneously. This is a bit unfortunate

1. The inner region
2. The middle region
3. The outer region

Distinguishing between these three regions is done by two values

1. Gas pressure vs radiation pressure. When gas pressure < radiation pressure, we're either in the inner or outer region
2. Whether opacity is driven by free-free interactions, or electron scattering. If free-free > electron scattering, we're in the outer region, otherwise we're in the inner or middle region

The details of this are interesting[^interesting], but we're going to focus on how to actually implement this rather than what it means (unfortunately) - there's lots of information available in the linked articles. The idea is to iterate from the innermost boundary of our accretion disk ($r = r_{isco}$), and terminate at some finite radius away from the accretion disk ($r = 100M$). We know at the start, we must be in region #1 - the inner region - at the isco

[^interesting]: If there was time, it would be interesting to lay it all out. This is one of the downsides of writing articles which are intended to be implementation focused, and a bit jumbo like this one. While these papers themselves contain all the theory you need, the thing we are lacking is actually how to implement this. In the future, I may revisit accretion disks in a lot more detail

To distinguish when we transition from region 1, to region 2, which want to calculate (gas pressure / radiation pressure), and if its > 1 move into the middle region. This quantity is labelled $\beta / (1-\beta)$ I believe

To distinguish when we transition from region 2, to region 3, we calculate the quantity $T_ff / T_es$, which is the free-free opacity / the electron scattering opacity. When this quantity is > 1, we swap to region 3

## Other details

To be able to implement this, we first need to calculate our spatial functions $A$ $B$ $C$ $D$ $E$ $Q_0$ $Q$, as well as $y_1$ $y_2$ $y_3$, and $y_0$. While not mentioned on the page we're looking at, subscript $\_0$ variables are evaluated at the ISCO. We do have an explicit formula for $Q_0$, which means that the only other one we need is $y_0 = \frac{r_{isco}}{M}^{\frac{1}{2}}$

The meaning of the variables here are, in order

|variable|meaning|
|F|surface radiant flux (ie brightness)|
|$\Sigma$|surface density|
|H|Disk thickness|
|$\rho_0$ |rest mass density in the local rest frame|
|$T$ |temperature|
|$\beta / (1-\beta)$|ratio of gas pressure to radiation pressure|
|$T_{ff}/T_{es}$|Ratio of free-free opacity to electron scattering opacity|

With this, we should have everything[^onemore] we need to implement this correctly

[^onemore]: The equations we're implementing mention the eddington luminosity. We don't need this as we're using a fraction of it, but you may want this for yourself, check out over [here](https://www-astro.physics.ox.ac.uk/~garret/teaching/lecture7-2012.pdf) for details. I've added an implementation of this into the code sample as well

## Code

The equations here - as written, are long and complicated to implement correctly. I won't reproduce the equations in full here - it just introduces a risk of mistakes, but you can find the code for implementing this method, and producing a nice accretion disk texture over here

This gives pretty nice results. For $M=1$[^geometric], $a=0.6$, $\dot{m} = 0.3$:

[^geometric]: Note that this is in geometric units

![Disk](/assets/disk.png)

In practice you don't need to use a 2d texture of an accretion disk - because its spherically symmetric, you could just take a radial slice

## Rendering

Rendering the accretion disk is extremely straightforward. Because we model it as a very thin disk, you can check as a geodesic crosses the equatorial plane ($theta = n pi + pi/2$), sample the accretion disk texture, and render that colour out if we hit. Or you could carry on rendering, and add it to the background colour

One key thing to note here is that we're going to implement redshift, and to do that, we need to know the velocity of the fluid in the accretion disk. We already know that its moving approximately in circular orbits (otherwise it'd escape, or hit the black hole), but what is that as a velocity vector specifically?

Well, in the equatorial plane for circular orbits, we know that $dr = 0$, and $d\theta = 0$. This means that we're only left to solve for $dt$, and $d\phi$.

https://physics.stackexchange.com/questions/502796/how-to-derive-the-angular-velocity-of-circular-orbits-in-kerr-geometry

https://articles.adsabs.harvard.edu/pdf/1972ApJ...178..347B

Luckily other people have put in the legwork, and figured out that the angular velocity of a geodesic in the kerr spacetime is

$$
w = \frac{1}{r^{3/2} + a}
$$

Given that we know that our geodesic must point in the $d\phi$ direction, we can work out the $d\phi$ component as being $wr$. This means that $dt = d\phi/w = 1/r$. We now have the 4-velocity of a geodesic on our equatorial orbit

One extremely tricky aspect is that our geodesics in general will not intersect the equatorial plane at $\theta = pi/2$, and could be any multiple of it.

# Taking a trip through Interstellar's wormhole

Video:

# Taking a trip through Interstellar's other wormhole

So, the title wasn't a typo if a bit clickbaity. Interstellar contains a spinning black hole, which we can model by the Kerr (or Kerr-Newman, with charge) metric. The interior of these metrics are actually pretty interesting. In addition to a ringularity (a ring singularity), the centre of a Kerr style black hole contains a wormhole, and copious time travel - which are certainly unusal things to find[^tofind]

[^tofind]: Though I feel like at this point, physicists would be *more* happy if there were a library inside a black hole instead of singularities and time travel

The metric for a Kerr-Newman black hole in ingoing[^kerr] coordinates looks like this:

[^kerr]: http://www.scholarpedia.org/article/Kerr-Newman_metric (47), with signs flipped due to the metric signature

$$ds^2 = -(1-\frac{2Mr - Q^2}{R^2}) \;dv^2 + 2 dv dr - 2 a \frac{\sin^2 \theta}{R^2}(2 M r - Q^2)\; dv d\phi - 2 a \sin^2 \theta \;dr d\phi + R^2\; d\theta^2 - \frac{\sin^2 \theta}{R^2}(\Delta a^2 \sin^2 \theta - (a^2 + r^2)^2)\; d\phi^2
$$

where

$$
\begin{align}
R^2 &= r^2 + a^2 \cos^2 \theta\\
\Delta &= r^2 + a^2 - 2 M r + Q^2
\end{align}
$$

$Q$ is our charge, $M$ is our mass parameter, and $a$ is the black hole's spin

In code:

todo: code

This gives us a pretty nice looking black hole, which looks like this:

The more interesting part is taking a trip inside the black hole to examine the ringularity. Starting at position p=todo, and giving our observer an initial velocity of todo, we end up with a result that looks like this:

Purrfect!

While I partly built this for the memes, its worth noting that the interior of a black hole and the wormhole itself is generally considered to be a mathematical artifact, and that it wouldn't show up in reality. With that in mind, nobody really has a clue whats inside a black hole!

# The end

That's the end of this small series of rendering arbitrary analytic metric tensors - there's a lot more we could do here, and there'll be articles in the future on this topic, but they'll likely be on specific elements of this topic rather than the more broad approach we've taken here. The focus after this is largely going to be on numerical relativity - simulating the evolution of spacetime itself, and how to do that on a GPU

With that in mind, here's the current list of topics that I'm going to get to in the future:

1. Examining the BSSN equations and the ADM formalism in numerical relativity, to evolve some simple numerical spacetimes
2. Binary black hole mergers
3. Fast, simple and accurate symmetric laplacian solving
4. Gravitational wave extraction
5. Approximate numerical raytracing
6. Full numerical raytracing
7. Stability modifications, and constraint damping
8. Integrators for the EFEs
9. Examining and calculating mass in GR
10. Boundary conditions
11. Particle system dynamics, including massless and massive accretion disks
12. Investigating the hoop conjecture, mainly because its cool
13. Negative energy in general relativity
14. Relativistic eularian hydrodynamics
15. Neutron star initial conditions, and equations of state
16. Relativistic smooth particle hydrodynamics, which is a fun phrase to say
17. Relativistic magnetohydrodynamics
18. CCZ4/Z4CC, and friends
19. Solving elliptic equations

On top of this, there are a number of misc topics to cover in analytic metrics:

1. 4d Triangle rasterisation
2. Time travel, and CTCs
3. More interesting metric tensors, like the misner spacetime, cosmic strings, or analytic binary black hole metrics
4. Near-massless (dust) particle systems
5. Near-massless (dust) accretion disks
6. Wave propagation in curved spacetimes
7. 256x anisotropic texture filtering
8. Relativistic volumetric rendering
~~9. QFT in curved spacetime~~ ok not that one

This is going to take a hot minute to write up in a tutorially format - though its all largely work I've done previously (except smooth particle hydrodynamics, and magnetohydrodynamics). This kind of information doesn't really exist outside of papers - and there are no gpu implementations - so it seems good to write it up

Until then!

todo: Cat photo


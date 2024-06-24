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
At the moment, we're constructing a tetrad directly from the underying metric. This works great, but results in a tetrad that - in polar metrics - tends to point directly at our object. This lads to very unintuitive camera controls as we move our camera around. In this segment we're going to commit some crimes against coordinate systems to get them to point roughly where we want them to

\<iframe width="560" height="315" src="https://www.youtube.com/embed/L-sXQdiCkCY?si=4Hu52YdR1hoJZBUd" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

Fixing this is a bit of a pain. What we'd like to do is have our tetrads consistently point in a specific direction - any direction would do. While this isn't possible to do in the general case (because the tetrads and coordinates are inherently arbitrary), as it turns out - we actually can do this in a pretty wide variety of cases. Most coordinate systems aren't out to get you, so lets examine how to do this. This process is unfortunately going to be very coordinate dependent

First off, lets define 3 vectors in cartesian, where it'd sure be ideal if our coordinate basis vectors/tetrads pointed in that direction

$$
b_x = (1, 0, 0)\\
b_y = (0, 1, 0)\\
b_z = (0, 0, 1)\\
$$

Now, our coordinate system isn't in cartesian, so we need to convert these ideal vectors into our actual coordinate system, whatever that may be. Here we have polar coordinates, so lets use that illustratively

todo: the rest of this
$$
r = \sqrt{x^2 + y^2 + z^2}
$$

We are dealing with *tangent* vectors, so we need to treat $b$ as a 'velocity' - for that reason we want the derivative of the above equation:

todo: the rest of this
$$
dr = whatever
$$

And we'll call these new vectors $c_k$, for our new coordinate vectors. Note that we're only working with 3d vectors at the moment, not 4d vectors. We'll need to construct 4d vectors for our next step, and we do that as following:

$$d_k^\mu = (0, c_k^0, c_k^1, c_k^2)$$

Where we hope that the 0th component is a timelike[^untested] coordinate. We now have 3 ($_k$) spatial vectors $d_k^\mu$, which have 4 components ($^\mu$) each. Now, calculate your tetrads as per the usual fashion, via gram-schmidt to get a valid tetrad frame $e_i^\mu$, followed by projecting your vectors $d_k^\mu$ into the local frame of reference. We'll call these projected local vectors $l_k^\mu$

$$l_i^\mu = e^i_\mu d^\mu_i$$

Note that $i$ ranges over 1-3, instead of 0-3

We now have 3 vectors - which we *hope* are spacelike. If they are - which they often will be - we can proceed

We can now orthonormalise these vectors to produce a new set of spacelike basis vectors for our tetrad. Because we're in a minkowski metric, this orthonormalisation is very easy - we can use gram schmidt, but with a trivial 3x3 identity matrix as the metric tensor

As you may have spotted, orthornormalising these vectors changes them - we start from a vector, which means that we *can* preserve one of these vectors as pointing in our original direction, as long as it wasn't null or timelike. The correct vector to preserve is the 'up' vector - this means that as you spin the camera left and right, the axis you rotate around remains consistent, and this provides the best user experience

In code, this looks like this:

```c++
todo: port my bad code
```

Which results in your metric looking like this as you fly around, making it way easier to implement camera controls!

[^untested]: You could potentially calculate which coordinate is timelike, and improve this step by setting the timelike coordinate to be 0 instead. However, your tetrad orientation would no longer be consistent - this is a basic problem with this method - we cannot truly globally orient our tetrad vectors

Todo: video

# Observers with velocity / Lorentz boosts

In a previous article, we learnt that given a set of tetrad vectors $e_i^\mu$, the 0th vector $e_0^\mu$ represents the velocity of our observer. If we want to represent an observer with a different speed, sadly we can't just modify that component - to transform the entire tetrad is a little more involved. The standard method for this is something called a lorentz boost (or more generally, a lorentz transform - which may include rotations), which - in special relativity - relates two observers moving at different speeds. A lorentz boost (or transform) in general relativity is often denoted by the symbol $\Lambda^i_{\;j}$, or $B^i_{\;j}$ for a lorentz boost specifically here. [This](https://arxiv.org/pdf/1106.2037) paper and [this](https://arxiv.org/pdf/2404.05744) paper contain more information in general

For us, we're looking to perform a lorentz boost in an arbitrary direction, and apply it to our basis vectors. Before we get there, we need to know what a 4-velocity is - something we've skimmed over a bit

## 4-velocities are not quite like 3-velocities

Lets imagine you have a regular good ol' velocity in 3 dimensional space. We'll normalise our velocities so that $1$ represents the speed of light, and $0$ represents stationary relative to our observer. We're going to examine the difference between:

1. 3-velocities parameterised by coordinate time $\frac{dx^i}{dt}$, your regular everyday concept of velocity. We will call this $v$, and its euclidian magnitude is $|v|$
2. 4-velocities parameterised by coordinate time $\frac{dx^\mu}{dt}$
3. 4-velocities parameterised by an affine parameter $\frac{dx^\mu}{d\lambda}$
4. 4-velocities parameterised by proper time $\frac{dx^\mu}{d\tau}$
5. Timelike vs lightlike geodesics, with all of the different parameterisations

It is common to define 4-velocities as only being those velocities which are parameterised by proper time, but we need to encompass every kind of geodesic

### Lightlike Geodesics

#### Coordinate time parameterisation

Constructing a 4-velocity for a lightlike geodesic which is parameterised by coordinate time is easy. Definitionally for a lightlike geodesic:

$$ds^2 = 0$$

We can use the line element for minkowski as such, plugging in our 3-velocity to get a 4-velocity

$$
ds^2 = 0 = -dt^2 + dx^2 + dy^2 + dz^2\\
-dt^2 = |v|^2\\
$$

We know a ray of light moves with a speed of $1$, therefore

$$
dt = \pm 1
$$

#### Affine parameterisation

This is the most common parameterisation for a geodesic, and the one we will be using. Luckily, because the parameter for a geodesic has no particularly useful interpretation, we simply set $\lambda = t$

In minkowski, because spacetime is trivially flat, this parameterisation will always hold, and affine and coordinate parameterisations are equivalent for light rays. In curved spacetime, this will only hold at the moment of construction, and then will diverge. This is because we're use [different geodesic equations](https://en.wikipedia.org/wiki/Geodesics_in_general_relativity#Equivalent_mathematical_expression_using_coordinate_time_as_parameter) depending on the parameterisation we pick

### Lightlike 4-Velocities

From our perspective, we can model a ray of light has having a velocity through space as eg $l_3=(1, 0, 0)$, meaning something moving with a speed of 1, in the +x direction. Turning this into lightlike 4-velocity[^definitions] in general relativity, in minkowski spacetime, is easy. We represent this as:

[^definitions]: Note that there is seemingly disagreement as to whether or not to call the velocities of lightlike geodesics 4-velocities

$$
l_4^\mu = (\pm 1, l_3^0, l_3^1, l_3^2)
$$

We can verify with our minkowski metric tensor $n_{\mu\nu}$ that this is lightlike. One thing of key importance here is the parameterisation - our 3-velocity is parameterised by coordinate time $l_3^i = dx^i/dt$, and therefore the lightray $l_4^\mu$ is also parameterised by coordinate time, $l_4^\mu = \frac{dx^\mu}{dt}$. For our geodesic equation we actually want an affine parameterisation $l_4^\mu = \frac{dx^\mu}{d\lambda}$, as we're going to use the affine form of the geodesic equation. The nice thing is, because we don't need a physical interpretation for the parameter of the geodesic, we can set $\lambda = t$. Its worth noting that while this equation generally holds true in minkowski as space is very trivially flat, it is only true for one moment in time in curved spacetime

The parameterisations diverge as we follow the geodesic forwards, because we use [different geodesic equations](https://en.wikipedia.org/wiki/Geodesics_in_general_relativity#Equivalent_mathematical_expression_using_coordinate_time_as_parameter) depending on whether or not we consider our parameterisation to be affine $\lambda$, or wish to enforce that $\lambda = t$. Note that the affine parameterisation is also often called $s$, or $ds$ for the delta

Please also note: its quite common to consider 4-velocities to be specifically only parameterised by proper time, and lightlike geodesics cannot be parameterised by proper time. Therefore we're abusing terminology a bit

### Timelike 4-Velocities

Constructing a timelike geodesic is therefore a bit more tricky. Lets start off with a regular 3-velocity $v^i = dx^i/dt$, and imagine we're trying to construct a timelike 4-velocity $v_4^\mu = dx^\mu/d\tau$. We're looking for a specific parameterisation by $d\tau$, and constructing that requires more work

Lets first up construct a timelike 4-velocity ($dx^\mu/dt$) parameterised by coordinate time. We know that the $dx^0/dt$ component must still be $1$, as $dx^0 = dt$. Lets check if $(1, v^0, v^1, v^2)$ is timelike:

$$
ds^2 = -dt^2 + dx^2 + dy^2 + dz^2\\
\\
= -1 + dx^2 + dy^2 + dz^2 < 0\\
\\s
$$

Therefore, if $|v| < 1$, we do get a valid timelike geodesic $\frac{dx^\mu}{dt} = (1, v^0, v^1, v^2)$ parameterised by coordinate time. If we want to change our parameterisation, we need to multiply by the quantity $\frac{dt}{d\tau}$. Using the line element for Minkowski again:

$$
\begin{align}
-d\tau^2 &= ds^2 = -1 + dx^2 + dy^2 + dz^2\\
d\tau &= ds^2 = \sqrt{1 - (dx^2 + dy^2 + dz^2)}\\
dt/d\tau^2 &= ds^2 = 1/\sqrt{1 - (dx^2 + dy^2 + dz^2)}\\
dt/d\tau &= 1 / \sqrt{1 - |v^i|^2}\\
\end{align}
$$

If you're at all familiar with general or special relativity, you will recognise this as the equation for the lorentz factor $\gamma$, and indeed $\gamma = dt/d\tau = \frac{1}{\sqrt{1 - |v|^2}}$. Lets proceed now with calculating our timelike 4-velocity

$$\begin{align}
\frac{dx^\mu}{d\tau} &= \frac{dt}{d\tau} \frac{dx^\mu}{dt}\\
\frac{dx^\mu}{d\tau} &= \frac{1}{\sqrt{1 - |v^i|^2}}  \frac{dx^\mu}{dt}\\
\frac{dx^\mu}{d\tau} &= \gamma  \frac{dx^\mu}{dt}\\
\frac{dx^\mu}{d\tau} &= \gamma  (1, v^0, v^1, v^2)\\
\frac{dx^\mu}{d\tau} &= \gamma  (1, v)\\
\frac{dx^\mu}{d\tau} &=  (\gamma, \gamma v^0, \gamma v^1, \gamma v^2)\\
\end{align}
$$

Part of the reason why I'm spelling this out so explicitly is because this notation is thrown around a lot, so hopefully you can come back to this in the future

## Calculating the lorentz boost

If we have an initial 4-velocity $u$ of our tetrad, and we want to boost the tetrad to represent an observer with a 4-velocity $v$, the formula is this[^form]:

[^form]: [https://arxiv.org/pdf/2404.05744](https://arxiv.org/pdf/2404.05744) (18)

$$
\begin{align}
B^i_{\;\;j} &= \delta^i_{\;\;j} + \frac{(v^i + u^i)(v_j + u_j)}{1 + \gamma} - 2 v^i u^j\\
\gamma &= -v_i u^i
\end{align}
$$

I need more of the construction so I can talk about it, 4-velocity construction etc. Existing literature makes this overly complicated, but that does mean I need to rederive the method I use in this part of the article

# Redshift

## Physically accurate redshift

Calculating a physically accurate rendering of redshift is an extremely involved process, and I am not aware of any visually accurate simulations of this. This may surprise you if you know general relativity, because the equations for redshift are very simple. I will outline the full process below of rendering redshift, and then we will use a visual approximation to skip the difficult steps

1. We first need a skymap across all frequencies, giving us the different intensities. A good starting point is over [here](http://aladin.cds.unistra.fr/hips/list), luckily we live in 2024 and a significant amount of this information is simply public - unfortunately these skymaps do not come with what units their intensity data is in, making them unusable[^digging] . Still, you can go find the original surveys - although it often requires significant digging

[^digging]: This step is the bottleneck for actually achieving what we're trying to do here. Try as I might, I cannot find any standardised way to obtain anything corresponding to physical units (instead of raw data in unknown units). If you know, please contact me! It looks like [adadin](https://aladin.cds.unistra.fr/hips/HipsIn10Steps.gml) may be able to do what we want, but its certainly not straightforward. Apparently the 'default' unit is ADU, which is the raw CCD readout data, but its not even vaguely clear how to go about converting this into a calibrated physical unit

2. We then perform our geodesic tracing as per normal. At our starting point, the camera, we calculate the quantity $g_{\mu\nu} k^\mu_{obs} u^\mu_{obs}$, where $k_{obs}$ is our geodesic velocity, and $u_{obs}$ is our observer velocity

3. At our rays termination point, we need to define some observer that was considered to have emitted that ray, and get the velocity $u_{emit}$. Eg, if your ray originated from an accretion disk, you need the velocity of the matter. You might calculate a new set of tetrads $e_k$ at our termination point, and consider $u_{emit} = e_0$ to be the velocity of whatever emitted the ray. Then, calculate $g_{\mu\nu} k^\mu_{emit} u^\mu_{emit}$. Remember that because we're tracing rays backwards in time, our termination point is where the ray was emitted!

4. Next we calculate the quantity $z$, which is defined as $z+1 = \frac{g_{\mu\nu} k^\mu_{emit} u^\mu_{emit}}{g_{\mu\nu} k^\mu_{obs} u^\mu_{obs}}$, bearing in mind that our two metric tensors are evaluated at different coordinates

5. The change in frequency of light due to redshift is defined as $v_{obs} = \frac{v_{emit}}{z+1}$

6. The intensity $I_{obs}$ is calculated by using the [lorentz invariant quantity](https://physics.stackexchange.com/questions/321220/derivation-of-the-lorentz-transform-of-brightness-dp-d-omega) $\frac{I_{emit}}{v_{emit}^3}$, where $v$ is frequency, and $I$ is spectral radiance[^whatisthis] . Lorentz invariant quantities do not change based on your frame of reference, therefore $\frac{I_{emit}}{v_{emit}^3} = \frac{I_{obs}}{v_{obs}^3}$, and $ \frac{I_{emit} v_{obs}^3}{v_{emit}^3} = I_{obs}$, our observed intensity of light. Note the linearity of this equation in terms of the intensity of light

[^whatisthis]: [Radiance per unit frequency](https://en.wikipedia.org/wiki/Spectral_radiance)

7. We now have a way to calculate the change in intensity, and frequency, of emitted rays. We now sample all the frequencies from our skymaps observed in step 1, and calculate the new frequencies and intensities in our local frame of reference by applying the above equation to out frequency + intensity distribution. Next up, we need to calculate what this actually looks like to a human being

8. Human colour response to a frequency spectrum is defined by the LMS (long medium short - your eyes cone response) colour system. First up, you need to download the cie 1931 2 degree fov data from [here](http://www.cvrl.org/cmfs.htm). Using later observers may work as well. This gives you a table of colour matching functions[^moredetail] which we convolve against our frequency data. This convolution returns a new set of tristimulus values in the LMS colour space, which represents how much each eye cone responds to a particular frequency

[^moredetail]: We're glossing over colouriometry theory at high speed here, the short version is that you can work out how much human eyes respond to different frequencies of light

9. Once we have an LMS triplet, we then convert that to the XYZ colour space, by calculating the inverse of [this](https://en.wikipedia.org/wiki/LMS_color_space#Hunt,_RLAB) matrix, which we use under D65 lighting

10. We then convert this to the sRGB' linear colour space, via this [matrix](https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB), before finally using the CsRGB conversion below it. If you use $pow(x, 2.2)$ then this is wrong

11. Then you display this sRGB data, hoping that your operating system isn't mad[^madness], and you have a decent colour calibrated monitor

[^madness]: Note that while I'm being a bit jokey, getting sRGB to display correctly has been a major stumbling block on linux, and in graphics APIs, and you do need to check that your graphics pipeline is doing what you think it is

This, as things go, is not all that straightforward. It would be a lot easier if the hips standard for asytrophysics included what the datatypes of the data was when downloading chunks, and would make this implementable

## Illustrative redshift

A much more likely situation is that we have some kind of texture, and we'd like to show off how red/blue shifted something is by altering the colour of the texture. This is a lot simpler to do than the full version, and we're going to use a simplified process:

1. Read our sRGB texture colour from where our ray hits

2. Calculate $z+1$ as we do in the physical test case

3. Calculate the intensity of our texture colour, by converting our sRGB colour to XYZ (see above), and using the Y component

4. Pick a very arbitrary frequency of visible light to represent our frequency information, eg 555 nanometers (in Hz)

5. Calculate the new frequency and intensity using this frequency

6. Map higher frequencies to bluer colours, and lower frequencies to darker colours

todo: finish this off

# Accretion disk?

This segment was up for debate. No black hole is really complete without an accretion disk, but as someone who is in this for the brutalist physical accuracy, I didn't want to simply put in a texture and call it a day without it having at least some physical basis, and this is one of the few things in this article series that I haven't written prior to writing this article, so I'm starting as fresh as you! In future articles we'll be simulating the accretion disk directly rather than using an analytic solution - which will be super fun[^fun]

[^fun]: My 29th birthday present to myself was implementing a new set of equations for simulating spacetime, which I didn't have a good excuse for using otherwise

## Thin disk

The archetypal accretion disk model is known as the [Novikov-Thorne model](https://www.its.caltech.edu/~kip/scripts/PubScans/II-48.pdf)

Accretion disk models are normally based a few assumptions, notably:

1. That the accretion disk has neglegible mass
2. That it is thin. You probably figured this one out

You may notice in renderings of an accretion disk, there's a gap between the inner most boundary of the accretion disk, and the black hole - this boundary is known as the innermost stable circular orbit (ISCO). This will crop up again in the future, but the orbit of any particle within the ISCO is unstable, and will eventually hit the black hole, whereas circular orbits outside this region are stable[^stable] to a degree

[^stable]: No orbit in general relativity is ever truly stable, as everything emits gravitational waves and orbits decay. Near a black hole, this effect is particularly intense

### ISCO equation

We can find the definition of ISCO for a kerr typed black hole via [2.21](https://articles.adsabs.harvard.edu/pdf/1972ApJ...178..347B) here, or [here](https://en.wikipedia.org/wiki/Innermost_stable_circular_orbit#Rotating_black_holes) for wikipedia

## Sigh

Lets look at the set of equations we're going to implement, from [here](https://arxiv.org/pdf/1110.6556). This segment is a pretty good introduction to the kind of stuff we have to deal with once we start implementing physical things

$$
\begin{align}
z &= r \cos\theta\\
a_* &= a/M \\
x_1 &= 2\cos(acos(a_*)/3 - pi/3) \\
x_2 &= 2\cos(acos(a_*)/3 + pi/3) \\
x_3 &= -2\cos(acos(a_*)/3) \\
x &= (r/M)^{1/2} \\
 \\
A &= 1 + a_*^2 x^{-4} + 2 a_*^2 x^{-6} \\
B &= 1 + a_* x^{-3} \\
C &= 1 - 3 x^{-2} + 2 a_*^2 x^{-3} \\
D &= 1 - 2x^{-2} + a_*^2 x^{-4} \\
E &= 1 + 4a_*^2 x^{-4} - 4a_*^2 x^{-6} + 3 a_*^4 x^{-8} \\
F &= 1 - 2 a_*^{-3} + a_*^2 x^{-4} \\
G &= 1 - 2x^{-2} + a_* x^{-3} \\
H &= 1 - 2x^{-2} + 2 a_* x^{-2} x_0^{-1} F_0^{-1} G_0 \\
I &= A - 2a_* x^{-6} x_0 F_0 G_0^{-1} \\
J &= O - x^{-2} J^{-1} (1 - a_* x_0^{-1} F_0^{-1} G_0 + a_*^2 x^{-2} H J^{-1} (1 + 3x^{-2} - 3 a_*{-1} x^{-2} x_0 F_0 G_0^{-1})) \\
K &= |AJ(1 - x^{-4} A^2 D^{-1} (x_0 F_0 G_0^{-1} O - 2 a_* x^{-2} A^{-1})^2)^{-1}| \\
O &= HJ^{-1} \\
Q &= B E^{-1/2} (1/x) (x - x_0 - (3/2) a_* ln(x/x_0) \\
                      &- (\frac{3(x_1 - a_*)^2}{x_1 (x_1 - x_2) (x_1 - x_3)}) ln(\frac{x - x_1}{x_0 - x_1}) \\
                      &- (\frac{3(x_2 - a_*)^2} {x_2 (x_2 - x_1) (x_2 - x_3)}) ln(\frac{x - x_2}{x_0 - x_2}) \\
                      &- (\frac{3(x_3 - a_*)^2} {x_3 (x_3 - x_1) (x_3 - x_2)}) ln(\frac{x - x_3}{x_0 - x_3})) \\
R &= F^2 C^{-1} - a_*^2 x^{-2} (G E^{-1/2} - 1) \\
S &= A^2 B^{-2} C D^{-1} R \\
V &= D^{-1} (1 + x^{-4} (a_*^2 - x_0^2 F_0^2 G_0^{-2}) + 2 x^{-6} (a_* - x_0 F_0 G_0^{-1}))\\
\\
\Phi &= Q + (0.02) (a^{9/8} M_*^{-3/8} \dot{M}_*^{1/4}) x^{-1} B C^{-1/2} (x_0^{9/8} C_0^{-5/8} G_0 V_0^{1/2})
\\
\frac{p^{gas}}{p^{rad}} &= (5*10^{-5})(a^{-1/4} M_*^{7/4} \dot{M}_*^{-2}) x^{21/4} A^{-5} B^{9/2} D S^{5/4} \Phi^{-2}\\
\end{align}
$$

What fun. If this looks horrendous to you: remember that this is literally just a large expression of basic operations, there's nothing fancy or tricky to evaluate here, and we don't have to do any swanky maths. We literally just need to type this all out correctly.  Do note that quantities with the subscript $_0$ are evaluated at the ISCO. Luckily, the paper we link mentions that the code is available [here](https://www.cfa.harvard.edu/%E2%88%BCrpenna/thindisk), which is very helpful for us to compare (this is a joke, its a broken link)

Now, these are the intermediate variables that define the disk, and we need to pick and choose the real quantities we want to calculate

1. Radial velocity in the local nonrotating frame $v^{\hat{r}}$, if we would like accurate redshift, or to generate faux particles for visual effects. Hang on, i think radial velocity is inwards velocity
2. Temperature $T$, for visually varying the colour. You could also assume a black body radiator, to determine frequency content
3. Radiant flux on the surface of the disk, for determining the brightness
4. Disk height $h$, where $H$ is the scaled disk height $h=H/r$. A very thin disk has $h$ -> $0$, which we will assume. Wait, h << 0 h is disk opening angle? Todo?
5. Density, if we want to do volumetric tracing. This is complicated, so I'm going to assume the disk is opaque, and not do this

The thin disk accretion model is split into 5 segments

1. The plunging region, within the ISCO
2. The edge region
3. The inner region
4. The middle region
5. The outer region

The definition of where these lie is determined by the quantity $\frac{p^{gas}}{p^{rad}}$. In the edge region, gas pressure > radiation pressure. In the inner region, gas pressure < radiation pressure. In the middle region gas pressure > radiation pressure, and unfortunately an outer region where gas pressure also > radiation pressure

So while we can use the gas ratios to distinguish between the middle three segments, we need an extra criteria for the outer region. In the outer region, opacity is dominated by a free-free (ff) term, and in the middle: opacity is dominated by electron scattering (es). We can therefore use the relations

$$
K_{ff} = (0.64 * 10^23) (\rho / (g/cm^3)) (\frac{T}{K})^{-7/2} cm^2/g\\
K_{es} = 0.4 cm^2/g
$$

To calculate if we're in the inner region, or the outer region

## Solving the equations

What we want, is A: Concrete radial values for each transition segment for a black hole, and B: Concrete functions for whatever physical values we're looking for in each segment. So the very first thing to do is to solve these equations, and spit out 5 values of $r$ that denote the transition boundaries. Once we do that, we're going to fit a basic interpolating polynomial through the data we fit, and then hey presto bobs your uncle

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












---
layout: post
title:  "General relativity for dummies: Rendering the schwarzschild black hole, in C++"
date:   2025-05-18 00:35:23 +0000
categories: C++
---

Black holes are pretty cool. Rendering them is a bit of a pain, and a lot of information is left to the dark arts. More than that, a significant proportion of information available on the internet is simply wrong, and most visualisations of black holes are significantly incorrect in one form or other. Today we're going to focus on two things

1. A very pragmatic understanding of general relativity. This is the journymans toolkit approach
2. Integrating the equations of motion, and basic initial conditions

Lets get into it

# Breaking down a simulation

There are 3 parts to any simulation

1. The initial conditions
2. The evolution equations
3. Termination

A lot of relativity articles tend to focus on the theory in isolation, so we're going to be focusing heavily on the entire pipeline of simulating this from start to end

Step 1 will be left until last, and is not the focus of this article. Step 2 requires us to solve something called the geodesic equation, and to be able understand it we'll need some mathematical background

# Mathematical background

## Tensor index notation

General relativity is particularly obtuse to get into, because it uses its own specific convention, which are not used that widely anywhere else. Its often ambiguous, and unclear, so lets spend a bit demystifying this first

There are two fundamental concepts we need to clear up first:

### Contravariance, and covariance

In everyday maths, a vector is just a vector. We express this as something like $v = 1x + 2y + 3z$, where $x$ $y$ and $z$ are our coordinate system basis vectors[^basisvectors]. When dealing with vectors, its not uncommon to index the vector's components by a variable/index:

$$ s = \sum_{k=0}^2 v_k $$

This is an example of how we'd express summing the components of a vector

General relativistic notation extends this further. Indices can be

1. Contravariant (raised):

$$ V^\mu $$

2. Covariant (lowered)

$$ V_\mu $$

Additionally, objects such as matrices can have more than one index, and the indices can have any "valence" (up/down-ness).

$ A^{\mu\nu} $,  $ A^\mu_{\;\;\nu} $, $ A_\mu^{\;\;\nu} $, and $ A_{\mu\nu} $ are all different representations of the same object. The first is the contravariant form, the middle two have mixed indices, and the last one is the covariant form, all of the tensor $A$

We can add more dimensions to our objects as well, eg: $ \Gamma^\mu_{\;\;\nu\sigma} $, more commonly written $ \Gamma^\mu_{\nu\sigma} $, and is a 4x4x4 object in this article. These objects are all referred to as "tensors", a term which has lost all mathematical meaning. In its strict definition, a tensor is an object that transforms in a particular fashion in a coordinate change: in practice, everyone calls everything a tensor, unless its relevant for it not to be

Here, we will refer to anything which takes an index as being a tensor, unless it is relevant. The other important class of objects are scalars, which are just values

One thing to note: Tensors and scalars are generally functions of the coordinate system, and vary from point to point in our spacetime. While we write $A_{\mu\nu}$, what we really *mean* is $A_{\mu\nu}(x, y, z, w)$ - the coordinates are just left implicit

### Changing the valence of an index: raising and lowering indices

The most important object in general relativity, that we'll introduce here is the metric tensor. The metric tensor is a 4x4 symmetric matrix, which is normally spelt $ g_{\mu\nu} $. Because it is symmetric, $ g_{\mu\nu} =  g_{\nu\mu} $, and has 10 independent components. This is the covariant form of the metric tensor. The metric tensor is also often thought of as a function taking two vector arguments, $g(u,v)$, and performs the same function as the euclidian dot product. That is to say, where in 3 dimensions you might say $a = dot(v, u)$, in general relativity you might say $a = g(v, u)$

For the metric tensor, and the metric tensor *only*, the contravariant form of the metric tensor is calculated as such:

$$ g^{\mu\nu} = (g_{\mu\nu})^{-1} $$

That is to say, the regular 4x4 matrix inverse of treating $g$ as a matrix. This is generally never true of any other object, and $  A^{\mu\nu} = (A_{\mu\nu})^{-1} $ is likely a mistake

The metric tensor is responsible for many things, and general relativity is in part the study of this fundamental object. To raise an index, we do it as such:

$$ v^\mu = g^{\mu\nu} v_\nu $$


And lowering an index is performed like this

$$ v_\mu = g_{\mu\nu} v^\nu $$

I've introduced some new syntax here known as the einstein summation convention, so lets go through it. In general, any repeated index in an expression is summed:

$$ g_{\mu\nu} v^\nu == \sum_{\nu=0}^3 g_{\mu\nu} * v^\nu $$

with $*$ being normal scalar multiplication, which is always unwritten. In code, this looks like this, which may be clearer:

```c++
tensor<float, 4> lower_index(const tensor<float, 4, 4>& metric, const tensor<float, 4>& v) {
    tensor<float, 4> result = {};

    for(int mu = 0; mu < 4; mu++) {
        float sum = 0;
        for(int nu = 0; nu < 4; nu++) {
            sum += metric[mu, nu] * v[nu];
        }
        result[mu] = sum;
    }

    return result;
}
```

A more complicated expression looks like this:

$$ \Gamma^\mu_{\nu\sigma} v^\nu u^\sigma == \sum_{\nu=0}^3 \sum_{\sigma=0}^3 \Gamma^\mu_{\nu\sigma} v^\nu u^\sigma $$

```c++
tensor<float, 4> example(const tensor<value, 4, 4, 4>& christoff2, const tensor<value, 4>& v, const tensor<value, 4>) {
    tensor<float, 4> result = {};

    for(int mu = 0; mu < 4; mu++) {
        float sum = 0;
        for(int nu=0; nu < 4; nu++) {
            for(int sigma = 0; sigma < 4; sigma++) {
                sum += christoff2[mu, nu, sigma] * v[nu] * u[sigma];
            }
        }
        result[mu] = sum;
    }

    return result;
}
```

Here, $\nu$ and $\sigma$ are 'dummy' indices (ie they are repeated), and $\mu$ is a 'free' index (not repeated). The size of the resulting tensor is equal to the number of free indices. One more key rule is that only indices of opposite valence sum: eg an up index *always* sums with a down index, and you do not sum two indices of the same valence - though this almost never crops up

See this [^indices] footnote for more examples

The last thing we need to learn now is how to raise and lower the indices of a multidimensional object, eg $ A^{\mu\nu} $. As a rule (which seems a bit arbitrary), we set the dummy index to the index we wish to raise and lower, and set the free index to the new relabeled index. Eg to lower the first index, we sum over it, and set the second index of the metric tensor to our new index name

$$ \Gamma_{\mu\nu\sigma} = g_{\mu\gamma} \Gamma^{\gamma}_{\;\;\nu\sigma} $$

Remember that the metric tensor is always symmetric, so we could also write this

$$ \Gamma_{\mu\nu\sigma} = g_{\gamma\mu} \Gamma^{\gamma}_{\;\;\nu\sigma} $$

Here's another example:

$$\Gamma^{\mu\nu}_{\;\;\;\sigma} = g^{\nu\gamma} \Gamma^\mu_{\;\;\gamma\sigma}$$

More examples are provided here [^indices2]

At the top, we considered the metric tensor as a function taking two arguments, $g(u,v)$. One helpful way to look at covariant and contravariant indices is as partial function applications of the metric tensor:

$$ u_\mu v^\mu == u^\mu v_\mu == g_{\mu\nu} u^\nu v^\mu == g^{\mu\nu} u_\nu v_\mu == g(u, v) == g(u, \cdot) \circ v $$

That is to say, we can treat applying the metric tensor to $v^\mu$ to get $v_\mu$ like currying

If like me you've hit this point and are feeling a bit tired, don't worry. These are rules we can refer back to whenever we need them, and the footnotes will contain lots of examples. Here's a picture of my cat for making it this far:

## Geodesics: the basics

What we're trying to accomplish here in this article is basic raytracing. In a normal, 3d raytracer, we construct a ray with a start position $x^i$, and a velocity $v^i == dx^i/dt$, where dt is in general, time. If what we're raytracing doesn't involve anything funky, rays travel in straight lines, and its universal to simply consider that rays have a *direction* instead of a velocity. When this is true, $dt$ (the 'parameterisation') becomes pretty arbitrary, and we no longer care about the exact magnitude of $v^\mu$

To find a point on a ray when rays move in straight lines, we can simply do $x_2^\mu = x^\mu + v^\mu * dt$, where $dt$ may either be time, or just an arbitrary length

Lets imagine that our rays do not move in straight lines, and that at every point in space, an acceleration $a^\mu$ is applied to our ray. To calculate $a^\mu$, we may need to know the values of both $x^\mu$ and $v^\mu$ at each point

To find the points on our ray in the general case, we must therefore integrate. That is to say, we need to solve the following equations for $x$

$$dx^\mu/dt = v^\mu \\
dv^\mu/dt = a^\mu\\
$$

Where $a^\mu$ gives our acceleration. In newtonian dynamics, $a^\mu$ might be the acceleration given by all the other bodies in our simulation applying a force to us (which we will label with the index ${s}$):

$$a^\mu = (-G \sum_{k=0, k != s}^n \frac{m_k}{|x_s - x_k|^3} (x_s - x_k))^\mu$$

https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation#Vector_form

And in straight line raytracing, $a^\mu=0$, allowing us to directly integrate the equations. In general relativity:

$$a^\mu = -\Gamma^\mu_{\alpha\beta} v^\alpha v^\beta$$

Where $\Gamma^\mu_{\alpha\beta}$ is derived from the metric tensor, and is implicitly a function of $x^\mu$. This is called the geodesic equation, and describes the motion of GR's notion of rays, which are called geodesics

In general relativity, the motion of all objects, lightrays, and additionally measuring distances are described by geodesics. On earth, if you move in a straight line you'll end up going in a circle: geodesics are the more mathematical version of this concept. The only physically accurate way to get from point A, to point B, or relate them in any fashion in the general case, is via a geodesic

There are three kinds of geodesics:

1. Spacelike
2. Timelike
3. Lightlike

Timelike geodesics represent the path that objects with rest mass follow, and lightlike geodesics represent the path that objects without *rest* mass follow (ie light). We will be ignoring spacelike geodesics entirely, but they represent spacetime distances that can't be travelled by an observer or light, and are causally disconnected from us. Note that it does not matter what the mass of an object is, only that it has mass at rest[^massissortofignored]

We'd like to render our black hole by firing light rays around and finding out what our spacetime looks like, so what we're looking for in this article is lightlike geodesics

## Geodesics: the less basics

A geodesic has two properties: a position $x^\mu$, and a velocity. In normal every day life, velocity is defined as the rate of change of position with respect to time. In general relativity, we have several concepts of time that we could use

1. The concept of time given to us by our coordinate system, which is completely arbitrary and has no meaning
2. The concept of time as experienced by an observer (including particles)
3. A even more completely arbitrary concept of time that has no meaning

No observer can move at the speed of light, so 2. is right out for lightlike geodesics, though works well for timelike geodesics. 1. Is dependent on our coordinate system and is hard to apply generally (not every coordinate system has a time coordinate), so in general we will always be using 3 for light.

This makes our velocity: $v^\mu = dx^\mu/ds$. $ds$ is known as an affine parameter, and represents a fairly arbitrary parameterisation[^whatsaparameterisation] of our curve/geodesic

One other fun fact is that the velocity of a geodesic is always tangent to the curve that that geodesic traces out. This defines what it means for something to be a geodesic, that it is in a sense parallel to itself. In fact, transporting the velocity of a geodesic along the curve defined by the geodesic, is one way to derive the geodesic equation (autoparallel transport)

## Integrating the geodesic equation

We're finally going to solve one of our major components now: We've already briefly seen the geodesic equation, which looks like this:

$$ a^\mu = -\Gamma^\mu_{\alpha\beta} v^\alpha v^\beta $$

Where more formally, our acceleration $ a^\mu == \frac{d^2x^\mu}{ds^2} $, and

$$ \Gamma^\mu_{\alpha\beta} = \frac{1}{2} g^{\mu\sigma} (g_{\sigma\alpha,\beta} + g_{\sigma\beta,\alpha} - g_{\alpha\beta,\sigma}) $$

Note that:

$$ g_{\mu\nu,\sigma} == \partial_\sigma g_{\mu\nu} $$

Aka, taking the partial derivatives in the direction $\sigma$, as defined by our coordinate system. This equation is likely to stretch our earlier understanding of how to sum things, so lets write this out manually:

$$ \Gamma^\mu_{\alpha\beta} = \frac{1}{2} \sum_{\sigma=0}^3 g^{\mu\sigma} (\partial_\beta g_{\sigma\alpha} + \partial_\alpha g_{\sigma\beta} - \partial_\sigma g_{\alpha\beta})$$

or

```c++

tensor<float, 4, 4, 4> Gamma;

for(int mu = 0; mu < 4, mu++)
{
    for(int al = 0; al < 4; al++)
    {
        for(int be = 0; be < 4; be++)
        {
            float sum = 0;

            for(int sigma = 0; sigma < 4; sigma++)
            {
                sum += metric_inverse[mu, sigma] * (diff(metric[sigma, al], be) + diff(metric[sigma, be], al) - diff(metric[al, be], sigma));
            }

            Gamma[mu, al, be] = sum;
        }
    }
}
```

Phew. This loop may look horrendously inefficient, but there are some symmetries here that reduce the computational complexity, notably that $\Gamma^\mu_{\alpha\beta} = \Gamma^\mu_{\beta\alpha}$. Additionally, in many metrics most of $\Gamma$'s components are 0, which helps too

So: to integrate a geodesic, we start off with an appropriate position, $x^\mu$, an appropriate velocity, $v^\mu$, and calculate our acceleration via the geodesic equation. We then integrate this. During the process of this, we'll need a metric tensor $g_{\mu\nu}$ and the partial derivatives of it

We're getting close to being able to integrate our equations now. We just need three more things:

1. A real metric tensor
2. An initial position $x^\mu$, which will be our camera position
3. An initial velocity, which is a lot more complex

## A real metric tensor

There are many different black holes, and many different ways of representing each of them. Today we're going to pick the simplest kind: the schwarzschild black hole

A metric tensor fundamentally defines the curvature of spacetime - and it is the central object of general relativity. It also generally implicitly defines a coordinate system associated with it. The metric tensor is often expressed in a form called the "line element", which reads like this:

$$ ds^2 = -d\tau^2 = -(1-\frac{r_s}{r}) dt^2 + (1-\frac{r_s}{r})^{-1} dr^2 + r^2 d\Omega^2 $$

This is the wikipedia definition[^wikipedia] [^signconventions], where $d\Omega^2 = d\theta^2 + sin^2(\theta) d\phi^2$, and $r_s$ is the schwarzschild radius - that is twice the 'mass'[^blackholesdonthavemass] $M$ in geometric units of $c=g=1$

This isn't very clear, so lets examine it. The $d$ terms on the right hand side (eg $dt$) represent infinitesimals. $ds^2$ represents the square of the length of this curve - note that $ds^2$ can be 0 or negative, and is called the spacetime interval. When your displacement $(dt, dr, dtheta, dphi)$ is timelike, $ds^2$ is also the (negative) proper time squared ($d\tau^2$). The fact that this $ds^2$ and the $ds$ we picked for our parameterisation are the same is not a total coincidence - for timelike curves, we pick $ds_{parameterisation} = d\tau$. In reality, $ds$ refers to the general concept of arc length, the length of a curve, which is why the notation is re-used

The sign[^signconventions] of $ds^2$ defines what kind of geodesic we have. If we plug a velocity $(dt, dr, d\theta, d\phi)^\mu = v^\mu$ in here, and calculate the metric at our position $(t, r, \theta, \phi)^\mu = x^\mu$, then:

1. If $ds^2 > 0$, our curve is spacelike
2. If $ds^2 == 0$, our curve is lightlike
3. If $ds^2 < 0$, our curve is timelike

note that when a curve is lightlike, no proper time $d\tau$ ever elapses along our curve, as $d\tau = 0$

The line element can also be thought of as an expanded out form of when you apply your metric tensor to an infinitesimal displacement, $g(du, du)$. When $du = (dt, dr, d\theta, d\phi)$, we recover our line element

This means that we can read the matrix $g_{\mu\nu}$ directly off from the line element. If we have the line element

$ds^2 = k_1 da^2 + k_2 dadb + k_3 db^2$ etc, we get the metric tensor:

|.|a|b|
|-|-|-|
|a|$k_1$|$\frac{1}{2}k_2$|
|b|$\frac{1}{2}k_2$| $k_3$|

Note that all offdiagonal terms are multiplied by 1/2. The schwarzschild metric is diagonal, so we get

|.|t|r|$\theta$|$\phi$|
|-|-|-|-|-|
|t|$-(1-\frac{r_s}{r})$|0|0|0|
|r|0|$(1-\frac{r_s}{r})^{-1}$|0|0
|$\theta$|0|0|$r^2$|0|
|$\phi$|0|0|0|$r^2 sin^2(\theta)$|

As a 4x4 matrix. Note that this matrix is a function of the coordinate system, and it must be recalculated at a specific point in space where you want to apply it. If we want to raise or lower the velocity of our geodesic, we must calculate the metric tensor *at* the position where the velocity vector is

Vectors, and tensorial objects, and scalar functions are generally associated with a point in spacetime, which is their origin in a sense, more formally they are tangent vectors - tangent to the 'manifold' that is spacetime. Their origin is where you must calculate the metric tensor (and other tensors) to be able to do operations on them

## Numerical differentiation vs automatic differentiation

Now that we have our metric tensor, we need to take its derivatives. There are two easy way to calculate the derivatives: numerical differentiation, and automatic differentiation. A recent post I made recently covered AD, and I would recommend using it. However here, we will approximately define the derivatives for space savings. Remember that the metric tensor is calculated at a coordinate, and is a function of the coordinate system. Lets make this explicit, by defining $g_{\mu\nu}$ as $g_{\mu\nu}(t, r, \theta, \phi)$

$$g_{\mu\nu,t} = \partial_t g_{\mu\nu} == \frac{(g_{\mu\nu}(t + h, r, \theta, \phi) - g_{\mu\nu}(t - h, r, \theta, \phi))}{2h} $$

This is a simple centered derivative to calculate the derivative in the first coordinate direction, often called $\partial_0$, which is equivalent to $\partial_t$ in schwarzschild

# A brief pause to review

So far we have:

1. Defined the paths that lightrays take as likelight geodesics. These have a position $x^\mu$, and a velocity $v^\mu$. We know these are geodesics where $g_{\mu\nu} v^\mu v^\nu = 0$

2. Found out how to read a metric tensor $g_{\mu\nu}$ from a line element

3. Understood how to plug numbers into the geodesic equation, to get our acceleration

It might surprise you to learn that this is the simple part of what we're trying to do, in general. For the purposes of trying to keep things terse today, we're going to use some pre-baked initial conditions, instead of calculating them ourselves

# Initial conditions

Initial conditions in this corner of general relativity are not a good, fun time. This is where we get into the less well understood corners of general relativity, and where mistakes tend to be made

This article is not the correct one for a long discussion of initial conditions - that'll be next time, likely when you all conveniently go on holiday

## What are we trying to get out of our initial conditions?

In this phase, what we're trying to do is construct an initial direction $v\mu$ that our lightray travels towards - a geodesic velocity. In a regular, flat, 3d simulation, its very easy - we define a plane in front of our camera, and construct a ray, from the camera's origin, through a pixel on that plane. If a pixel has a position $p={x-width/2, y-height/2, d}$ on that plane, then the ray's direction in 3d space is $d=norm(p-o)$

The question then becomes: how do we translate our ray direction $d$ in 3d space, to a valid geodesic velocity in 4d spacetime? The answer is: tetrads

# Tetrads, in short

https://www2.mpia-hd.mpg.de/homes/tmueller/pdfs/catalogue_2014-05-21.pdf

Tetrads, also known as frame fields, or vielbein, are a field of four 4-vectors, that are orthogonal to each other. These make up the "frame of reference" of an observer, and are used to translate between what one observer sees and experiences, and the wider universe that we're simulating

General relativity demands that spacetime is quite literally locally flat from the perspective of any observer. And yet, we can observe that spacetime is clearly curved - planets go round the sun, and black holes exist. Translating from an observers locally flat spacetime, to that curved spacetime, is done via the tetrads. It fundamentally relates the view of an observer, to the wider universe - or to another observer

Each tetrad is labelled with a latin index $i$, as such: $e_i$. Each one of these tetrads $e_i$ has 4 components, and so they are spelt $e^\mu_i$ in the literature

General relativity demands that spacetime is locally flat from the perspective of an observer, no matter where they are or what they're doing. The technical definition of locally flat is the minkowski metric, $\eta_{\mu\nu}$, which is always in cartesian coordinates, no matter the coordinate system of our metric tensor $g_{\mu\nu}$:

| |t|x|y|z
|-|-|-|-|-
|t|-1|0|0|0
|x|0|1|0|0
|y|0|0|1|0
|z|0|0|0|1

Say we have a quantity like a light ray, which we define as such:

$$v_{flat}^\mu = (1, d_0, d_1, d_2)$$

Where $|d| = 1$. Note that $\eta_{\mu\nu} v^\mu_{flat} v^\nu_{flat} = 0$, which makes this a lightlike geodesic in our minkowski spacetime, as we discussed earlier[^typesofgeodesics]

Each tetrad defines a series of basis vectors which we can use to transform from our minkowski spacetime, to our curved spacetime, as follows

$$v^\mu_{curved} = e^\mu_i v^i_{flat}$$

If the direction $d$ points through a pixel in our local plane, we now have a way to construct the initial velocity of our geodesic. One key thing to note is that we almost always trace lightlike geodesics *backwards* in time, which we can accomplish by negating the time component of our lightray, and getting $v_{flat}^\mu = (-1, d_0, d_1, d_2)$

## That's all well and good, but how do I calculate my tetrads?

Luckily, a lot of metric tensors have precalculated tetrads, and we can simply read them off of this page here https://arxiv.org/pdf/0904.4184. For schwarszchild:

$e^0_0 = 1/sqrt(1-rs/r) \\$
$e^1_1 = sqrt(1-rs/r) \\$
$e^2_2 = 1/r \\$
$e^3_3 = 1/(r sin(\theta))$

Note that these are not unique, and represent a particular kind of observer in this spacetime. While there *is* a unique natural choice, it has no special meaning, and we'll get to this soon. Also note that this paper refers to the upper indices of the tetrads by their coordinate basis, ie $\partial_t$ means the 0th component of the tetrad $e_t$, which is $e_0$ for us. A vector may in general be written $a \partial_t + b\partial_x + c\partial_y + d\partial_z$ assuming a coordinate system $(t, x, y, z)$, and the paper linked above follows this convention for specifying the tetrad components

# The complete procedure

Step 1: We calculate our metric tensor $g_{\mu\nu}$ at our starting coordinate, our camera position. It is a 4x4 matrix, defined generally by a line element. Our cameras position is in schwarzschild coordinates: t, r, $\theta$, $\phi$, and this is the origin of our geodesics

Step 2: We calculate our tetrad basis vectors, $e^\mu_i$

Step 3: We then construct an initial geodesic velocity in locally flat spacetime. To do this, we pick a cartesian direction $v^k$ with $|v|$ = 1, the direction through a pixel on our screen, where the camera is at the origin

Step 4: We then use this tetrad to construct a geodesic velocity in our curved spacetime, by doing $v_{curved}^\mu = e^\mu_i v^i$, and flip the sign of $e_0$ for light rays so that we trace backwards in time

Step 5: Once we have a position, and velocity, we plug these into the geodesic equation, and integrate

Step 6: Once we have run a certain number of iterations, or our coordinate radius r > some constant, or r < the event horizon, we terminate the simulation and render

Step 7: Then we go outside and talk to our friends and family, who are probably getting worried

# This is actually simpler in code

## Step 1: The metric tensor

First up, we need a tensor type. Libraries like eigen allow you to implement this kind of thing easily. For the code in this article, I'm going to assume you're using a vector library that supports N dimensional matrices. This article comes with an accompanying implementation, which will implement this by hand

To calculate the metric tensor, we grab it in matrix form, reading off the components of the line element

```c++
tensor<float, 4, 4> schwarzschild_metric(const tensor<float, 4>& position) {
    float rs = 1;

    float r = position[1];
    float theta = position[2];

    return {-(1-rs/r), 0,          0,   0,
            0,         1/(1-rs/r), 0,   0,
            0          0,          r*r, 0,
            0          0, 0,            r*r * sin(theta)*sin(theta)};
}
```

## Step 2: Calculate our tetrads

```c++

struct tetrad
{
    std::array<tensor<float, 4>, 4> v;
};

tetrad calculate_schwarzschild_tetrad(const tensor<float, 4>& position) {
    float rs = 1;
    float r = position[1];
    float theta = position[2];

    tensor<float, 4> et = {1/sqrt(1 - rs/r), 0, 0, 0};
    tensor<float, 4> er = {0, sqrt(1 - rs/r), 0, 0};
    tensor<float, 4> etheta = {0, 0, 1/r, 0};
    tensor<float, 4> ephi = {0, 0, 0, 1/(r * sin(theta))};

    return {et, er, etha, ephi};
}
```

## Step 3: calculating our pixel direciton in flat spacetime

```c++
tensor<float, 3> get_ray_through_pixel(int sx, int sy, int screen_width, int screen_height, float fov_degrees) {
    float fov_rad = (fov_degrees / 360.f) * 2 * M_PI;
    float f_stop = (width/2) / tan(fov_rad/2);

    tensor<float, 3> pixel_direction = {cx - width/2, cy - height/2, f_stop};
    //pixel_direction = rot_quat(pixel_direction, camera_quat); //if you have quaternions, or some rotation library, rotate your pixel direction here by your cameras rotation

    return pixel_direction.norm();
}
```

## Step 4: Get our initial geodesic, by constructing its velocity from the tetrads

```c++
struct geodesic
{
    tensor<float, 4> position;
    tensor<float, 4> velocity;
};

geodesic make_lightlike_geodesic(const tensor<float, 4>& position, const tensor<float, 3>& direction, const tetrad& tetrads) {
    float fov_rad = (fov_degrees / 360.f) * 2 * M_PI;
    float f_stop = (width/2) / tan(fov_rad/2);

    geodesic g;
    g.position = position;
    g.velocity = tetrads.v[0] * -1 //Flipped time component, we're tracing backwards in time
               + tetrads.v[1] * pixel_direction[0]
               + tetrads.v[2] * pixel_direction[1]
               + tetrads.v[3] * pixel_direction[2];

    return g;
}
```

## Step 5: Integrate the geodesic equation

```c++
//function to numerically differentiate an arbitrary function that takes a position, and a direction
auto diff(auto&& func, const tensor<float, 4>& position, int direction) {
    auto p_up = position;
    auto p_lo = position;

    float h = 0.001f;

    p_up[direction] += h;
    p_lo[direction] += h;

    return (func(p_up) - func(p_lo)) / (2 * h);
}

tensor<float, 4, 4> calculate_christoff2(const tensor<float, 4>& position, auto&& get_metric) {
    tensor<float, 4, 4> metric = get_metric(position);
    tensor<float, 4, 4> metric_inverse = metric.invert();
    tensor<float, 4, 4, 4> metric_diff; ///uses the index signature, diGjk

    for(int i=0; i < 4; i++) {
        tensor<float, 4, 4> differentiated = diff(get_metric, i);

        for(int j=0; j < 4; j++) {
            for(int k=0; k < 4; k++) {
                metric_diff[i, j, k] = differentiated[j, k];
            }
        }
    }

    tensor<float, 4, 4, 4> Gamma;

    for(int mu = 0; mu < 4, mu++)
    {
        for(int al = 0; al < 4; al++)
        {
            for(int be = 0; be < 4; be++)
            {
                float sum = 0;

                for(int sigma = 0; sigma < 4; sigma++)
                {
                    sum += metric_inverse[mu, sigma] * (metric_diff[be, sigma, al] + metric_diff[al, sigma, be] - metric_diff[sigma, al, be]);
                }

                Gamma[mu, al, be] = sum;
            }
        }
    }

    //note that for simplicities sake, we fully calculate all the christoffel symbol components
    //but the lower two indices are symmetric, and can be mirrored to save significant calculations
    return Gamma;
}

tensor<float, 4> calculate_acceleration_of(const tensor<float, 4>& X, const tensor<float, 4>& v, auto&& get_metric) {
    tensor<float, 4, 4, 4> christoff2 = calculate_christoff2(X, get_metric);

    tensor<float, 4> acceleration;

    for(int mu = 0; mu < 4; mu++) {
        float sum = 0;

        for(int al = 0; al < 4; al++) {
            for(int be = 0; be < 4; be++) {
                sum += -christoff2[mu, al, be] * v[al] * v[be]
            }
        }

        acceleration[mu] = sum;
    }

    return acceleration;
}

tensor<float, 4> calculate_schwarzschild_acceleration(const tensor<float, 4>& X, const tensor<float, 4>& v) {
    return calculate_acceleration_of(X, v, schwarzschild_metric);
}

struct integration_result {
    enum hit_type {
        ESCAPED,
        EVENT_HORIZON,
        UNFINISHED
    };

    hit_type type = UNFINISHED;
    geodesic g;
}

integration_result integrate(geodesic& g) {
    integration_result result;

    float dt = 0.1f;
    float rs = 1;
    float start_time = g.position[0];

    for(int i=0; i < 1024; i++)
        tensor<float, 4> acceleration = calculate_schwarzschild_acceleration(g.position, g.velocity);

        g.velocity += acceleration * dt;
        g.position += g.velocity * dt;

        float radius = g.position[1];

        if(radius > 100) {
            //ray escaped
            result.g = g;
            result.type = integration_result::ESCAPED;

            return result;
        }

        if(radius <= rs + 0.0001f || g.position[0] > start_time + 1000) {
            //ray has very likely hit the event horizon
            result.g = g;
            result.type = integration_result::EVENT_HORIZON;

            return result;
        }
    }

    result.g = g;
    return result;
}
```



### Spacelike, and timelike coordinates

## The 'default' tetrad

Lets calculate *a*  tetrad

## Arbitrary tetrads

Lets calculate a specific-ish tetrad

## Parallel transport

Oh dear

# Evolution Equations

## Integrating the equations

Verlet integration. Hamiltonians

### Coordinate time

# Finishing up

Redshift

The lorentz factor

# Hang on, where's the black hole?

## Metric tensors
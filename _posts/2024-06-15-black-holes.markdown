---
layout: post
title:  "Rendering the schwarzschild black hole, in C++/OpenCL"
date:   2025-05-18 00:35:23 +0000
categories: C++
---

Black holes are pretty cool. Rendering them is a bit of a pain, and a lot of information is left to the dark arts. More than that, a significant proportion of information available on the internet is simply wrong, and most visualisations of black holes are significantly incorrect in one form or other. This is going to be a comprehensive guide to rendering the schwarzschild black hole, for people who's maths stops at a basic understanding of what differentiation is

There are 3 parts to any simulation

1. The initial conditions
2. The evolution equations
3. Stopping evolution

A lot of relativity articles tend to focus on the theory in isolation, so we're going to be focusing heavily on the entire pipeline of simulating this from start to end

# Tensor index notation

General relativity is particularly obtuse to get into, because it uses its own specific convention for arithmetic, which is not used that widely anywhere else. Its often ambiguous, and unclear, so lets spend a bit demystifying this first

There are two fundamental concepts we need to clear up first:

$$ 2^{\frac{n-1}{3}} $$

$$ \sum_{k=0}^{3} 2^{-k} = 1 $$

$$ \frac{1}{2} $$

$$ \Gamma^i_{kl} $$

## Contravariance, and covariance

In everyday maths, a vector is just a vector. We express this as something like $v = 1x + 2y + 3z$. When dealing with vectors, its not uncommon to index the vector's components by a variable/index:

$$ s = \sum_{k=0}^2 v_k $$

This is an example of how we'd express summing the components of a vector

General relativistic notation extends this further. Indices can be

Contravariant (raised):

$$ V^\mu $$

Or covariant (lowered)

$$ V_\mu $$

Additionally, objects such as matrices can have more than one index, and the indices can have any valence (up/down-ness)

$ A^{\mu\nu} $,  $ A^\mu_{\;\;\nu} $, $ A_\mu^{\;\;\nu} $, and $ A_{\mu\nu} $ are all different representations of the same object

We can add more dimensions to our objects as well, eg: $ \Gamma^\mu_{\;\;\nu\sigma} $, more commonly written $ \Gamma^\mu_{\nu\sigma} $

and is a 4x4x4 object. These objects are all referred to as `tensors`, a term which has lost all mathematical meaning. In its strict definition, a tensor is an object that transforms in a particular fashion in a coordinate change: in practice, everyone calls everything a tensor, unless its relevant for it not to be

## Changing the valence of an index: raising and lowering indices

Next up is the most important object, that we'll introduce here: the metric tensor. The metric tensor is generally a 4x4 symmetric matrix, which is normally spelt $ g_{\mu\nu} $. Because it is symmetric, $ g_{\mu\nu} =  g_{\nu\mu} $. This is the covariant form of the metric tensor. The metric tensor is also often thought of as a function taking two arguments, $g(u,v)$, and performs the same function as the euclidian dot product

For the metric tensor, and the metric tensor *only*, the contravariant form of the metric tensor is calculated as such:

$ g^{\mu\nu} = (g_{\mu\nu})^{-1} $

That is to say, the regular 4x4 matrix inverse of treating $g$ as a matrix. This is never true of any other object, and

$  A^{\mu\nu} = (A_{\mu\nu})^{-1} $

Is incorrect

The metric tensor is responsible for many things, and general relativity is in part the study of this fundamental object. To lower an index, we do it as such:

$$ v_\mu = g_{\mu\nu} v^\nu $$

And raising an index is performed like this

$$ v^\mu = g^{\mu\nu} v_\nu $$

I've introduced some new syntax here known as the einstein summation convention, so lets go through it quickly. In general, any repeated index in an expression is summed:

$$ g^{\mu\nu} v_\nu == \sum_{\nu=0}^3 g^{\mu\nu} * v_\nu $$

with $*$ being normal scalar multiplication. A more complicated expression looks like this:

$$ \Gamma^\mu_{\nu\sigma} v^\nu v^\sigma == \sum_{\nu=0}^3 \sum_{\sigma=0}^3 \Gamma^\mu_{\nu\sigma} v^\nu v^\sigma $$

Here, $\nu$ and $\sigma$ are 'dummy' indices, and $\mu$ is a 'free' index. One more key rule is that only indices of opposite valence sum: eg an up index *always* sums with a down index

See this [^indices] footnote for more examples:

I promise we'll get to some code soon. The last thing we need to learn now is how to raise and lower the indices of a multidimensional object, eg $ A^{\mu\nu} $. As a rule (which seems a bit arbitrary), we set the dummy index to the index we wish to raise and lower, and set the free index to the new relabeled index. Eg to lower the first index, we sum over it, and set the second index of the metric tensor to our new index name

$$ \Gamma_{\mu\nu\sigma} = g_{\mu\gamma} \Gamma^{\gamma}_{\;\;\nu\sigma} $$

Remember that the metric tensor is always symmetric, so we could also write this

$$ \Gamma_{\mu\nu\sigma} = g_{\gamma\mu} \Gamma^{\gamma}_{\;\;\nu\sigma} $$

At the top, we considered the metric tensor as a function taking two arguments, $g(u,v)$. One helpful way to look at covariant and contravariant indices is as partial function applications of the metric tensor:

$$ u_\mu v^\mu == u^\mu v_\mu == g_{\mu\nu} u^\nu v^\mu == g^{\mu\nu} u_\nu v_\mu == g(u, v) == g(u, \cdot) \circ v $$

Put more obtusely:

``` math
f(\cdot) = g(u, \cdot) \\

f(\cdot) == u_\mu \\
u_\mu v^\mu == f(v) == g(u, v)
```

If like me you've hit this point and are feeling a bit tired, don't worry. These are rules we can refer back to whenever we need them, and the footnotes will contain lots of examples. Here's a picture of my cat for making it this far:

## I was promised simulating black holes

Lets take a look at our evolution equations, before we get to the initial conditions. In general relativity, all objects move along geodesics. On earth, if you move in a straight line you'll end up going in a circle: geodesics are the more mathematical version of this concept. In general relativity, all objects move along geodesics

There are three kinds of geodesics:

1. Spacelike
2. Timelike
3. Lightlike

Timelike geodesics represent the path that objects with rest mass follow, and lightlike geodesics represent the path that objects without *rest* mass follow (ie light). We will be ignoring spacelike geodesics entirely, but represent spacetime distances that can't be travelled, and are casually disconnected from us. Note that it does not matter what the mass of an object is, only that it has mass at rest[^massissortofignored]

We'd like to render our black hole by firing light rays around, so what we're looking for is lightlike geodesics

## Geodesics

A geodesic has two properties: a position $x^\mu$, and a velocity. In normal every day life, velocity is defined as the rate of change of position with respect to time. In general relativity, we have several concepts of time that we could use

1. The concept of time given to us by our coordinate system, which is completely arbitrary and has no meaning
2. The concept of time as experienced by an observer
3. A even more completely arbitrary concept of time that has no meaning

No observer can move at the speed of light, so 2. is right out for lightlike geodesics. 1. Is dependent on our coordinate system and is hard to apply generally (not every coordinate system has a time coordinate), so in general we will be using 3.

This makes our velocity: $dx^\mu/ds$. $ds$ is known as an affine parameter, and represents a fairly arbitrary parameterisation of our curve/geodesic

One other fun fact is that the velocity of a geodesic is always tangent to curve, defined as the union of all our positions

## This still isn't a black hole

We're finally going to solve one of our major components now: The evolution equations. This is the equation of motion for a geodesic, called the geodesic equation:

$$ a^\mu = -\Gamma^\mu_{\alpha\beta} v^\alpha v^\beta $$

Where $ a^\mu == \frac{d^2x^\mu}{ds^2} $ and represents our acceleration

$$ \Gamma^\mu_{\alpha\beta} = \frac{1}{2} g^{\mu\sigma} (g_{\sigma\alpha,\beta} + g_{\sigma\beta,\alpha} - g_{\alpha\beta,\sigma}) $$

Note that:

$$ g_{\mu\nu,\sigma} == \partial_\sigma g_{\mu\nu} $$

Aka, taking the partial derivatives in the direction $\sigma$, as defined by our coordinate system. This equation is likely to stretch our earlier understanding of how to sum things, so lets write this out manually:

$$ \frac{1}{2} \sum_{\sigma=0}^3 g^{\mu\sigma} (\partial_\beta g_{\sigma\alpha} + \partial_\alpha g_{\sigma\beta} - \partial_\sigma g_{\alpha\beta})$$

Phew. To trace a geodesic forwards in time, we start off with an appropriate position, $x^\mu$, an appropriate velocity, $v^\mu$, and calculate our acceleration via the geodesic equation. We then integrate this - and I would recommend verlet integration (though euler works pretty ok too here). During the process of this, we'll need a metric tensor $g_{\mu\nu}$ and the partial derivatives of it, which is a bit of a pain

///geodesic tracing code, with a plugin in metric tensor

## A real metric tensor

There are many different black holes, and many different ways of representing each of them. Today we're going to pick the simplest kind: the schwarzschild black hole

A metric tensor fundamentally defines the curvature of spacetime - and it is the central object of general relativity. It also generally implicitly defines a coordinate system associated with it. The metric tensor is often expressed in a form called the "line element", which reads like this:

$$ ds^2 = -d\tau^2 = -(1-\frac{r_s}{r}) dt^2 + (1-\frac{r_s}{r})^{-1} dr^2 + r^2 d\Omega^2 $$

This is the wikipedia definition[^wikipedia] [^signconventions], where $d\Omega^2 = d\theta^2 + sin^2(\theta) d\phi^2$, and $r_s$ is the schwarzschild radius - that is twice the 'mass'[^blackholesdonthavemass] $M$ in geometric units of $c=g=1$

This isn't very clear, so lets examine it. The $d$ terms on the right hand side (eg $dt$) represent infinitesimals. $ds^2$ represents the square of the length of this curve - note that $ds^2$ can be 0 or negative, and is called the spacetime interval. When your displacement $(dt, dr, dtheta, dphi)$ is timelike, $ds^2$ is also the proper time squared ($d\tau^2$). The fact that this $ds^2$ and the $ds$ we picked for our parameterisation are the same is not a total coincidence - for timelike curves, we pick $ds_{parameterisation} = d\tau$. In reality, $ds$ refers to the general concept of arc length, the length of a curve, which is why the notation is re-used

The sign[^signconventions] of $ds^2$ defines what kind of geodesic we have

1. If $ds^2 > 0$, our curve is spacelike
2. If $ds^2 == 0$, our curve is lightlike
3. If $ds^2 < 0$, our curve is timelike

note that when a curve is lightlike, no proper time $d\tau$ ever elapses along our curve, as $d\tau = 0$

The line element can also be thought of as an expanded out form of when you apply your metric tensor to an infinitesimal displacement, $g(du, du)$. When $du = (t, r, \theta, \phi)$, we recover our line element

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

Vectors, and tensorial objects in general are always associated with a point in spacetime, which is their origin in a sense, more formally they are tangent vectors. Their 'origin' is where you must calculate the metric tensor (and other tensors) to be able to do operations on them

## Numerical differentiation vs automatic differentiation

Now that we have our metric tensor, we need to take its derivatives. There are two easy way to calculate the derivatives: numerical differentiation, and automatic differentiation. A recent I made recently covered AD, and I would recommend using it. However here, we will approximately define the derivatives for space savings. Remember that the metric tensor is calculated at a coordinate, and is a function of the coordinate system. Lets make this explicit, by defining $g_{\mu\nu}$ as $g_{\mu\nu}(t, r, \theta, \phi)$

$$g_{\mu\nu,t} = \partial_t g_{\mu\nu} == \frac{(g_{\mu\nu}(t + h, r, \theta, \phi) - g_{\mu\nu}(t - h, r, \theta, \phi))}{2h} $$

This is a simple centered derivative to calculate the derivative in the first coordinate direction, often called $\partial_0$, which is equivalent to $\partial_t$ in schwarzschild

# A brief pause to review

So far we have:

1. Defined the paths that lightrays take as likelight geodesics. These have a position $x^\mu$, and a velocity $v^\mu$. We know these are geodesics where $g_{\mu\nu} v^\mu v^\nu = 0$

2. Found out how to read a metric tensor $g_{\mu\nu}$ from a line element

3. Understood how to plug numbers into the geodesic equation, to get our acceleration

Unfortunately, this was the simple part of this article

# Initial conditions

Initial conditions in this corner of general relativity are not a good, fun time. This is where we get into the less well understood corners of general relativity, and where mistakes tend to be made

The most foundational element of general relativity is that locally, from the perspective of an observer, spacetime is always flat. Even within the event horizon of a black hole, or out in the wackiest regions of general relativity, this is strictly always true - except where there are singularities, which are where the theory breaks down

The metric tensor for flat spacetime is the minkowski metric, called $\eta_{\mu\nu}$. It is a 4x4 metric tensor, that has the definition

| |t|x|y|z|
|-|-|-|-|-|
|t|-1|0|0|0|
|x|0|1|0|0|
|y|0|0|1|0|
|z|0|0|0|1|

We also know that spacetime is defined by our metric tensor, $g_{\mu\nu}$. So to state that spacetime is always flat, we must be able to define a transform to make spacetime flat

Diagonalising a matrix is defined via the following equation

[^diag]https://physics.stackexchange.com/questions/721265/can-the-metric-tensor-always-be-diagonalized

$$g = e \eta e^{-1}$$

Where $g$ is our metric tensor, $\eta$ is our flat spacetime[^hugecaveat], and $e$ is our diagonalising matrix. Great!

Next we take the column vectors of $e$: $e_0, e_1, e_2, e_3$. These represent basis vectors for our spacetime, called *tetrads*. Tetrads are the fundamental component of our initial conditions, that we use to construct light rays

Note: Here lies a very fundamental property of general relativity that makes it work. The minkowski metric is always in *cartesian* coordinates. This means that no matter what coordinate system is, we can always define a transformation to locally flat cartesian coordinates, and back again. This is the fundamental basis of how to work with a coordinate free theory, but still be able to do useful calculations on it

## Tetrads

Tetrads, also called vierbein, a frame field, or triads in 3 dimensions, represent the "reference frame" of an observer. A reference frame defines a mapping from our locally flat spacetime, to our global spacetime, and back. Some key properties of reference frames are:

1. They are not unique, different values for $e_k$ represent different reference frames, but still solve our equation $g = e \eta e^{-1}$
2. The vectors are orthonormal to each other
3. $e_0$ represents the motion of an inertial observer through that spacetime, and creates a timelike geodesic
4. Used as basis vectors, these define a mapping from our locally flat spacetime $\eta_{\mu\nu}$ to our globally curved spacetime $g_{\mu\nu}$

Notationally, they are used like this:

$v_{curved}^{\mu} = e^{\mu}_i v_{flat}^i $

This says that a vector $v_{flat}$ in our locally flat, minkowski spacetime, may be transformed into the equivalent vector in our curved spacetime $v_{curved}$, by multiplying by $e^\mu_i$

This notation is deeply confusing in my opinion, but is common in the literature. The lower index represents the individual tetrads, and the upper index the components of that tetrad, so the equation in vector form reads:

$v_{curved} = e_0 \cdot v^0_{flat} + e_1 \cdot v^1_{flat} + e_2 \cdot v^2_{flat} + e_3 \cdot v^3_{flat} $

or

$v^\mu_{curved} = (e_0 \cdot v^0_{flat} + e_1 \cdot v^1_{flat} + e_2 \cdot v^2_{flat} + e_3 \cdot v^3_{flat})^\mu $

Tetrads also have an *inverse*, which notationally is spelt like this:

$v^i_{flat} = e^i_\mu v^\mu_{curved}$

If you're thinking, that sure seems unnecessarily confusing and incredibly ambiguous, then yes it is. To calculate the inverse $e^i_{\mu}$, we simply take the matrix inverse $e^i_{\mu} = (e^\mu_i)^{-1}$. If you recall from earlier, $e^\mu_i$ is just the notation for the 4x4 matrix we got out of solving the equation $g = e\eta e^{-1}$ for $e$, and that inverse is the 4x4 matrix inverse

To put all of this in a *far* simpler way, given a solution $e$ to the equation $g_{curved} = e \eta_{flat} e^{-1}$, $v_{curved} = e v_{flat}$, and $v_{flat} = e^{-1} v_{curved}$

If this section seems overly long, its because I'm attempting to spare some poor individual the pain of understanding all the hidden details of the notation that are not explained in the literature anywhere. There is a lot of literature that explains what a tetrad is, and what it represents - but seemingly very little that goes through the basics

# How do we use this to make a geodesic?

## Minkowski

In minkowski spacetime, the metric tensor is trivially diagonal. To make a lightlike geodesic, recall that we want to ensure that $\eta_{\mu\nu}v^\mu v^\nu = 0$. Equivalently, we say:

$ds^2 = -dt^2 + dx^2 + dy^2 + dz^2 = 0$

So first off, lets set $dt$ to $-1$. Then we know that we want $-dt^2 == dx^2 + dy^2 + dz^2$, so therefore $-1 = dx^2 + dy^2 + dz^2$

Because of this, if we plug in any 3-vector with length $1$ as our spatial components $dx, dy, dz$, and use $-1$ as our $dt$ component, we'll end up with a lightlike geodesic. That is to say:

$(-1, v_x, v_y, v_z)$ with $|v| = 1$ is lightlike.

Phew. We now know how to define the velocity of a lightray in flat spacetime. Note that while we construct $v^\mu$ according to this procedure, the position $x^\mu$ can be arbitrary in a flat spacetime. In the next step however, we assume that $x^\mu$ is strictly $0$

## From minkowski, to curved spacetime

W want to go from a quantity in our local minkowski spacetime, to our globally curved spacetime. We already know how to take a vector, and promote it from a locally flat minkowski spacetime, to the overarching curved spacetime:

$v^\mu_{curved} = e^\mu_i v^i_{flat}$

This gives us a valid geodesic velocity, that we can plug into our equations and trace forwards in time. Note that the tetrads, and all other quantities are calculated at a position $p^\mu$, and so this position dictates the position of our geodesic as well. If we wanted to create geodesics across an area, we'd have to do this entire process for each point separately

One key note is the phrase *forwards* in time. It is extremely common in these simulations to start light rays at the camera, negate the $e_0$ component, and trace rays *backwards* in time. This is equivalent to rays going forwards in time, entering from the external universe - but is a lot cheaper to simulate

# Tetrads define an observer, and observers aren't unique

A frame field defines *a* observer, but there are multiple. The tetrad field has infinitely many solutions, for different orientations and velocities of observers. There are two key things here

1. $e_0$ is a timelike vector, and represents the motion of our observer. When treated as a geodesic, it represents the path our observer takes
2. There is no way to know what the actual motion of our observer is, other than by following that geodesic forwards in time, and observing what happens

General relativity, in the most general case, gives us absolutely no information locally whatsoever. We have no way to say for certain what an observer will do. If an observer has the velocity (1, 0, 0, 0), that could mean literally anything. From your perspective, they might be moving near the speed of light, or be nearly in a black hole, or stationary relative to us. That value means next to nothing. The only way to know how fast an observer is going relative to something else, is to a priori know something about the spacetime that observer is in

That said, we're still missing one key piece, which is: Given an observer, how do we make an observer that is moving relative to that observer?

Equivalently, how do we 'boost' a frame field, to make it so it is of a frame field travelling in a direction with a velocity?

## Boosting a frame field

https://arxiv.org/pdf/0904.4184.pdf 1.4.18

We know already that $e_0$ represents the observer in our tetrad field. If we manipulate it, we can therefore set the relative speed of a new observer, relative to our old one. Given a tetrad field $e_k$, we already know how to construct a lightlike velocity, so what we need to do is use our existing tetrad field $e_k$ to instead construct a new *timelike* velocity, representing a boosted observer. We additionally need to perform a *lorentz boost*, representing the change in the frame field as a result of changing its velocity

To make a new timelike velocity in our locally flat spacetime, we're looking for a vector $k^\mu$ that satisfies the identity $\eta_{\mu\nu} k^\mu k^\nu = -1$

As it turns out, it is not overly complicated, so lets define it here. Lets assume we have a regular ol' 3-velocity $v$ with $|v| < 1$. A timelike 4-velocity then has the general definition:

$$U = (\gamma, \gamma v_x, \gamma v_y, \gamma v_z) $$

$\gamma$ is the lorentz factor, defined in minkowski spacetime as:

$$\gamma = 1/sqrt(1 - |v|^2)$$

To build our timelike 4-velocity is then straightforward, we calculate the lorentz factor first, and then build our 4 velocity $U$. To calculate our new boosted $e_0$, we do

$$e_{0new} = e^\mu_i U^i$$

And replace our old one

## We need to modify the other tetrad components too

To perform a lorentz boost, we need to calculate the coefficients of the boost matrix, $B^i_j$, and then apply it to our tetrads

https://arxiv.org/pdf/2404.05744

$$ B^i_j = \delta^i_j + (v^i + u^i)(v_j + u_j) / (1 + \gamma) - 2v^i u_j $$

Where $u$ = our old $e_0$, and $v$ = our new $e_0$. $\delta^i_j$ is called the kronecker delta, and takes on a value of $1$ when $_{i == j}$, and $0$ otherwise. Ie it is the identity matrix. $\gamma$ is defined as $-v \cdot u$, ie the dot product of the two 4-vectors

To then utilise this boost, we do, for i=1,2,3

$$ E^\mu_i = B^\mu_j e^j_i $$

Where $E_k$ is our new spatial tetrad parts

# The complete procedure

Step 1: We calculate our metric tensor $g_{\mu\nu}$ at our starting coordinate, probably a camera position. It is a 4x4 matrix, defined generally by a line element

Step 2: We solve the equation $g=e\eta e^{-1}$ for some arbitrary $e$, *or* we google an appropriate frame field (INSERT SPACETIME PAPER). To do this, we can use a relativistic gram-schmidt equivalent, which is discussed [^gram]

Step 3: We define a direction and velocity for our observer, relative to that initial arbitrary frame if we want to, and construct a new $e_0$ by performing a boost on our frame field

Step 4: We perform a lorentz boost $B$ on $e_{1,2,3}$, to calculate the rest of the frame field for this observer

Step 5: We then construct an initial geodesic velocity in locally flat spacetime. To do this, we pick a cartesian direction $v^k$ with $|v|$ = 1, the direction through a pixel on our screen, where the camera is at the origin

Step 6: We then use this tetrad to construct a geodesic velocity in our curved spacetime, by doing $v_{curved}^\mu = e^\mu_i v^i$, and flip the sign of $e_0$ for light rays so that we trace backwards in time

Step 7: Once we have a position, and velocity, we plug these into the geodesic equation, and integrate

Step 8: Once we have run a certain number of iterations, or some more advanced condition based on the specific coordinate system or metric we're using, we terminate the simulation and render out

Step 9: Then we go outside and talk to our friends and family, who are probably getting worried

# This is actually simpler in code

## Step 1: The metric tensor

First up, we need a tensor type. We'll be dealing with $x*y*z$ tensors, so we're going to go through the slightly painful step of defining a multidimensional type up front. Unfortunately, std::mdarray hasn't been voted in yet, and std::mdspan isn't widely supported, so we're going to have to Roll Our Own, which is going to be a common theme in this tutorial series

```c++
template<typename T, size_t size, size_t... sizes>
struct md_array_impl
{
    using type = std::array<typename md_array_impl<T, sizes...>::type, size>;
};

template<typename T, size_t size>
struct md_array_impl<T, size>
{
    using type = std::array<T, size>;
};

template<typename T, size_t... sizes>
using md_array = typename md_array_impl<T, sizes...>::type;

template<typename T, typename Next>
constexpr
auto& index_md_array(T& arr, Next v)
{
    assert(v < (Next)arr.size());

    return arr[v];
}

template<typename T, typename Next, typename... Rest>
constexpr
auto& index_md_array(T& arr, Next v, Rest... r)
{
    assert(v < (Next)arr.size());

    return index_md_array(arr[v], r...);
}

//strictly NxM
template<typename T, int... N>
struct tensor
{
    md_array<T, N...> v = {};

    template<typename... V>
    T& operator[](V... vals)
    {
        return index_md_array(backing, vals...);
    }

    template<typename... V>
    const T& operator[](V... vals) const
    {
        return index_md_array(backing, vals...);
    }
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
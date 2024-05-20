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

The metric tensor is responsible for many things, and general relativity is in part the study of this fundamental object. To raise an index, we do it as such:

$$ v_\mu = g_{\mu\nu} v^\nu $$

And lowering an index is performed like this

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

There are three kinds of geodesics

1. Spacelike
2. Timelike
3. Lightlike

Timelike geodesics represent the path that objects with rest mass follow, and lightlike geodesics represent the path that objects without *rest* mass follow (ie light). We will be ignoring spacelike geodesics entirely. Note that it does not matter what the mass of an object is, only that it has mass at rest[^massissortofignored]

We'd like to render our black hole by firing light rays around, so what we're looking for is lightlike geodesics

## Geodesics

A geodesic has two properties: a position $x^\mu$, and a velocity. In normal every day life, velocity is defined with respect to time. In general relativity, we have several concepts of time that we could use

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

## A real metric tensor

There are many different black holes, and many different ways of representing each of them. Today we're going to pick the simplest kind: the schwarzschild black hole

A metric tensor defines both the curvature of spacetime, as well as implicitly a coordinate system associated with it. For schwarzschild, the `line element` is often defined like this. For simplicity, we're assuming that the speed of light $c=1$

$$ ds^2 = d\tau^2 = -(1-\frac{r_s}{r}) dt^2 + (1-\frac{r_s}{r})^{-1} dr^2 + r^2 d\Omega^2 $$

This is the wikipedia definition, where $d\Omega^2 = d\theta^2 + sin^2(\theta) d\phi^2$, and $r_s$ is the schwarzschild radius - that is twice the mass $M$ in geometric units of $c=g=1$

This still isn't that useful, so lets examine it. the $d$ terms on the right hand side (eg $dt$) represent infinitesimals. $ds^2$ represents the square of the length of this curve - note that $ds^2$ can be 0 or negative, and is called the spacetime interval. When your displacement $(dt, dr, dtheta, dphi)$ is timelike, $ds^2$ is also the proper time squared ($d\tau^2$). The fact that this $ds^2$ and the $ds$ we picked for our parameterisation are the same is not a total coincidence - for timelike curves, we pick $ds_{parameterisation} = d\tau$

The sign of $ds^2$ defines what kind of geodesic we have

1. If $ds^2 > 0$, our curve is spacelike
2. If $ds^2 == 0$, our curve is lightlike
3. If $ds^2 < 0$, our curve is timelike

note that when a curve is lightlike, no proper time $d\tau$ ever elapses along our curve

The line element can also be thought of as an expanded out form when you apply your metric tensor to an infinitesimal displacement, $g(du, du)$. When $du = (t, r, \theta, \phi)$, we recover our line element

This means that we can read the matrix $g_{\mu\nu}$ directly off from the line element. If we have the line element

$ds^2 = k_1 da^2 + k_2 dadb + k_3 db^2$ etc, we get the metric tensor:

|.|a|b|
|-|-|-|
|a|$k_1$|$\frac{1}{2}k_2$|
|b|$\frac{1}{2}k_2$| $k_3$|

The schwarzschild metric is diagonal, so we get

|.|t|r|$\theta$|$\phi$|
|-|-|-|-|-|
|t|$-(1-\frac{r_s}{r})$|0|0|0|
|r|0|$(1-\frac{r_s}{r})^{-1}$|0|0
|$\theta$|0|0|$r^2$|0|
|$\phi$|0|0|0|$r^2 sin^2(\theta)$|

As a 4x4 matrix. Note that this matrix is a function of the coordinate system, and it must be recalculated at a specific point in space where you want to apply it. If we want to raise or lower the velocity of our geodesic, we must calculate the metric tensor *at* the position where the velocity vector is

Vectors are always associated with a point in spacetime, which is their origin in a sense, more formally they are tangent vectors. Their 'origin' is where you must calculate the metric tensor (and other tensors) to be able to do operations on them

## Numerical differentiation vs automatic differentiation

Lets calculate the derivatives

## The metric tensor

# Initial conditions

## Tetrads?

Oh boy

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
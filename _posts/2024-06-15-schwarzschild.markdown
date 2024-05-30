---
layout: post
title:  "Implementing General Relativity: Rendering the schwarzschild black hole, in C++"
date:   2025-05-18 00:35:23 +0000
categories: C++
---

General relativity is pretty cool. Doing anything practical with it is a bit of a pain, and a lot of information is left to the dark arts. More than that, a significant proportion of information available on the internet is unfortunately incorrect, and many visualisations of black holes are wrong in one way or another - even ones produced by physicists! We're going to focus on two things:

1. Giving you a practical crash course in the theory of general relativity
2. Turning this into code that actually does useful things

This tutorial series is not going to be a rigorous introduction to the theory of general relativity. There's a lot of that floating around - what we're really missing is how to translate all that theory into action

By the end of this infinitely long tutorial series, you will smash black holes and neutron stars together, without a supercomputer, in just a few minutes flat [^pleasehireme]

<iframe width="560" height="315" src="https://www.youtube.com/embed/AKSHXiBGPpw?si=z3AAMvbCAP02Pw8p" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

[^pleasehireme]: If someone hired me to do this full time I'd be very happy

Our first step along this journey is the classic starting point: The schwarzschild black hole

# What are we actually simulating?

The end goal here is to end up with this:

![Black hole](/assets/schwarzs.png)

First up, a black hole isn't a physical thing you can touch, and the schwarzschild black hole isn't made of anything tangible. It exists purely gravitationally - its a stable, self supporting gravitational field. It is made up of spacetime being curved, and nothing else. The schwarzschild black hole is in the class of objects known as a "vaccum solution", which is to say that there is no matter or what you might call mass[^blackholemass] present whatsoever

What we actually see in this picture is purely as a result of spacetime being curved, bending the path of anything that travels through it. We create light rays at the camera, trace them around our curved spacetime according to the equations of general relativity, and then render the background according to where our rays of light end up. Some rays of light get trapped at the event horizon, resulting in the black 'shadow' of the black hole

![Raytracing](/assets/blograys.png)

There's a reason I got into programming and not art

[^blackholemass]: Black holes have no mass, in a relativistic stress-energy-tensor sense. This might be confusing, because A: Everyone talks about the mass of a black hole, and B: stuff clearly orbits a black hole. What black holes actually have is a mass equivalent parameter, which is often called $M$, and the effect that a black hole has on spacetime is equivalent to a mass-ive body with that mass. This effect is often referred to as mass, but is *not* the same thing as the everyday concept of mass, which is called 'matter'. This level of pedantry is necessary in the future, when we will begin to struggle what we mean by mass. In the general case, there is no single definition for the mass of a black hole, and in some cases it is undefineable

# Breaking down a simulation

There are 3 parts to any simulation

1. The initial conditions
2. The evolution equations
3. Termination

We're going to be focusing on the pipeline of simulating this from start to end, with complete examples so that nothing gets left in the gaps

Step 1 will be left until last, and is not the focus of this article. Step 2 requires us to solve something called the geodesic equation: which is the equations of motion for a ray of light (and objects in general) in general relativity. To be able understand it we'll need some background

# Mathematical background

## Tensor index notation

General relativity uses its own specific conventions for a lot of maths that is somewhat dense at first glance, though very expressive once you get used to it. This is the very first thing we need to demystify, because understanding how to translate this new notation into familiar concepts is at the core of being able to interpret equations

In general relativity, there's one key idea compared to maths you might be more familiar with, that we need to get a handle on first before we go anywhere else today:

### Contravariance, and covariance

In everyday maths, a vector is just a vector. We informally express this as something like $v = 1x + 2y + 3z$, where $x$ $y$ and $z$ are our coordinate system basis vectors[^basisvectors]. When dealing with vectors, its not uncommon to index the vector's components by a variable/index as a shorthand:

[^basisvectors]:
    Basis vectors are the direction vectors of our coordinate system that we use to build our own vectors on top of. When you have a vector $(1, 2, 3)$, its generally implicit in the definition that each of these components refers to a different direction in your coordinate system, where the direction is dependent on your basis vectors. Normally your basis vectors are something like $(1,0,0)$, $(0,1,0)$, $(0,0,1)$ for x, y, and z - but in theory they could be anything - as long as they're 'linearly independent'. All that means is that we aren't repeating ourselves with our basis vectors, and they truly represent different directions

$$ \sum_{k=0}^2 v_k == v_0 + v_1 + v_2$$

This is an example of how we'd express summing the components of a vector. Tensor index notation takes this notation one step further, by giving the vertical positioning of the index (here $_k$ ) additional meaning. In general relativity, $v_k$, and $v^k$ mean two different things. So, from now on, indices can be:

1. Contravariant[^variance] (raised):

$$ v^\mu $$

2. Covariant[^variance] (lowered)

$$ v_\mu $$

[^variance]:
    Contravariant vectors are so called because when your coordinate system changes, they scale *against* the axis. Eg if you have a position 0.1 in meters, and your coordinate system changes to kilometers, you have 0.1/1000 km. Covariant vectors change *with* the axis. Invariance is something that does not change with a change in the coordinate system, eg scalar values

    A good mnemonic for remembering which is an up index, and which is a down index, is that up indices are contravariant, and down indices are covariant. But seriously, you just have to remember it


Additionally, objects such as matrices can have more than one index, and the indices can have any "valence" (up/down-ness). For example, $A^{\mu\nu} $,  $ A^\mu_{\;\;\nu} $, $ A_\mu^{\;\;\nu} $, and $ A_{\mu\nu} $ are all different representations of the same object $A$. The first is the contravariant form, the middle two have mixed indices, and the last one is the covariant form

We can add more dimensions to our objects as well, eg: $ \Gamma^\mu_{\;\;\nu\sigma} $[^oftenwritten] is a 4x4x4 object in this article. These objects are all referred to as "tensors", a term which has lost virtually all meaning in computer science. In its strict definition, a tensor is an object that transforms in a particular fashion in a coordinate change: in practice, everyone calls everything a tensor, unless its relevant for it not to be. Here, we will refer to anything which takes an index as being a tensor, unless it is relevant. The other important class of objects are scalars, which are just values

[^oftenwritten]: This example object is generally written slightly more compactly, as $ \Gamma^\mu_{\nu\sigma} $, and is known as "christoffel symbols of the second kind". They crop up a lot. In the literature, the notation is often slightly ambiguous as to which specific index is raised or lowered, and you have to deduce it from context

One thing to note: Tensors and scalars are generally functions of the coordinate system, and vary from point to point in our spacetime. While we write $A_{\mu\nu}$, what we really *mean* is $A_{\mu\nu}(x, y, z, w)$ - its really a function that returns a 4x4 matrix, when we feed coordinates into it

### Changing the valence of an index: raising and lowering indices

The most important object in general relativity that I'll introduce here is the metric tensor, normally spelt $g_{\mu\nu}$, and is generally given in its covariant form (see the section about spacetime at the end for a more detailed discussion). This object defines spacetime - how it curves, and how things move, and virtually everything in general relativity involves the metric tensor in some form or other

The metric tensor is a 4x4 symmetric matrix. Because it is symmetric, $ g_{\mu\nu} =  g_{\nu\mu} $, and as a result only has 10 independent components. The metric tensor is also often thought of as a function taking two arguments, $g(u,v)$, and performs the same function as the euclidian dot product. That is to say, where in 3 dimensions you might say $a = dot(v, u)$, in general relativity you might say $a = g(v, u)$

For the metric tensor, and the metric tensor *only*, the contravariant form of the metric tensor is calculated as such:

$$ g^{\mu\nu} = (g_{\mu\nu})^{-1} $$

That is to say, the regular 4x4 matrix inverse of treating $g$ as a matrix[^onlythemetrictensor]

[^onlythemetrictensor]: This is generally never true of any other object, and $  A^{\mu\nu} = (A_{\mu\nu})^{-1} $ is likely a mistake

The metric tensor is responsible for many things, and one of its responsibilities is raising and lowering indices. To raise an index, we do it as such:

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

Here, $\nu$ and $\sigma$ are 'dummy' indices (ie they are repeated), and $\mu$ is a 'free' index (not repeated). Each dummy index is summed only with itself, and form the bounds for our $\sum_{}$'s

The size of the resulting tensor is equal to the number of free indices. One more key rule is that only indices of opposite valence sum: eg an up index *always* sums with a down index, and you do not sum two indices of the same valence - though this almost never crops up. Jump to the indices footnote for more examples

The last thing we need to learn now is how to raise and lower the indices of a multidimensional object using the metric tensor, eg $ A^{\mu\nu} $. As a rule, we set the dummy index to the index we wish to raise and lower in our object, and set the free index to the new relabeled index, in the metric. Eg to lower the first index, we sum over it, and set the second index of the metric tensor to our new index name

$$ \Gamma_{\mu\nu\sigma} = g_{\mu\gamma} \Gamma^{\gamma}_{\;\;\nu\sigma} $$

Remember that the metric tensor is always symmetric, so we could also write this

$$ \Gamma_{\mu\nu\sigma} = g_{\gamma\mu} \Gamma^{\gamma}_{\;\;\nu\sigma} $$

Here's another example:

$$\Gamma^{\mu\nu}_{\;\;\;\sigma} = g^{\nu\gamma} \Gamma^\mu_{\;\;\gamma\sigma}$$

More examples are provided in the indices footnote

At the top, we considered the metric tensor as a function taking two arguments, $g(u,v)$. One helpful way to look at covariant and contravariant indices is as partial function applications of the metric tensor:

$$ u_\mu v^\mu == u^\mu v_\mu == g_{\mu\nu} u^\nu v^\mu == g^{\mu\nu} u_\nu v_\mu == g(u, v) == g(u, \cdot) \circ v $$

That is to say, we can treat applying the metric tensor to $v^\mu$ to get $v_\mu$ like currying

If like me you've hit this point and are feeling a bit tired, don't worry. These are rules we can refer back to whenever we need them, and the footnotes will contain lots of examples. Here's a picture of my cat for making it this far:

![wesley](/assets/wesleysmaller.png)

## Raytracing as differential equations, in 3d

What we're trying to accomplish here in this article is basic raytracing. In a normal, 3d raytracer, we construct a ray with a start position $x^i$, and a velocity $v^i == dx^i/ds$, where $s$ is a parameter that is often time. If we're raytracing simple 3d graphics, rays travel in straight lines. In general, we don't care about the magnitude of $v^\mu$, as it simply corresponds to rescaling $ds$. Eg 3 meters/second is equivalent to 30 meters/decasecond

Lets imagine that our rays do not move in straight lines, and that at every point in space, an acceleration $a^\mu$ is applied to our ray. To find the points on our ray in the general case, we must integrate some basic equations of motion. That is to say, we need to solve the following equations for $x$

$$dx^\mu/dt = v^\mu \\
dv^\mu/dt = a^\mu\\
$$

In newtonian dynamics, $a^\mu$ might be the acceleration given by all the other bodies in our simulation applying a force to us (which we will label with the index ${s}$):

$$a^\mu = (-G \sum_{k=0, k != s}^n \frac{m_k}{|x_s - x_k|^3} (x_s - x_k))^\mu$$

https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation#Vector_form

And in straight line raytracing, $a^\mu=0$, allowing us to directly integrate the equations

## Geodesics: the basics

In general relativity:

$$a^\mu = -\Gamma^\mu_{\alpha\beta} v^\alpha v^\beta$$

Where $\Gamma^\mu_{\alpha\beta}$ is derived from the metric tensor, and is implicitly a function of $x^\mu$. This is called the geodesic equation, and describes the motion of GR's notion of rays, which are called geodesics

In general relativity, the motion of all objects and lightrays are described by geodesics. On earth, if you move in a straight line you'll end up going in a circle: geodesics are the more mathematical version of this concept. The only physically accurate way to get from point A, to point B, or relate them in any fashion in the general case, is via a geodesic

## Timelike, lightlike, and spacelike geodesics

There are three kinds of geodesics:

1. Timelike: these are paths which can be travelled by an observer (or particle) with mass (even if very minimal)
2. Lightlike: these are paths which can only be travelled by light[^masslessparticles]
3. Spacelike: these geodesics are a bit more complicated to interpret, and we will ignore them

[^masslessparticles]: Or any particle without *rest* mass. Do not that light *does* have mass, it just doesn't have rest mass. It shows up in the stress-energy tensor as a result, and exerts a force of gravity

We'd like to render our black hole by firing light rays around and finding out what our spacetime looks like, so what we're looking for in this article is lightlike geodesics

## Geodesics: the less basics

A geodesic has two properties: a position $x^\mu$, and a velocity $v^\mu$. In normal every day life, velocity is defined as the rate of change of position with respect to time. In general relativity, we have several concepts of time that we could use

1. The concept of time given to us by our coordinate system, which is completely arbitrary and has no meaning
2. The concept of time as experienced by an observer (including particles), called proper time, $d\tau$
3. An even more completely arbitrary concept of time that has no meaning

No observer can move at the speed of light, so 2. is right out for lightlike geodesics, though works well for timelike geodesics. 1. Is dependent on our coordinate system and is hard to apply generally (not every coordinate system has a time coordinate), so in general we will always be using 3 for light.

This makes our velocity: $v^\mu = dx^\mu/ds$. $ds$ is known as an affine parameter, and represents a fairly arbitrary parameterisation[^whatsaparameterisation] of our curve/geodesic

## Integrating the geodesic equation

We're  going to solve one of our major components now: We've already briefly seen the geodesic equation, which looks like this:

$$ a^\mu = -\Gamma^\mu_{\alpha\beta} v^\alpha v^\beta $$

Where more formally, our acceleration $ a^\mu == \frac{d^2x^\mu}{ds^2} $, and

$$ \Gamma^\mu_{\alpha\beta} = \frac{1}{2} g^{\mu\sigma} (g_{\sigma\alpha,\beta} + g_{\sigma\beta,\alpha} - g_{\alpha\beta,\sigma}) $$

Note that:

$$ g_{\mu\nu,\sigma} == \partial_\sigma g_{\mu\nu} $$

Aka, taking the partial derivatives in the direction $\sigma$, as defined by our coordinate system. This equation is likely to stretch our earlier understanding of how to sum things, so we'll write it out manually[^bearinmind]:

$$ \Gamma^\mu_{\alpha\beta} = \frac{1}{2} \sum_{\sigma=0}^3 g^{\mu\sigma} (\partial_\beta g_{\sigma\alpha} + \partial_\alpha g_{\sigma\beta} - \partial_\sigma g_{\alpha\beta})$$

[^bearinmind]: Bear in mind that we're just multiplying scalar values together here, so we can do all the sums individually. That is to say, $ \Gamma^\mu_{\alpha\beta} = \frac{1}{2} g^{\mu\sigma} (g_{\sigma\alpha,\beta} + g_{\sigma\beta,\alpha} - g_{\alpha\beta,\sigma}) $ == $ \frac{1}{2} g^{\mu\sigma} g_{\sigma\alpha,\beta} + \frac{1}{2} g^{\mu\sigma}g_{\sigma\beta,\alpha} - \frac{1}{2} g^{\mu\sigma}g_{\alpha\beta,\sigma} $. Note that we apply the derivative *first*, then multiply. Expressions are often constructed to avoid these kind of ambiguity, by introducing intermediate tensors

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

Phew. This loop may look horrendously inefficient, and it is. GR is not cheap to render in the general case. There are some symmetries here that reduce the computational complexity, notably that $\Gamma^\mu_{\alpha\beta} = \Gamma^\mu_{\beta\alpha}$. Additionally, in many metrics most of $\Gamma$'s components are 0, which helps too

So: to integrate a geodesic, we start off with an appropriate position, $x^\mu$, an appropriate velocity, $v^\mu$, and calculate our acceleration via the geodesic equation. We then integrate this. During the process of this, we'll need a metric tensor $g_{\mu\nu}$ and the partial derivatives of it

We're getting close to being able to integrate our equations now. We now need three more things:

1. A real metric tensor
2. An initial position $x^\mu$, which will be our camera position
3. An initial velocity, which is a lot more complex

## A real metric tensor

There are many different black holes, and many different ways of representing each of them. Today we're going to pick the simplest kind: the schwarzschild black hole. Its important to note, schwarzschild black holes are just one kind of black hole, in a whole field of different kinds of black holes. The classic schwarzschild metric is also only one representation of the schwarzschild black hole - there are other coordinate systems for it that are radically different metric tensors

A metric tensor fundamentally defines the curvature of spacetime - and it is the central object of general relativity. It also implicitly expects you to be using a certain coordinate system, though the coordinates can be anything. The metric tensor is often expressed in a form called the "line element", which reads like this:

$$ ds^2 = -d\tau^2 = -(1-\frac{r_s}{r}) dt^2 + (1-\frac{r_s}{r})^{-1} dr^2 + r^2 d\Omega^2 $$

This is the wikipedia definition[^wikipedia] [^signconventions], where $d\Omega^2 = d\theta^2 + sin^2(\theta) d\phi^2$, and $r_s$ is the schwarzschild radius - that is twice the 'mass'[^blackholesdonthavemass] $M$ in geometric units of $c=g=1$

This isn't very clear, so lets examine it. The $d$ terms on the right hand side (eg $dt$) represent infinitesimals. $ds^2$ is called the spacetime interval: note that it can be negative, or 0. When your displacement $(dt, dr, dtheta, dphi)$ is timelike, $ds^2 > 0$, and $ds^2$ is also the (negative) proper time[^propertime] squared $d\tau^2$. The fact that this $ds^2$ and the $ds$ we picked for our parameterisation are the same is not a total coincidence - for timelike curves, we pick $ds_{parameterisation} = d\tau$. In reality, $ds$ refers to the general concept of arc length, which is why the notation is re-used

The sign[^signconventions] of $ds^2$ defines what kind of geodesic we have. If we plug a velocity $(dt, dr, d\theta, d\phi)^\mu = v^\mu$ in here, and calculate the metric at our position $(t, r, \theta, \phi)^\mu = x^\mu$, then:

1. If $ds^2 > 0$, our curve is spacelike
2. If $ds^2 == 0$, our curve is lightlike
3. If $ds^2 < 0$, our curve is timelike

note that when a curve is lightlike, no proper time $d\tau$ ever elapses along our curve, as $d\tau = 0$

We can read the matrix $g_{\mu\nu}$ directly off from the line element[^thisiswhy]/*The line element can also be thought of as an expanded out form of when you apply your metric tensor to an infinitesimal displacement, $g(du, du)$. When $du = (dt, dr, d\theta, d\phi)$, we recover our line element. */. If we have the line element

$$ds^2 = k_1 da^2 + k_2 dadb + k_3 db^2$$ etc, we get the metric tensor:

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

As a 4x4 matrix. Note that this matrix is a function of the coordinate system, and it must be recalculated at a specific point in space where you want to apply it. If we want to raise or lower the velocity of our geodesic, we must calculate the metric tensor *at* the position where the velocity vector is[^theyaretangentvectors]/*Tensors and scalar functions are generally associated with a point in spacetime, which is their origin in a sense, more formally they are tangent vectors - tangent to the 'manifold' that is spacetime. Their origin is where you must calculate the metric tensor (and other tensors) to be able to do operations on them*/

## Numerical differentiation vs automatic differentiation

Now that we have our metric tensor, we need to take its derivatives. There are two easy way to calculate the derivatives: numerical differentiation, and automatic differentiation. A recent post I made recently covered AD, and I would recommend using it. However here, we will approximately define the derivatives for space savings. Remember that the metric tensor is calculated at a coordinate, and is a function of the coordinate system. Lets make this explicit, by defining $g_{\mu\nu}$ as $g_{\mu\nu}(t, r, \theta, \phi)$

$$g_{\mu\nu,t} = \partial_t g_{\mu\nu} == \frac{(g_{\mu\nu}(t + h, r, \theta, \phi) - g_{\mu\nu}(t - h, r, \theta, \phi))}{2h} $$

This is a simple centered derivative to calculate the derivative in the first coordinate direction, often called $\partial_0$, which is equivalent to $\partial_t$ in schwarzschild

# A brief pause to review

So far we have:

1. Defined the paths that lightrays take as likelight geodesics. These have a position $x^\mu$, and a velocity $v^\mu$. We know these are geodesics where $g_{\mu\nu} v^\mu v^\nu = 0$

2. Found out how to read a metric tensor $g_{\mu\nu}$ from a line element

3. Understood how to plug numbers into the geodesic equation, to get our acceleration

It might surprise you to learn that this is the simple part of what we're trying to do, in general. For the purposes of trying to minimise the sheer information overload today before we get up and running with our first black hole, we're going to use some pre-baked initial conditions, instead of calculating them ourselves

# Initial conditions

Initial conditions in this corner of general relativity are not a good, fun time. This is where we get into the less well understood corners of general relativity, and where mistakes tend to be made. We're going to condense a much longer future discussion of initial conditions into the next article in this series - likely when you all conveniently go on holiday

## What are we trying to get out of our initial conditions?

In this phase, what we're trying to do is construct an initial direction $v^\mu$ that our lightray travels towards - a geodesic velocity. In a regular, flat, 3d simulation, its very easy - we define a plane in front of our camera, and construct a ray, from the camera's origin, through a pixel on that plane. If a pixel has a position $p=(x-width/2, y-height/2, d)$ on that plane, then the ray's direction in 3d space is $d=norm(p-o)$

The question then becomes: how do we translate our ray direction $d$ in 3d space, to a valid geodesic velocity in 4d spacetime? The answer is: tetrads

# Tetrads are my own personal nightmare, and soon they will be yours

TODO: CHECK INDEX NOTATION

https://www2.mpia-hd.mpg.de/homes/tmueller/pdfs/catalogue_2014-05-21.pdf

Tetrads, also known as frame fields, or vielbein, are a field of four 4-vectors, that are orthonormal (ie perpendicular, and unit lengthed, with respect to the metric tensor). These make up the "frame of reference" of an observer, and are used to translate between what one observer sees and experiences, and the wider universe that we're describing

General relativity demands that spacetime is very much locally flat from the perspective of any observer. And yet, we can observe that spacetime is clearly curved - planets go round the sun, and black holes exist. Translating from an observers locally flat spacetime, to that curved spacetime, is done via our tetrads. These are the objects that define the disconnect between "my space is locally flat" and "my friends space is clearly curved", and how you translate observations between the two

A tetrad consists of 4 orthonormal basis vectors, that form the basis of our spacetime. Each tetrad vector is labelled with a latin index $i$, as such: $e_i$. Each one of these tetrad vectors $e_i$ has 4 components, and so they are spelt $e^\mu_i$ in the literature. This makes up a 4x4 matrix when treated as column vectors, and the inverse (which is a matrix inverse $(e^\mu_i)^{-1}$) is spelt $e_\mu^i$. If this seems very confusing, you are absolutely correct horse

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

$$v^i_{flat} = e_\mu^i v^\mu_{curved}$$

If the direction $d$ points through a pixel in our local plane, we now have a way to construct the initial velocity of our geodesic. One key thing to note is that we almost always trace lightlike geodesics *backwards* in time, which we can accomplish by negating the time component of our lightray, and getting $v_{flat}^\mu = (-1, d_0, d_1, d_2)$

## That's all well and good, but how do I calculate my tetrads?

Luckily, a lot of metric tensors have precalculated tetrads, and we can simply read them off of this page here https://arxiv.org/pdf/0904.4184. For schwarszchild:

$e^0_0 = 1/\sqrt{1-r_s/r} \\$
$e^1_1 = \sqrt{1-r_s/r} \\$
$e^2_2 = 1/r \\$
$e^3_3 = 1/(r sin(\theta))$

Note that these are not unique, and represent a specific kind of observer in this spacetime. While there *is* a unique 'natural' choice, it has no special meaning, and we'll get to this soon. Also note that this paper refers to the upper indices of the tetrads by their coordinate basis, ie $\partial_t$ means the 0th component of the tetrad $e_t$, which is $e_0$ for us. A vector may in general be written $a \partial_t + b\partial_x + c\partial_y + d\partial_z$ assuming a coordinate system $(t, x, y, z)$, and the paper linked above follows this convention for specifying the tetrad components

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
metric<float, 4, 4> schwarzschild_metric(const tensor<float, 4>& position) {
    float rs = 1;

    float r = position[1];
    float theta = position[2];

    metric<float, 4, 4> m;
    m[0, 0] = -(1-rs/r);
    m[1, 1] = 1/(1-rs/r);
    m[2, 2] = r*r;
    m[3, 3] = r*r * std::sin(theta)*std::sin(theta);

    return m;
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

    tensor<float, 4> et = {1/std::sqrt(1 - rs/r), 0, 0, 0};
    tensor<float, 4> er = {0, std::sqrt(1 - rs/r), 0, 0};
    tensor<float, 4> etheta = {0, 0, 1/r, 0};
    tensor<float, 4> ephi = {0, 0, 0, 1/(r * std::sin(theta))};

    return {et, er, etheta, ephi};
}
```

## Step 3: calculating our pixel direciton in flat spacetime

```c++
tensor<float, 3> get_ray_through_pixel(int sx, int sy, int screen_width, int screen_height, float fov_degrees) {
    float fov_rad = (fov_degrees / 360.f) * 2 * std::numbers::pi_v<float>;
    float f_stop = (screen_width/2) / tan(fov_rad/2);

    tensor<float, 3> pixel_direction = {(float)(sx - screen_width/2), (float)(sy - screen_height/2), f_stop};
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
    geodesic g;
    g.position = position;
    g.velocity = tetrads.v[0] * -1 //Flipped time component, we're tracing backwards in time
               + tetrads.v[1] * direction[0]
               + tetrads.v[2] * direction[1]
               + tetrads.v[3] * direction[2];

    return g;
}
```

One thing to note: if we directly plug our ray direction into make_lightlike_geodesic, it may not point where we want it to. Ideally we'd like our z direction to point towards the black hole, our +y direction to be up, and +x to be right. To convert our ray from get_ray_through_pixel, we therefore need to plug in the modified direction {-dir[2], dir[1], dir[0]}

## Step 5 + 6: Integrate the geodesic equation

```c++
//function to numerically differentiate an arbitrary function that takes a position, and a direction
auto diff(auto&& func, const tensor<float, 4>& position, int direction) {
    auto p_up = position;
    auto p_lo = position;

    float h = 0.00001f;

    p_up[direction] += h;
    p_lo[direction] -= h;

    auto up = func(p_up);
    auto lo = func(p_lo);

    return (func(p_up) - func(p_lo)) / (2 * h);
}

//get the christoffel symbols that we need for the geodesic equation
tensor<float, 4, 4, 4> calculate_christoff2(const tensor<float, 4>& position, auto&& get_metric) {
    metric<float, 4, 4> metric = get_metric(position);
    inverse_metric<float, 4, 4> metric_inverse = metric.invert();
    tensor<float, 4, 4, 4> metric_diff; ///uses the index signature, diGjk

    for(int i=0; i < 4; i++) {
        auto differentiated = diff(get_metric, position, i);

        for(int j=0; j < 4; j++) {
            for(int k=0; k < 4; k++) {
                metric_diff[i, j, k] = differentiated[j, k];
            }
        }
    }

    tensor<float, 4, 4, 4> Gamma;

    for(int mu = 0; mu < 4; mu++)
    {
        for(int al = 0; al < 4; al++)
        {
            for(int be = 0; be < 4; be++)
            {
                float sum = 0;

                for(int sigma = 0; sigma < 4; sigma++)
                {
                    sum += 0.5f * metric_inverse[mu, sigma] * (metric_diff[be, sigma, al] + metric_diff[al, sigma, be] - metric_diff[sigma, al, be]);
                }

                Gamma[mu, al, be] = sum;
            }
        }
    }

    //note that for simplicities sake, we fully calculate all the christoffel symbol components
    //but the lower two indices are symmetric, and can be mirrored to save significant calculations
    return Gamma;
}

//use the geodesic equation to get our acceleration
tensor<float, 4> calculate_acceleration_of(const tensor<float, 4>& X, const tensor<float, 4>& v, auto&& get_metric) {
    tensor<float, 4, 4, 4> christoff2 = calculate_christoff2(X, get_metric);

    tensor<float, 4> acceleration;

    for(int mu = 0; mu < 4; mu++) {
        float sum = 0;

        for(int al = 0; al < 4; al++) {
            for(int be = 0; be < 4; be++) {
                sum += -christoff2[mu, al, be] * v[al] * v[be];
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
};

//this integrates a geodesic, until it either escapes our small universe or hits the event horizon
integration_result integrate(geodesic& g, bool debug) {
    integration_result result;

    float dt = 0.005f;
    float rs = 1;
    float start_time = g.position[0];

    for(int i=0; i < 100000; i++) {
        tensor<float, 4> acceleration = calculate_schwarzschild_acceleration(g.position, g.velocity);

        //this is leapfrog integration
        g.velocity += acceleration * dt;
        g.position += g.velocity * dt;

        float radius = g.position[1];

        if(radius > 10) {
            //ray escaped
            result.type = integration_result::ESCAPED;

            return result;
        }

        if(radius <= rs + 0.0001f || g.position[0] > start_time + 1000) {
            //ray has very likely hit the event horizon
            result.type = integration_result::EVENT_HORIZON;

            return result;
        }
    }

    return result;
}
```

## Step 7: Going outside

This was never an option

# The end

Success!

![Black hole](/assets/e1success.png)

The first thing you might notice when putting all this together, is that it is excruciatingly slow to render a black hole, even with multiple threads. This problem is embarrassingly parallel, so next time round, we'll be porting the whole thing to the gpu - as well as building ourselves a custom high performance GPU programming langauge to use

The second thing you might notice is those very ugly polar singularities. This can be alleviated by lowering your timestep near the poles, or by exploiting the spherical symmetry of the metric to move rays into a plane where there is no polar singularity

# Footnote: What is spacetime?

In general relativity, objects move through something called 'spacetime'. Instead of having 3 coordinates, and a time coordinate, we have a single unified coordinate $(a, b, c, d)$, where all 4 elements are on equal footing. It might be tempting to ask if this isn't similar to describing an objects position as $(t, x, y, z)$ in newtonian mechanics, but there are two key differences here

1. There is in general no coordinate $t$ that strictly represents time
2. Spatial distances can no longer be measured consistently, and the only invariant distance is one that fundamentally involves the fourth coordinate called the 'spacetime interval'

Say we have two positions $p_0 = (t_0, x_0, y_0, z_0)$ and $p_1 = (t_1, x_1, y_1, z_1)$, and we'd like to define the distance between them. In newtonian mechanics, we have $d^2 = (x_1 - x_0)^2 + (y_1 - y_0)^2 + (z_1 - z_0)^2$, and we assume that we're measuring at a point in time when $t_1 == t_0$

In general relativity, this distance is no longer invariant unfortunately, and changes from observer to observer. The only thing we can measure consistently is something called the spacetime interval $ds^2$. In flat spacetime (called minkowski), this looks like this:

$$ds^2 = -(t_1 - t_0)^2 + (x_1 - x_0)^2 + (y_1 - y_0)^2 + (z_1 - z_0)^2$$

Here we do have a time coordinate, $t$. Working with minkowski spacetime is special relativity

In general relativity, the above distance equation is generalised, and looks more like this: If $v = p_1 - p_0$, then

$$ds^2 = A v_0^2 + B v_0 v_1 + C v_0 v_2 + D v_0 v_3 + E v_1^2 + F v_1 v_2 + G v_1 v_3 + H v_2^2 + I v_2 v_3 + J v_3^2$$

This is a lot less fun to work with. The coefficients make up a symmetric 4x4 matrix called the metric tensor, and generalises special relativity to curved spacetime with this new invariant spacetime interval. Importantly, general relativity has no fixed coordinate system like cartesian, and no time coordinate to work with: its very difficult to say anything useful given a vector in a random coordinate system

Here are some of the less fun consequences of moving from special to general relativity

1. There is no universal coordinate system. Each coordinate system is only valid for a patch of spacetime, and these patches are stitched together as necessary to traverse the spacetime
2. Space is not necessarily simply connected, eg wormholes
3. None of the coordinates have to represent time and are entirely arbitrary
4. The coordinate system may be periodic
5. Space as a bulk has no concept of saying which direction is *forwards* in time. In general time is defined as the directions that an observer can move in that spacetime
6. Time travel is explicitly permitted, and shows up literally everywhere. What direction is forwards in time when the causal structure of spacetime itself forms a loop? Good question

# Footnote: Indices

I am a pattern matcher by nature, so here's a bunch of examples

$$ \begin{aligned}
&\partial_\mu \beta^\mu &&= \partial_0 \beta^0 + \partial_1 \beta^1 + \partial_2 \beta^2 + \partial_3 \beta^3 \\
\\
&A_{ij} \partial_m \beta^m &&= A_{ij}(\partial_0 \beta^0 + \partial_1 \beta^1 + \partial_2 \beta^2 + \partial_3 \beta^3)\\
\\
&\Gamma^m \partial_m \beta^i &&= \Gamma^0 \partial_0 \beta^i + \Gamma^1 \partial_1 \beta^i + \Gamma^2 \partial_2 \beta^i + \Gamma^3 \partial_3 \beta^i \\
\\
&A_{im} A^m_{\;\;j} &&= A_{i0} A^0_{\;\;j} + A_{i1} A^1_{\;\;j} + A_{i2} A^2_{\;\;j} + A_{i3} A^3_{\;\;j} = B_{ij}\\
\\
&\gamma^{mn} D_m A_{ni} - \frac{3}{2} A^m_{\;\;i} \frac{\partial_m \chi}{\chi} &&= \sum_{m=0}^3 \sum_{n=0}^3 \gamma^{mn} D_mA_{ni} - \frac{3}{2\chi} \sum_{m=0}^3 A^m_{\;\;i} \partial_m \chi \\
\\
&A^{im} (3\alpha \frac{\partial_m \chi}{\chi} + 2\partial_m \alpha) &&= \sum_{m=0}^3 A^{im} (3\alpha \frac{\partial_m \chi}{\chi} + 2\partial_m \alpha) = \frac{3 \alpha}{\chi}\sum_{m=0}^3 A^{im} \partial_m \chi + 2 \sum_{m=0}^3 \partial_m \alpha \\
\end{aligned}
$$

Edge cases for raising and lowering, we *can* raise inside a derivative

$$\partial_{\mu} \beta^{\nu} = g^{k\nu} \partial_{\mu} \beta_k$$

And here I introduce some of the slightly more obscure notation

$$ \begin{aligned}
\partial^\mu \beta_\nu &= g^{\mu k} \partial_k \beta_\nu \\
\\
D^\mu \beta_\nu &= g^{\mu k} D_k \beta_\nu \\
\\
A_{ij,}^{\;\;\;\;k} &= \partial^k A_{ij} = g^{mk} \partial_m A_{ij}\\
\end{aligned}$$

One other fun thing is that derivatives are often juggled like they are variables, so keep an eye out for that

$$(\partial_k + 2) A_{ij} = \partial_k A_{ij} + 2A_{ij}$$
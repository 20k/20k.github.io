#ifndef RANDOM_HPP_INCLUDED
#define RANDOM_HPP_INCLUDED

#include <stdint.h>
#include <array>
#include <bit>

///https://en.wikipedia.org/wiki/Xorshift
///todo: C++ify this
inline
uint64_t rol64(uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}

struct xoshiro256ss_state
{
	std::array<uint64_t, 4> s = {};
};

inline
uint64_t xoshiro256ss(xoshiro256ss_state& state)
{
	std::array<uint64_t, 4>& s = state.s;
	uint64_t const result = rol64(s[1] * 5, 7) * 9;
	uint64_t const t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;
	s[3] = rol64(s[3], 45);

	return result;
}

struct splitmix64_state
{
	uint64_t s;
};

inline
uint64_t splitmix64(struct splitmix64_state *state)
{
	uint64_t result = (state->s += 0x9E3779B97f4A7C15);
	result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
	result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
	return result ^ (result >> 31);
}

inline
xoshiro256ss_state xoshiro256ss_init(uint64_t seed)
{
	splitmix64_state smstate = {seed};

	uint64_t s0 = splitmix64(&smstate);
	uint64_t s1 = splitmix64(&smstate);
	uint64_t s2 = splitmix64(&smstate);
	uint64_t s3 = splitmix64(&smstate);

    xoshiro256ss_state state;

	state.s[0] = s0;
	state.s[1] = s1;
	state.s[2] = s2;
	state.s[3] = s3;

	return state;
}

///returns [0, 1]
inline
double uint64_to_double(uint64_t v)
{
    uint64_t up = (v & ((1ull << 52) - 1)) | 0x3FF0000000000000;

    return std::bit_cast<double>(up) - 1.0;
}

#endif // RANDOM_HPP_INCLUDED

#ifndef ACCRETION_DISK_HPP_INCLUDED
#define ACCRETION_DISK_HPP_INCLUDED

#include <array>

struct accretion_segment
{
    float r_start = 0;
    float r_end = 0;

    float F_start = 0;
    float F_end = 0;
};

struct accretion_disk
{
    std::array<accretion_segment, 5> segments;
};

accretion_disk make_for_kerr(float M, float a);

#endif // ACCRETION_DISK_HPP_INCLUDED

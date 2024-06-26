#ifndef ACCRETION_DISK_HPP_INCLUDED
#define ACCRETION_DISK_HPP_INCLUDED

#include <array>
#include <SFML/Graphics.hpp>
#include <vec/tensor.hpp>

struct accretion_disk
{
    std::vector<tensor<float, 3>> brightness;
    std::vector<float> temperature;
};

accretion_disk make_accretion_disk_kerr(float M, float a);

#endif // ACCRETION_DISK_HPP_INCLUDED

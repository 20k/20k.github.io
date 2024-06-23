#ifndef ACCRETION_DISK_HPP_INCLUDED
#define ACCRETION_DISK_HPP_INCLUDED

#include <array>
#include <SFML/Graphics.hpp>

struct accretion_disk
{
    sf::Image normalised_brightness;
};

accretion_disk make_accretion_disk_kerr(float M, float a);

#endif // ACCRETION_DISK_HPP_INCLUDED

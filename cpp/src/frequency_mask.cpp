#include "frequency_mask.hpp"

#include "matrix8.hpp"

#include <stdexcept>

namespace jpeg {

namespace {

std::size_t checked_cutoff(std::size_t cutoff, std::size_t maximum)
{
    if (cutoff == 0 || cutoff > maximum) {
        throw std::invalid_argument{"fréquence de coupure hors de [1, "
                                    + std::to_string(maximum) + "] : "
                                    + std::to_string(cutoff)};
    }
    return cutoff;
}

} // namespace

SquareMask::SquareMask(std::size_t cutoff)
    : cutoff_{checked_cutoff(cutoff, block_size)}
{}

bool SquareMask::keeps(std::size_t k, std::size_t l) const
{
    return k < cutoff_ && l < cutoff_;
}

std::string SquareMask::name() const
{
    return "carré F = " + std::to_string(cutoff_);
}

TriangleMask::TriangleMask(std::size_t cutoff)
    : cutoff_{checked_cutoff(cutoff, 2 * block_size - 1)}
{}

bool TriangleMask::keeps(std::size_t k, std::size_t l) const
{
    return k + l < cutoff_;
}

std::string TriangleMask::name() const
{
    return "triangle F = " + std::to_string(cutoff_);
}

} // namespace jpeg

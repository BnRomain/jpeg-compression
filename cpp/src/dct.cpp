#include "dct.hpp"

#include <cmath>
#include <numbers>

namespace jpeg {

namespace {

Matrix8 dct_basis()
{
    Matrix8 p{};
    for (std::size_t k{0}; k < block_size; ++k) {
        const double c{k == 0 ? 1.0 / std::sqrt(2.0) : 1.0};
        for (std::size_t i{0}; i < block_size; ++i) {
            const double angle{(2.0 * static_cast<double>(i) + 1.0)
                               * static_cast<double>(k) * std::numbers::pi / 16.0};
            p(k, i) = 0.5 * c * std::cos(angle);
        }
    }
    return p;
}

} // namespace

Dct::Dct()
    : p_{dct_basis()},
      p_transposed_{p_.transposed()}
{}

const Matrix8& Dct::basis() const noexcept
{
    return p_;
}

Matrix8 Dct::forward(const Matrix8& block) const
{
    return p_ * block * p_transposed_;
}

Matrix8 Dct::inverse(const Matrix8& coefficients) const
{
    return p_transposed_ * coefficients * p_;
}

} // namespace jpeg

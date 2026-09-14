#include "matrix8.hpp"

namespace jpeg {

Matrix8::Matrix8(double value)
{
    values_.fill(value);
}

double& Matrix8::operator()(std::size_t row, std::size_t col)
{
    return values_[row * block_size + col];
}

const double& Matrix8::operator()(std::size_t row, std::size_t col) const
{
    return values_[row * block_size + col];
}

Matrix8 Matrix8::transposed() const
{
    Matrix8 result{};
    for (std::size_t i{0}; i < block_size; ++i) {
        for (std::size_t j{0}; j < block_size; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

Matrix8 operator*(const Matrix8& lhs, const Matrix8& rhs)
{
    Matrix8 result{};
    // Loop order i, k, j: row k of rhs is traversed in one go, which is more
    // cache-friendly than the naive order i, j, k.
    for (std::size_t i{0}; i < block_size; ++i) {
        for (std::size_t k{0}; k < block_size; ++k) {
            const double factor{lhs(i, k)};
            for (std::size_t j{0}; j < block_size; ++j) {
                result(i, j) += factor * rhs(k, j);
            }
        }
    }
    return result;
}

} // namespace jpeg

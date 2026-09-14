#ifndef JPEG_MATRIX8_HPP
#define JPEG_MATRIX8_HPP

#include <array>
#include <cstddef>

namespace jpeg {

// The JPEG algorithm works on 8 x 8 pixel blocks.
constexpr std::size_t block_size{8};

// 8 x 8 square matrix of reals: a block of pixels, the DCT change-of-basis
// matrix or a block of frequency coefficients.
//
// The 64 coefficients are stored row by row in a std::array. Its size is fixed
// at compile time: unlike std::vector, no dynamic allocation happens, which
// matters since several matrices are created for each of the thousands of
// blocks of an image.
// std::array knows how to copy and destroy itself: no special member function
// is written (rule of zero).
class Matrix8 {
public:
    Matrix8() = default;                 // all coefficients are 0
    explicit Matrix8(double value);      // all coefficients are equal to value

    // Access without bounds checking, like operator[] in lab 4 of the course:
    // one overload to write to a mutable object, one to read a const object.
    double& operator()(std::size_t row, std::size_t col);
    const double& operator()(std::size_t row, std::size_t col) const;

    Matrix8 transposed() const;

private:
    std::array<double, block_size * block_size> values_{};
};

// Matrix product, the equivalent of the NumPy @ operator.
Matrix8 operator*(const Matrix8& lhs, const Matrix8& rhs);

} // namespace jpeg

#endif

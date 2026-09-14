#ifndef JPEG_DCT_HPP
#define JPEG_DCT_HPP

#include "matrix8.hpp"

namespace jpeg {

// Two-dimensional discrete cosine transform (DCT-II) of an 8 x 8 block.
//
// For a block M of intensities centered in [-128, 127]:
//   D_{k,l} = 1/4 C_k C_l  sum_{i,j} M_{i,j} cos((2i+1) k pi / 16) cos((2j+1) l pi / 16)
// with C_0 = 1/sqrt(2) and C_k = 1 for k > 0.
//
// This formula is an orthonormal change of basis, written
//   D = P M P^T     (to the frequency domain: compression)
//   M = P^T D P     (back to intensities: decompression)
// where P_{k,i} = (C_k / 2) cos((2i+1) k pi / 16). Since P is orthogonal, its
// inverse is its transpose: no matrix inversion is needed.
//
// P and P^T are computed once, in the constructor, then reused for every block
// (the Python version passed P as an argument for the same reason).
class Dct {
public:
    Dct();

    const Matrix8& basis() const noexcept;               // the matrix P
    Matrix8 forward(const Matrix8& block) const;         // D = P M P^T
    Matrix8 inverse(const Matrix8& coefficients) const;  // M = P^T D P

private:
    // The declaration order sets the initialization order: p_ must exist
    // before p_transposed_ is computed from it.
    Matrix8 p_;
    Matrix8 p_transposed_;
};

} // namespace jpeg

#endif

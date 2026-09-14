#ifndef JPEG_QUANTIZATION_TABLE_HPP
#define JPEG_QUANTIZATION_TABLE_HPP

#include "matrix8.hpp"

namespace jpeg {

// Quantization matrix Q: each DCT coefficient D_{k,l} is divided by Q_{k,l},
// then truncated toward zero. The larger Q_{k,l}, the more likely the
// coefficient is zeroed.
//
// Invariant: all divisors are >= 1.
//  - a zero or negative divisor makes no sense for a quantization;
//  - since |D_{k,l}| <= 8 * 128 = 1024, a divisor >= 1 guarantees that each
//    quantized coefficient fits in a std::int16_t, the type stored in the CSR
//    matrix (like .astype(np.int16) in Python).
// The constructor checks the invariant and no method modifies the table.
class QuantizationTable {
public:
    // Throws std::invalid_argument if a divisor is < 1.
    explicit QuantizationTable(const Matrix8& divisors);

    // Tables studied in the MAM3 project.
    static QuantizationTable standard();            // JPEG standard, known as psychovisual
    static QuantizationTable uniform(double value); // same divisor everywhere
    static QuantizationTable low_frequencies();     // small divisors at the top left
    static QuantizationTable high_frequencies();    // small divisors at the bottom right

    // Quality factor alpha of the report: returns the table alpha * Q.
    // Throws std::invalid_argument if alpha <= 0 or if alpha * Q breaks the invariant.
    QuantizationTable scaled(double alpha) const;

    double operator()(std::size_t k, std::size_t l) const;
    const Matrix8& divisors() const noexcept;

private:
    Matrix8 divisors_;
};

} // namespace jpeg

#endif

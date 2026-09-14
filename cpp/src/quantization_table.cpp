#include "quantization_table.hpp"

#include <array>
#include <stdexcept>

namespace jpeg {

namespace {

using Rows = std::array<std::array<double, block_size>, block_size>;

// JPEG standard matrix as given in the MAM3 assignment and in
// jpeg_compression.py. The standard has 14 at position (1, 2); the 13 of the
// Python version is kept so that both programs produce exactly the same
// coefficients and can be compared.
const Rows standard_rows{{
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 13, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99},
}};

Matrix8 checked_divisors(const Matrix8& divisors)
{
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            // Written !(d >= 1) rather than d < 1 to also reject NaN.
            if (!(divisors(k, l) >= 1.0)) {
                throw std::invalid_argument{"quantization divisors must be >= 1"};
            }
        }
    }
    return divisors;
}

} // namespace

QuantizationTable::QuantizationTable(const Matrix8& divisors)
    : divisors_{checked_divisors(divisors)}
{}

QuantizationTable QuantizationTable::standard()
{
    Matrix8 divisors{};
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            divisors(k, l) = standard_rows[k][l];
        }
    }
    return QuantizationTable{divisors};
}

QuantizationTable QuantizationTable::uniform(double value)
{
    return QuantizationTable{Matrix8{value}};
}

QuantizationTable QuantizationTable::low_frequencies()
{
    // Q_filtre_HF matrix of the MAM3 Python script: divisor 1 for k + l <= 2,
    // 10 on the diagonal k + l = 3 and 1000 elsewhere. Only the very first
    // frequencies survive: this is a low-pass filter.
    Matrix8 divisors{1000.0};
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            if (k + l <= 2) {
                divisors(k, l) = 1.0;
            } else if (k + l == 3) {
                divisors(k, l) = 10.0;
            }
        }
    }
    return QuantizationTable{divisors};
}

QuantizationTable QuantizationTable::high_frequencies()
{
    // Q_special matrix of the MAM3 Python script: divisor 1000 on the square of
    // low frequencies (k, l < 5), except the DC component D_{0,0} kept at 16,
    // and 1 elsewhere. Mostly the high frequencies (edges) are kept.
    Matrix8 divisors{1.0};
    for (std::size_t k{0}; k < 5; ++k) {
        for (std::size_t l{0}; l < 5; ++l) {
            divisors(k, l) = 1000.0;
        }
    }
    divisors(0, 0) = 16.0;
    return QuantizationTable{divisors};
}

QuantizationTable QuantizationTable::scaled(double alpha) const
{
    if (!(alpha > 0.0)) {
        throw std::invalid_argument{"the alpha factor must be strictly positive"};
    }
    Matrix8 divisors{divisors_};
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            divisors(k, l) *= alpha;
        }
    }
    // The constructor checks the invariant again: alpha < 1 can produce a divisor < 1.
    return QuantizationTable{divisors};
}

double QuantizationTable::operator()(std::size_t k, std::size_t l) const
{
    return divisors_(k, l);
}

const Matrix8& QuantizationTable::divisors() const noexcept
{
    return divisors_;
}

} // namespace jpeg

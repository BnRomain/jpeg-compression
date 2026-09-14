#ifndef JPEG_FREQUENCY_MASK_HPP
#define JPEG_FREQUENCY_MASK_HPP

#include <cstddef>
#include <string>

namespace jpeg {

// Abstract interface: tells whether the frequency coefficient (k, l) of a block
// is kept. Rejected coefficients are set to zero after quantization, which
// filters out the high frequencies (and part of the noise).
//
// The compression only knows this interface: it receives a
// const FrequencyMask& and calls keeps() without knowing which strategy is
// behind it (same idea as ScalarFunction and midpoint in lab 6 of the course).
// Adding a new mask shape therefore does not change the compression.
class FrequencyMask {
public:
    virtual bool keeps(std::size_t k, std::size_t l) const = 0;
    virtual std::string name() const = 0;

    // Virtual destructor: destroying a derived object through the base stays correct.
    virtual ~FrequencyMask() = default;
};

// Square truncation of the Python version: D[F:, :] = 0 and D[:, F:] = 0.
// Keeps the coefficients with k < F and l < F.
// Invariant: 1 <= F <= 8 (F = 8 removes nothing).
class SquareMask : public FrequencyMask {
public:
    explicit SquareMask(std::size_t cutoff);   // throws std::invalid_argument

    bool keeps(std::size_t k, std::size_t l) const override;
    std::string name() const override;

private:
    std::size_t cutoff_;
};

// Triangular truncation of the MAM3 assignment: zeroes the coefficients with
// k + l >= F, F being the cutoff frequency.
// Invariant: 1 <= F <= 15 (k + l is at most 14, so F = 15 removes nothing).
class TriangleMask : public FrequencyMask {
public:
    explicit TriangleMask(std::size_t cutoff);   // throws std::invalid_argument

    bool keeps(std::size_t k, std::size_t l) const override;
    std::string name() const override;

private:
    std::size_t cutoff_;
};

} // namespace jpeg

#endif

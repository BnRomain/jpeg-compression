#include "codec.hpp"

#include "dct.hpp"
#include "matrix8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace jpeg {

namespace {

// Centering offset: the intensities [0, 255] become [-128, 127].
constexpr double center{128.0};

Matrix8 extract_centered_block(const Image& image, std::size_t top, std::size_t left,
                               std::size_t channel)
{
    Matrix8 block{};
    for (std::size_t i{0}; i < block_size; ++i) {
        for (std::size_t j{0}; j < block_size; ++j) {
            block(i, j) = image(top + i, left + j, channel) - center;
        }
    }
    return block;
}

// Quantizes a coefficient: division by Q, truncation toward zero, then
// zeroing if it is below the threshold or rejected by the mask.
std::int16_t quantize(double coefficient, double divisor, int threshold, bool kept)
{
    const double quantized{std::trunc(coefficient / divisor)};
    if (!kept || std::abs(quantized) < threshold) {
        return 0;
    }
    // The QuantizationTable invariant (divisors >= 1) guarantees |quantized| <= 1024.
    return static_cast<std::int16_t>(quantized);
}

} // namespace

CompressedImage compress(const Image& image, const QuantizationTable& table,
                         int threshold, const FrequencyMask& mask)
{
    if (threshold < 0) {
        throw std::invalid_argument{"the threshold must be non-negative"};
    }

    const Image cropped{image.cropped_to_blocks()};
    const std::size_t width{cropped.width()};
    const std::size_t height{cropped.height()};
    const Dct dct{};

    std::vector<SparseMatrix> channels;
    channels.reserve(Image::channels);
    for (std::size_t channel{0}; channel < Image::channels; ++channel) {
        // Dense height x width plane of the quantized coefficients of the channel,
        // the equivalent of img_compressed[:, :, channel] in Python.
        std::vector<std::int16_t> plane(width * height, 0);
        for (std::size_t top{0}; top < height; top += block_size) {
            for (std::size_t left{0}; left < width; left += block_size) {
                const Matrix8 coefficients{dct.forward(extract_centered_block(cropped, top, left, channel))};
                for (std::size_t k{0}; k < block_size; ++k) {
                    for (std::size_t l{0}; l < block_size; ++l) {
                        plane[(top + k) * width + left + l] =
                            quantize(coefficients(k, l), table(k, l), threshold, mask.keeps(k, l));
                    }
                }
            }
        }
        // CSR conversion: only the non-zero coefficients are kept.
        channels.emplace_back(height, width, plane);
    }
    return CompressedImage{width, height, table, std::move(channels)};
}

Image decompress(const CompressedImage& compressed)
{
    const std::size_t width{compressed.width()};
    const std::size_t height{compressed.height()};
    const QuantizationTable& table{compressed.table()};
    const Dct dct{};

    Image result{width, height};
    for (std::size_t channel{0}; channel < Image::channels; ++channel) {
        const std::vector<std::int16_t> plane{compressed.channel(channel).to_dense()};
        for (std::size_t top{0}; top < height; top += block_size) {
            for (std::size_t left{0}; left < width; left += block_size) {
                Matrix8 coefficients{};
                for (std::size_t k{0}; k < block_size; ++k) {
                    for (std::size_t l{0}; l < block_size; ++l) {
                        coefficients(k, l) = plane[(top + k) * width + left + l] * table(k, l);
                    }
                }
                const Matrix8 block{dct.inverse(coefficients)};
                for (std::size_t i{0}; i < block_size; ++i) {
                    for (std::size_t j{0}; j < block_size; ++j) {
                        result(top + i, left + j, channel) = std::clamp(block(i, j) + center, 0.0, 255.0);
                    }
                }
            }
        }
    }
    return result;
}

} // namespace jpeg

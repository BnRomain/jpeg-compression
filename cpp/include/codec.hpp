#ifndef JPEG_CODEC_HPP
#define JPEG_CODEC_HPP

#include "compressed_image.hpp"
#include "frequency_mask.hpp"
#include "image.hpp"
#include "quantization_table.hpp"

namespace jpeg {

// Compresses the image channel by channel, one 8 x 8 block at a time
// (compression() in the Python version):
//   1. cropping to multiples of 8, then centering: [0, 255] -> [-128, 127];
//   2. DCT: D = P M P^T;
//   3. quantization: D_{k,l} / Q_{k,l} truncated toward zero (np.trunc);
//   4. threshold: coefficients with an absolute value < threshold are zeroed;
//   5. mask: the high frequencies rejected by `mask` are zeroed;
//   6. the coefficients of each channel are stored in a CSR matrix.
//
// The arguments are only read: const references, no copies.
// `mask` is a reference to the FrequencyMask interface: any truncation
// strategy works without changing this function.
// Throws std::invalid_argument if threshold < 0 or if the image is too small.
CompressedImage compress(const Image& image, const QuantizationTable& table,
                         int threshold, const FrequencyMask& mask);

// Decompression (decompression() in the Python version), block by block:
// element-wise multiplication by Q, inverse DCT M = P^T D P, un-centering
// (+128), then clamping of the intensities to [0, 255] (np.clip).
Image decompress(const CompressedImage& compressed);

} // namespace jpeg

#endif

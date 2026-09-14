#ifndef JPEG_METRICS_HPP
#define JPEG_METRICS_HPP

#include "image.hpp"

namespace jpeg {

// Quality metrics between a reference image and its approximation
// (post-processing step of the assignment). Both images must have the same
// dimensions, otherwise std::invalid_argument is thrown.

// Relative error in L2 norm: ||reference - approximation|| / ||reference||,
// computed on the [0, 255] intensities of the three channels.
double relative_l2_error(const Image& reference, const Image& approximation);

// Peak signal-to-noise ratio, in dB: 10 log10(255^2 / mean squared error).
// The higher it is, the closer the approximation; it is +infinity when both
// images are identical.
double psnr(const Image& reference, const Image& approximation);

} // namespace jpeg

#endif

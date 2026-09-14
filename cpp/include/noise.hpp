#ifndef JPEG_NOISE_HPP
#define JPEG_NOISE_HPP

#include "image.hpp"

namespace jpeg {

// "Salt-and-pepper" noise: with probability `probability`, each pixel is
// replaced by a black (0) or white (255) pixel, both cases being equally
// likely. This noise is made of very fast variations, hence of high
// frequencies, which the compression attenuates (low-pass effect studied in
// the report).
//
// The image is modified in place, hence the non-const reference.
// The pseudo-random generator is seeded with `seed`: experiments are
// reproducible.
// Throws std::invalid_argument if probability is not in [0, 1].
void add_salt_and_pepper(Image& image, double probability, unsigned seed = 42);

} // namespace jpeg

#endif

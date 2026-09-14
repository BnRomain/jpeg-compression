#include "noise.hpp"

#include <random>
#include <stdexcept>

namespace jpeg {

void add_salt_and_pepper(Image& image, double probability, unsigned seed)
{
    if (!(probability >= 0.0 && probability <= 1.0)) {
        throw std::invalid_argument{"the noise probability must be in [0, 1]"};
    }

    std::mt19937 generator{seed};
    std::uniform_real_distribution<double> uniform{0.0, 1.0};
    for (std::size_t row{0}; row < image.height(); ++row) {
        for (std::size_t col{0}; col < image.width(); ++col) {
            if (uniform(generator) < probability) {
                const double value{uniform(generator) < 0.5 ? 0.0 : 255.0};
                for (std::size_t channel{0}; channel < Image::channels; ++channel) {
                    image(row, col, channel) = value;
                }
            }
        }
    }
}

} // namespace jpeg

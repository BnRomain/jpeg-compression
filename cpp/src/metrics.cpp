#include "metrics.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace jpeg {

namespace {

void check_same_dimensions(const Image& a, const Image& b)
{
    if (a.width() != b.width() || a.height() != b.height()) {
        throw std::invalid_argument{"les deux images n'ont pas les mêmes dimensions"};
    }
}

// Somme des carrés des écarts pixel à pixel sur les trois canaux.
double squared_distance(const Image& a, const Image& b)
{
    double sum{};
    for (std::size_t row{0}; row < a.height(); ++row) {
        for (std::size_t col{0}; col < a.width(); ++col) {
            for (std::size_t channel{0}; channel < Image::channels; ++channel) {
                const double difference{a(row, col, channel) - b(row, col, channel)};
                sum += difference * difference;
            }
        }
    }
    return sum;
}

double squared_norm(const Image& image)
{
    double sum{};
    for (std::size_t row{0}; row < image.height(); ++row) {
        for (std::size_t col{0}; col < image.width(); ++col) {
            for (std::size_t channel{0}; channel < Image::channels; ++channel) {
                sum += image(row, col, channel) * image(row, col, channel);
            }
        }
    }
    return sum;
}

} // namespace

double relative_l2_error(const Image& reference, const Image& approximation)
{
    check_same_dimensions(reference, approximation);
    const double error{std::sqrt(squared_distance(reference, approximation))};
    const double norm{std::sqrt(squared_norm(reference))};
    if (norm == 0.0) {
        // Référence entièrement noire : l'erreur relative n'a de sens que si elle est nulle.
        return error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    }
    return error / norm;
}

double psnr(const Image& reference, const Image& approximation)
{
    check_same_dimensions(reference, approximation);
    const double mse{squared_distance(reference, approximation) / static_cast<double>(reference.size())};
    if (mse == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

} // namespace jpeg

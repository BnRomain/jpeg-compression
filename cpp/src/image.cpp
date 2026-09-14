#include "image.hpp"

#include "matrix8.hpp"

#include <stdexcept>
#include <string>

namespace jpeg {

namespace {

std::size_t checked_dimension(std::size_t dimension)
{
    if (dimension == 0 || dimension > Image::max_dimension) {
        throw std::invalid_argument{"an image dimension must be between 1 and "
                                    + std::to_string(Image::max_dimension)};
    }
    return dimension;
}

void check_bounds(std::size_t row, std::size_t col, std::size_t channel,
                  std::size_t width, std::size_t height)
{
    if (row >= height || col >= width || channel >= Image::channels) {
        throw std::out_of_range{"pixel outside the image"};
    }
}

} // namespace

Image::Image(std::size_t width, std::size_t height, double value)
    : width_{checked_dimension(width)},
      height_{checked_dimension(height)},
      pixels_(width_ * height_ * channels, value)
{}

std::size_t Image::width() const noexcept
{
    return width_;
}

std::size_t Image::height() const noexcept
{
    return height_;
}

std::size_t Image::size() const noexcept
{
    return pixels_.size();
}

std::size_t Image::index(std::size_t row, std::size_t col, std::size_t channel) const noexcept
{
    return (row * width_ + col) * channels + channel;
}

double& Image::operator()(std::size_t row, std::size_t col, std::size_t channel)
{
    return pixels_[index(row, col, channel)];
}

const double& Image::operator()(std::size_t row, std::size_t col, std::size_t channel) const
{
    return pixels_[index(row, col, channel)];
}

double& Image::at(std::size_t row, std::size_t col, std::size_t channel)
{
    check_bounds(row, col, channel, width_, height_);
    return pixels_[index(row, col, channel)];
}

const double& Image::at(std::size_t row, std::size_t col, std::size_t channel) const
{
    check_bounds(row, col, channel, width_, height_);
    return pixels_[index(row, col, channel)];
}

Image Image::cropped_to_blocks() const
{
    // The constructor rejects a zero dimension: an image smaller than 8 pixels
    // on a side therefore throws instead of producing an empty image.
    Image result{width_ - width_ % block_size, height_ - height_ % block_size};
    for (std::size_t row{0}; row < result.height_; ++row) {
        for (std::size_t col{0}; col < result.width_; ++col) {
            for (std::size_t channel{0}; channel < channels; ++channel) {
                result(row, col, channel) = (*this)(row, col, channel);
            }
        }
    }
    return result;
}

} // namespace jpeg

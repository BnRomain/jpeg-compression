#ifndef JPEG_IMAGE_HPP
#define JPEG_IMAGE_HPP

#include <cstddef>
#include <vector>

namespace jpeg {

// RGB color image loaded in memory.
//
// Representation: width x height pixels on 3 channels (red, green, blue), the
// nx x ny x 3 array of the assignment. Intensities are real numbers in
// [0, 255] stored row by row in a single std::vector:
//   index = (row * width + col) * 3 + channel
//
// Invariant: 0 < width, height <= max_dimension and
// pixels_.size() == width * height * 3.
// The constructor establishes it, the representation is private and no method
// changes the size: outside code cannot break the invariant.
//
// Storage is delegated to std::vector, which copies, moves and frees itself:
// no special member function is written (rule of zero).
class Image {
public:
    static constexpr std::size_t channels{3};

    // Largest accepted width or height. stb computes some buffer sizes with
    // int products: with at most 16384 pixels per side,
    // (16384 * 3 + 1) * 16384 < 2^31 and these computations cannot overflow.
    static constexpr std::size_t max_dimension{16384};

    // Throws std::invalid_argument if a dimension is zero or exceeds max_dimension.
    Image(std::size_t width, std::size_t height, double value = 0.0);

    std::size_t width() const noexcept;
    std::size_t height() const noexcept;
    std::size_t size() const noexcept;   // number of values: width * height * 3

    // Fast access without bounds checking, for computation loops.
    double& operator()(std::size_t row, std::size_t col, std::size_t channel);
    const double& operator()(std::size_t row, std::size_t col, std::size_t channel) const;

    // Checked access: throws std::out_of_range outside the image.
    double& at(std::size_t row, std::size_t col, std::size_t channel);
    const double& at(std::size_t row, std::size_t col, std::size_t channel) const;

    // Copy cropped to the largest multiples of 8 (init() step of the Python
    // version). The current image is not modified, hence the const.
    // Throws std::invalid_argument if the image is smaller than 8 pixels on a side.
    Image cropped_to_blocks() const;

private:
    std::size_t index(std::size_t row, std::size_t col, std::size_t channel) const noexcept;

    std::size_t width_;
    std::size_t height_;
    std::vector<double> pixels_;
};

} // namespace jpeg

#endif

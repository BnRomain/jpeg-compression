#include "image_io.hpp"

// stb is a header-only library: its code is only compiled in the file that
// defines the *_IMPLEMENTATION macro, here and nowhere else.
// The third_party folder is passed to g++ with -isystem so that the internal
// warnings of stb are not shown.
//
// Security: only the needed decoders (PNG, JPEG, BMP) are compiled, which
// reduces the attack surface against a malicious file, and stb rejects images
// larger than Image::max_dimension pixels per side.
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_MAX_DIMENSIONS 16384
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// The stb limit and the Image invariant must stay identical.
static_assert(STBI_MAX_DIMENSIONS == jpeg::Image::max_dimension);

namespace jpeg {

namespace {

// Pixel array allocated by stbi_load, which must be released with
// stbi_image_free. Rather than calling stbi_image_free by hand (and forgetting
// it when an exception is thrown), the resource is tied to the lifetime of this
// object: RAII, like GmshSession in the final lab of the course.
class StbPixels {
public:
    explicit StbPixels(const std::string& path);
    ~StbPixels();

    // Two objects must not free the same array: copying is forbidden.
    StbPixels(const StbPixels&) = delete;
    StbPixels& operator=(const StbPixels&) = delete;

    std::size_t width() const noexcept;
    std::size_t height() const noexcept;
    unsigned char operator[](std::size_t index) const;

private:
    int width_{};
    int height_{};
    unsigned char* data_{nullptr};
};

StbPixels::StbPixels(const std::string& path)
{
    int channels_in_file{};
    // The last argument requests 3 channels whatever the file:
    // stb converts grayscale to RGB and drops the alpha channel.
    data_ = stbi_load(path.c_str(), &width_, &height_, &channels_in_file,
                      static_cast<int>(Image::channels));
    if (data_ == nullptr) {
        // If the constructor throws, the destructor is not called; this is not
        // a problem here since nothing was allocated.
        throw std::runtime_error{"cannot read image '" + path + "' ("
                                 + stbi_failure_reason() + ")"};
    }
}

StbPixels::~StbPixels()
{
    stbi_image_free(data_);
}

std::size_t StbPixels::width() const noexcept
{
    return static_cast<std::size_t>(width_);
}

std::size_t StbPixels::height() const noexcept
{
    return static_cast<std::size_t>(height_);
}

unsigned char StbPixels::operator[](std::size_t index) const
{
    return data_[index];
}

} // namespace

Image load_image(const std::string& path)
{
    const StbPixels pixels{path};
    Image image{pixels.width(), pixels.height()};

    // stb also stores the pixels row by row, channel by channel.
    std::size_t index{0};
    for (std::size_t row{0}; row < image.height(); ++row) {
        for (std::size_t col{0}; col < image.width(); ++col) {
            for (std::size_t channel{0}; channel < Image::channels; ++channel) {
                image(row, col, channel) = pixels[index];
                ++index;
            }
        }
    }
    return image;
}   // pixels is destroyed here: stbi_image_free is called automatically

void save_png(const Image& image, const std::string& path)
{
    std::vector<unsigned char> bytes(image.size());
    std::size_t index{0};
    for (std::size_t row{0}; row < image.height(); ++row) {
        for (std::size_t col{0}; col < image.width(); ++col) {
            for (std::size_t channel{0}; channel < Image::channels; ++channel) {
                const double value{std::clamp(image(row, col, channel), 0.0, 255.0)};
                bytes[index] = static_cast<unsigned char>(std::lround(value));
                ++index;
            }
        }
    }

    const int width{static_cast<int>(image.width())};
    const int height{static_cast<int>(image.height())};
    const int channels{static_cast<int>(Image::channels)};
    if (stbi_write_png(path.c_str(), width, height, channels, bytes.data(), width * channels) == 0) {
        throw std::runtime_error{"cannot write image '" + path + "'"};
    }
}

} // namespace jpeg

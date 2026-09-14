#ifndef JPEG_IMAGE_IO_HPP
#define JPEG_IMAGE_IO_HPP

#include "image.hpp"

#include <string>

namespace jpeg {

// Reading and writing image files.
// The whole dependency on the stb library is confined to image_io.cpp: the
// rest of the program only handles the Image class.

// Reads a PNG, JPEG or BMP image (the only compiled formats, at most
// Image::max_dimension pixels per side). A grayscale image is converted to RGB
// and an alpha (transparency) channel is ignored, like the preprocessing of the
// Python version.
// Throws std::runtime_error if the file cannot be read.
Image load_image(const std::string& path);

// Writes the image as PNG. Intensities are clamped to [0, 255], then rounded
// to the nearest integer.
// Throws std::runtime_error if writing fails.
void save_png(const Image& image, const std::string& path);

} // namespace jpeg

#endif

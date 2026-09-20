// Definitions of everything declared in jpeg.hpp, in the same order.

#include "jpeg.hpp"

// stb is a header-only library: its code is only compiled in the file that
// defines the *_IMPLEMENTATION macro, here and nowhere else.
//
// Security: only the needed decoders (PNG, JPEG, BMP) are compiled, which
// reduces the attack surface against a malicious file, and stb rejects images
// larger than Image::max_dimension pixels per side.
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_MAX_DIMENSIONS 16384
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <numbers>
#include <ostream>
#include <random>
#include <stdexcept>
#include <utility>

// The stb limit and the Image invariant must stay identical.
static_assert(STBI_MAX_DIMENSIONS == jpeg::Image::max_dimension);

namespace jpeg {

// ---------------------------------------------------------------------------
// 1. Matrix8
// ---------------------------------------------------------------------------

Matrix8::Matrix8(double value)
{
    values_.fill(value);
}

double& Matrix8::operator()(std::size_t row, std::size_t col)
{
    return values_[row * block_size + col];
}

const double& Matrix8::operator()(std::size_t row, std::size_t col) const
{
    return values_[row * block_size + col];
}

Matrix8 Matrix8::transposed() const
{
    Matrix8 result{};
    for (std::size_t i{0}; i < block_size; ++i) {
        for (std::size_t j{0}; j < block_size; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

Matrix8 operator*(const Matrix8& lhs, const Matrix8& rhs)
{
    Matrix8 result{};
    // Loop order i, k, j: row k of rhs is traversed in one go, which is more
    // cache-friendly than the naive order i, j, k.
    for (std::size_t i{0}; i < block_size; ++i) {
        for (std::size_t k{0}; k < block_size; ++k) {
            const double factor{lhs(i, k)};
            for (std::size_t j{0}; j < block_size; ++j) {
                result(i, j) += factor * rhs(k, j);
            }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// 2. Image
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 3. Dct
// ---------------------------------------------------------------------------

namespace {

Matrix8 dct_basis()
{
    Matrix8 p{};
    for (std::size_t k{0}; k < block_size; ++k) {
        const double c{k == 0 ? 1.0 / std::sqrt(2.0) : 1.0};
        for (std::size_t i{0}; i < block_size; ++i) {
            const double angle{(2.0 * static_cast<double>(i) + 1.0)
                               * static_cast<double>(k) * std::numbers::pi / 16.0};
            p(k, i) = 0.5 * c * std::cos(angle);
        }
    }
    return p;
}

} // namespace

Dct::Dct()
    : p_{dct_basis()},
      p_transposed_{p_.transposed()}
{}

const Matrix8& Dct::basis() const noexcept
{
    return p_;
}

Matrix8 Dct::forward(const Matrix8& block) const
{
    return p_ * block * p_transposed_;
}

Matrix8 Dct::inverse(const Matrix8& coefficients) const
{
    return p_transposed_ * coefficients * p_;
}

// ---------------------------------------------------------------------------
// 4. QuantizationTable
// ---------------------------------------------------------------------------

namespace {

using Rows = std::array<std::array<double, block_size>, block_size>;

// JPEG standard matrix as given in the MAM3 assignment and in
// jpeg_compression.py. The standard has 14 at position (1, 2); the 13 of the
// Python version is kept so that both programs produce exactly the same
// coefficients and can be compared.
const Rows standard_rows{{
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 13, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99},
}};

Matrix8 checked_divisors(const Matrix8& divisors)
{
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            // Written !(d >= 1) rather than d < 1 to also reject NaN.
            if (!(divisors(k, l) >= 1.0)) {
                throw std::invalid_argument{"quantization divisors must be >= 1"};
            }
        }
    }
    return divisors;
}

} // namespace

QuantizationTable::QuantizationTable(const Matrix8& divisors)
    : divisors_{checked_divisors(divisors)}
{}

QuantizationTable QuantizationTable::standard()
{
    Matrix8 divisors{};
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            divisors(k, l) = standard_rows[k][l];
        }
    }
    return QuantizationTable{divisors};
}

QuantizationTable QuantizationTable::uniform(double value)
{
    return QuantizationTable{Matrix8{value}};
}

QuantizationTable QuantizationTable::low_frequencies()
{
    // Q_filtre_HF matrix of the MAM3 Python script: divisor 1 for k + l <= 2,
    // 10 on the diagonal k + l = 3 and 1000 elsewhere. Only the very first
    // frequencies survive: this is a low-pass filter.
    Matrix8 divisors{1000.0};
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            if (k + l <= 2) {
                divisors(k, l) = 1.0;
            } else if (k + l == 3) {
                divisors(k, l) = 10.0;
            }
        }
    }
    return QuantizationTable{divisors};
}

QuantizationTable QuantizationTable::high_frequencies()
{
    // Q_special matrix of the MAM3 Python script: divisor 1000 on the square of
    // low frequencies (k, l < 5), except the DC component D_{0,0} kept at 16,
    // and 1 elsewhere. Mostly the high frequencies (edges) are kept.
    Matrix8 divisors{1.0};
    for (std::size_t k{0}; k < 5; ++k) {
        for (std::size_t l{0}; l < 5; ++l) {
            divisors(k, l) = 1000.0;
        }
    }
    divisors(0, 0) = 16.0;
    return QuantizationTable{divisors};
}

QuantizationTable QuantizationTable::scaled(double alpha) const
{
    if (!(alpha > 0.0)) {
        throw std::invalid_argument{"the alpha factor must be strictly positive"};
    }
    Matrix8 divisors{divisors_};
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            divisors(k, l) *= alpha;
        }
    }
    // The constructor checks the invariant again: alpha < 1 can produce a divisor < 1.
    return QuantizationTable{divisors};
}

double QuantizationTable::operator()(std::size_t k, std::size_t l) const
{
    return divisors_(k, l);
}

const Matrix8& QuantizationTable::divisors() const noexcept
{
    return divisors_;
}

// ---------------------------------------------------------------------------
// 5. SparseMatrix
// ---------------------------------------------------------------------------

namespace {

// Indices are stored on 32 bits: the matrix must remain indexable.
void check_dimensions(std::size_t rows, std::size_t cols)
{
    const auto max_index{static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())};
    if (rows > max_index || cols > max_index || (cols != 0 && rows > max_index / cols)) {
        throw std::invalid_argument{"matrix too large for 32-bit indices"};
    }
}

} // namespace

SparseMatrix::SparseMatrix(std::size_t rows, std::size_t cols, const std::vector<std::int16_t>& dense)
    : rows_{rows},
      cols_{cols}
{
    check_dimensions(rows, cols);
    if (dense.size() != rows * cols) {
        throw std::invalid_argument{"the dense matrix does not have size rows * cols"};
    }

    row_pointers_.reserve(rows + 1);
    row_pointers_.push_back(0);
    for (std::size_t i{0}; i < rows; ++i) {
        for (std::size_t j{0}; j < cols; ++j) {
            const std::int16_t value{dense[i * cols + j]};
            if (value != 0) {
                values_.push_back(value);
                column_indices_.push_back(static_cast<std::int32_t>(j));
            }
        }
        // End of row i = start of row i + 1.
        row_pointers_.push_back(static_cast<std::int32_t>(values_.size()));
    }
}

SparseMatrix::SparseMatrix(std::size_t rows, std::size_t cols,
                           std::vector<std::int16_t> values,
                           std::vector<std::int32_t> column_indices,
                           std::vector<std::int32_t> row_pointers)
    : rows_{rows},
      cols_{cols},
      values_{std::move(values)},
      column_indices_{std::move(column_indices)},
      row_pointers_{std::move(row_pointers)}
{
    check_dimensions(rows, cols);
    check_invariant();
}

void SparseMatrix::check_invariant() const
{
    if (row_pointers_.size() != rows_ + 1 || row_pointers_.front() != 0) {
        throw std::invalid_argument{"CSR: row_pointers must have rows + 1 elements and start at 0"};
    }
    if (values_.size() != column_indices_.size()
        || static_cast<std::size_t>(row_pointers_.back()) != values_.size()) {
        throw std::invalid_argument{"CSR: inconsistent sizes of values, column_indices and row_pointers"};
    }
    // First pass: non-decreasing pointers. They then all stay in [0, nnz],
    // which makes the accesses of the second pass safe.
    for (std::size_t i{0}; i < rows_; ++i) {
        if (row_pointers_[i + 1] < row_pointers_[i]) {
            throw std::invalid_argument{"CSR: row_pointers must be non-decreasing"};
        }
    }
    for (std::size_t i{0}; i < rows_; ++i) {
        const auto begin{static_cast<std::size_t>(row_pointers_[i])};
        const auto end{static_cast<std::size_t>(row_pointers_[i + 1])};
        for (std::size_t p{begin}; p < end; ++p) {
            const std::int32_t col{column_indices_[p]};
            if (col < 0 || static_cast<std::size_t>(col) >= cols_) {
                throw std::invalid_argument{"CSR: column index outside the matrix"};
            }
            if (p > begin && col <= column_indices_[p - 1]) {
                throw std::invalid_argument{"CSR: columns not strictly increasing within a row"};
            }
            if (values_[p] == 0) {
                throw std::invalid_argument{"CSR: a stored value is zero"};
            }
        }
    }
}

std::size_t SparseMatrix::rows() const noexcept
{
    return rows_;
}

std::size_t SparseMatrix::cols() const noexcept
{
    return cols_;
}

std::size_t SparseMatrix::nnz() const noexcept
{
    return values_.size();
}

std::int16_t SparseMatrix::at(std::size_t row, std::size_t col) const
{
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range{"coefficient outside the CSR matrix"};
    }
    const auto begin{static_cast<std::size_t>(row_pointers_[row])};
    const auto end{static_cast<std::size_t>(row_pointers_[row + 1])};
    for (std::size_t p{begin}; p < end; ++p) {
        if (static_cast<std::size_t>(column_indices_[p]) == col) {
            return values_[p];
        }
    }
    return 0;
}

std::vector<std::int16_t> SparseMatrix::to_dense() const
{
    std::vector<std::int16_t> dense(rows_ * cols_, 0);
    for (std::size_t i{0}; i < rows_; ++i) {
        const auto begin{static_cast<std::size_t>(row_pointers_[i])};
        const auto end{static_cast<std::size_t>(row_pointers_[i + 1])};
        for (std::size_t p{begin}; p < end; ++p) {
            dense[i * cols_ + static_cast<std::size_t>(column_indices_[p])] = values_[p];
        }
    }
    return dense;
}

std::size_t SparseMatrix::storage_bytes() const noexcept
{
    return values_.size() * sizeof(std::int16_t)
           + column_indices_.size() * sizeof(std::int32_t)
           + row_pointers_.size() * sizeof(std::int32_t);
}

const std::vector<std::int16_t>& SparseMatrix::values() const noexcept
{
    return values_;
}

const std::vector<std::int32_t>& SparseMatrix::column_indices() const noexcept
{
    return column_indices_;
}

const std::vector<std::int32_t>& SparseMatrix::row_pointers() const noexcept
{
    return row_pointers_;
}

// ---------------------------------------------------------------------------
// 6. FrequencyMask
// ---------------------------------------------------------------------------

namespace {

std::size_t checked_cutoff(std::size_t cutoff, std::size_t maximum)
{
    if (cutoff == 0 || cutoff > maximum) {
        throw std::invalid_argument{"cutoff frequency outside [1, "
                                    + std::to_string(maximum) + "]: "
                                    + std::to_string(cutoff)};
    }
    return cutoff;
}

} // namespace

SquareMask::SquareMask(std::size_t cutoff)
    : cutoff_{checked_cutoff(cutoff, block_size)}
{}

bool SquareMask::keeps(std::size_t k, std::size_t l) const
{
    return k < cutoff_ && l < cutoff_;
}

std::string SquareMask::name() const
{
    return "square F = " + std::to_string(cutoff_);
}

TriangleMask::TriangleMask(std::size_t cutoff)
    : cutoff_{checked_cutoff(cutoff, 2 * block_size - 1)}
{}

bool TriangleMask::keeps(std::size_t k, std::size_t l) const
{
    return k + l < cutoff_;
}

std::string TriangleMask::name() const
{
    return "triangle F = " + std::to_string(cutoff_);
}

// ---------------------------------------------------------------------------
// 7. CompressedImage and the .csr file
// ---------------------------------------------------------------------------

CompressedImage::CompressedImage(std::size_t width, std::size_t height,
                                 const QuantizationTable& table,
                                 std::vector<SparseMatrix> channels)
    : width_{width},
      height_{height},
      table_{table},
      channels_{std::move(channels)}
{
    if (width_ == 0 || height_ == 0 || width_ % block_size != 0 || height_ % block_size != 0
        || width_ > Image::max_dimension || height_ > Image::max_dimension) {
        throw std::invalid_argument{"compressed dimensions must be non-zero multiples of 8, "
                                    "at most " + std::to_string(Image::max_dimension) + " pixels"};
    }
    if (channels_.size() != Image::channels) {
        throw std::invalid_argument{"a compressed image has exactly 3 channels"};
    }
    for (const SparseMatrix& channel : channels_) {
        if (channel.rows() != height_ || channel.cols() != width_) {
            throw std::invalid_argument{"a CSR channel does not have the dimensions of the image"};
        }
    }
}

std::size_t CompressedImage::width() const noexcept
{
    return width_;
}

std::size_t CompressedImage::height() const noexcept
{
    return height_;
}

const QuantizationTable& CompressedImage::table() const noexcept
{
    return table_;
}

const SparseMatrix& CompressedImage::channel(std::size_t index) const
{
    return channels_.at(index);
}

std::size_t CompressedImage::nnz() const noexcept
{
    std::size_t total{0};
    for (const SparseMatrix& channel : channels_) {
        total += channel.nnz();
    }
    return total;
}

std::size_t CompressedImage::coefficient_count() const noexcept
{
    return width_ * height_ * Image::channels;
}

double CompressedImage::conservation_rate() const noexcept
{
    return static_cast<double>(nnz()) / static_cast<double>(coefficient_count());
}

std::size_t CompressedImage::storage_bytes() const noexcept
{
    std::size_t total{0};
    for (const SparseMatrix& channel : channels_) {
        total += channel.storage_bytes();
    }
    return total;
}

namespace {

const char signature[4]{'J', 'C', 'S', 'R'};

// Binary output: reinterpret_cast presents the object as a sequence of bytes
// (char) that are written as is. The compiler picks the write and read
// overloads from the type of the argument.
void write(std::ostream& out, std::uint32_t value)
{
    out.write(reinterpret_cast<const char*>(&value), sizeof value);
}

void write(std::ostream& out, double value)
{
    out.write(reinterpret_cast<const char*>(&value), sizeof value);
}

void write(std::ostream& out, const std::vector<std::int32_t>& values)
{
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(std::int32_t)));
}

void write(std::ostream& out, const std::vector<std::int16_t>& values)
{
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(std::int16_t)));
}

void read(std::istream& in, std::uint32_t& value)
{
    in.read(reinterpret_cast<char*>(&value), sizeof value);
}

void read(std::istream& in, double& value)
{
    in.read(reinterpret_cast<char*>(&value), sizeof value);
}

// The next two overloads read values.size() elements: the vector must be
// sized before the call.
void read(std::istream& in, std::vector<std::int32_t>& values)
{
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(std::int32_t)));
}

void read(std::istream& in, std::vector<std::int16_t>& values)
{
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(std::int16_t)));
}

void check_stream(const std::ios& stream, const std::string& path)
{
    if (!stream) {
        throw std::runtime_error{"error while reading or writing file '" + path + "'"};
    }
}

} // namespace

void save_compressed(const CompressedImage& image, const std::string& path)
{
    // The file is closed by the std::ofstream destructor (RAII), even if an
    // exception interrupts the function.
    std::ofstream out{path, std::ios::binary};
    if (!out) {
        throw std::runtime_error{"cannot create file '" + path + "'"};
    }

    out.write(signature, sizeof signature);
    write(out, static_cast<std::uint32_t>(image.width()));
    write(out, static_cast<std::uint32_t>(image.height()));
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            write(out, image.table()(k, l));
        }
    }
    for (std::size_t c{0}; c < Image::channels; ++c) {
        const SparseMatrix& channel{image.channel(c)};
        write(out, static_cast<std::uint32_t>(channel.nnz()));
        write(out, channel.row_pointers());
        write(out, channel.column_indices());
        write(out, channel.values());
    }
    check_stream(out, path);
}

CompressedImage load_compressed(const std::string& path)
{
    std::ifstream in{path, std::ios::binary};
    if (!in) {
        throw std::runtime_error{"cannot open file '" + path + "'"};
    }

    char header[4]{};
    in.read(header, sizeof header);
    if (!in || !std::equal(header, header + sizeof header, signature)) {
        throw std::runtime_error{"'" + path + "' is not a .csr file"};
    }

    std::uint32_t width{};
    std::uint32_t height{};
    read(in, width);
    read(in, height);
    Matrix8 divisors{};
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            read(in, divisors(k, l));
        }
    }
    check_stream(in, path);

    // Checked before any allocation: a corrupted file must not trigger the
    // reservation of several gigabytes.
    if (width > Image::max_dimension || height > Image::max_dimension) {
        throw std::invalid_argument{"invalid dimensions in '" + path + "'"};
    }
    const std::size_t coefficient_count{static_cast<std::size_t>(width) * height};

    std::vector<SparseMatrix> channels;
    for (std::size_t c{0}; c < Image::channels; ++c) {
        std::uint32_t nnz{};
        read(in, nnz);
        check_stream(in, path);
        if (nnz > coefficient_count) {
            throw std::invalid_argument{"invalid coefficient count in '" + path + "'"};
        }

        std::vector<std::int32_t> row_pointers(static_cast<std::size_t>(height) + 1);
        std::vector<std::int32_t> column_indices(nnz);
        std::vector<std::int16_t> values(nnz);
        read(in, row_pointers);
        read(in, column_indices);
        read(in, values);
        check_stream(in, path);

        // The SparseMatrix constructor checks the CSR invariant; the vectors
        // are moved, not copied.
        channels.emplace_back(height, width, std::move(values),
                              std::move(column_indices), std::move(row_pointers));
    }
    // QuantizationTable and CompressedImage check their own invariants in turn.
    return CompressedImage{width, height, QuantizationTable{divisors}, std::move(channels)};
}

// ---------------------------------------------------------------------------
// 8. Compression and decompression
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 9. Reading and writing image files
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 10. Quality metrics
// ---------------------------------------------------------------------------

namespace {

void check_same_dimensions(const Image& a, const Image& b)
{
    if (a.width() != b.width() || a.height() != b.height()) {
        throw std::invalid_argument{"the two images do not have the same dimensions"};
    }
}

// Sum of the squared pixel differences over the three channels.
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
        // Fully black reference: the relative error only makes sense if it is zero.
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

// ---------------------------------------------------------------------------
// 11. Noise
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 12. Command line
// ---------------------------------------------------------------------------

namespace {

double parse_double(const std::string& text, const std::string& flag)
{
    std::size_t used{0};
    double value{};
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    // The whole text must be consumed: "5abc" is rejected.
    if (used == 0 || used != text.size()) {
        throw std::invalid_argument{"invalid number for " + flag + ": '" + text + "'"};
    }
    return value;
}

int parse_int(const std::string& text, const std::string& flag)
{
    std::size_t used{0};
    int value{};
    try {
        value = std::stoi(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used == 0 || used != text.size()) {
        throw std::invalid_argument{"invalid integer for " + flag + ": '" + text + "'"};
    }
    return value;
}

std::size_t parse_cutoff(const std::string& text)
{
    const int value{parse_int(text, "--cutoff")};
    if (value < 0) {
        throw std::invalid_argument{"--cutoff must be non-negative"};
    }
    return static_cast<std::size_t>(value);
}

TableKind parse_table(const std::string& text)
{
    if (text == "standard") {
        return TableKind::standard;
    }
    if (text == "uniform") {
        return TableKind::uniform;
    }
    if (text == "low") {
        return TableKind::low_frequencies;
    }
    if (text == "high") {
        return TableKind::high_frequencies;
    }
    throw std::invalid_argument{"unknown table: '" + text + "' (standard, uniform, low or high)"};
}

MaskKind parse_mask(const std::string& text)
{
    if (text == "square") {
        return MaskKind::square;
    }
    if (text == "triangle") {
        return MaskKind::triangle;
    }
    throw std::invalid_argument{"unknown mask: '" + text + "' (square or triangle)"};
}

} // namespace

Options parse_options(int argc, const char* const argv[])
{
    // The arguments are copied into std::string: comparisons with == instead of strcmp.
    std::vector<std::string> args;
    for (int i{1}; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }

    Options options{};
    if (args.empty() || args[0] == "help" || args[0] == "--help" || args[0] == "-h") {
        return options;
    }

    if (args[0] == "decompress") {
        if (args.size() != 3) {
            throw std::invalid_argument{"usage: decompress <file.csr> <output.png>"};
        }
        options.command = Command::decompress;
        options.input = args[1];
        options.output = args[2];
        return options;
    }

    if (args[0] != "compress") {
        throw std::invalid_argument{"unknown command: '" + args[0] + "'"};
    }
    if (args.size() < 2) {
        throw std::invalid_argument{"usage: compress <image> [options]"};
    }
    options.command = Command::compress;
    options.input = args[1];

    // Options come in pairs: --name value.
    for (std::size_t i{2}; i < args.size(); i += 2) {
        const std::string& flag{args[i]};
        if (i + 1 >= args.size()) {
            throw std::invalid_argument{"missing value after " + flag};
        }
        const std::string& value{args[i + 1]};

        if (flag == "--table") {
            options.table = parse_table(value);
        } else if (flag == "--alpha") {
            options.alpha = parse_double(value, flag);
        } else if (flag == "--threshold") {
            options.threshold = parse_int(value, flag);
        } else if (flag == "--mask") {
            options.mask = parse_mask(value);
        } else if (flag == "--cutoff") {
            options.cutoff = parse_cutoff(value);
        } else if (flag == "--noise") {
            options.noise = parse_double(value, flag);
        } else if (flag == "--out") {
            options.output = value;
        } else {
            throw std::invalid_argument{"unknown option: " + flag};
        }
    }
    return options;
}

void print_usage(std::ostream& out)
{
    out << "DCT image compression with CSR sparse storage (C++ version).\n"
           "\n"
           "Usage:\n"
           "  jpeg_csr compress <image> [options]\n"
           "  jpeg_csr decompress <file.csr> <output.png>\n"
           "  jpeg_csr help\n"
           "\n"
           "compress options (the defaults match the Python version):\n"
           "  --table standard|uniform|low|high   quantization matrix Q          [standard]\n"
           "  --alpha A                           quality factor: Q -> A * Q     [1]\n"
           "  --threshold S                       zero coefficients with |c| < S [2]\n"
           "  --mask square|triangle              truncation shape               [square]\n"
           "  --cutoff F                          cutoff frequency               [6]\n"
           "  --noise P                           salt-and-pepper noise, prob. P [0]\n"
           "  --out DIR                           output directory               [results]\n";
}

} // namespace jpeg

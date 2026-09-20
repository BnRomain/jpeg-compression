// jpeg_csr: JPEG-inspired image compression (DCT, quantization, high-frequency
// truncation) with CSR sparse storage.
//
// Single header of the project: it declares every type and every function.
// The definitions are in jpeg.cpp, the program entry point in main.cpp.
//
// Contents, in dependency order:
//   1. Matrix8             8 x 8 matrix of reals
//   2. Image               RGB image in memory
//   3. Dct                 discrete cosine transform of a block
//   4. QuantizationTable   quantization matrix Q
//   5. SparseMatrix        CSR sparse matrix
//   6. FrequencyMask       truncation of the high frequencies (interface + 2 shapes)
//   7. CompressedImage     result of the compression, and its .csr file
//   8. compress/decompress the algorithm itself
//   9. load_image/save_png reading and writing image files
//  10. metrics             relative L2 error and PSNR
//  11. noise               salt-and-pepper noise
//  12. Options             command line

#ifndef JPEG_HPP
#define JPEG_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace jpeg {

// ---------------------------------------------------------------------------
// 1. Matrix8
// ---------------------------------------------------------------------------

// The JPEG algorithm works on 8 x 8 pixel blocks.
constexpr std::size_t block_size{8};

// 8 x 8 square matrix of reals: a block of pixels, the DCT change-of-basis
// matrix or a block of frequency coefficients.
//
// The 64 coefficients are stored row by row in a std::array. Its size is fixed
// at compile time: unlike std::vector, no dynamic allocation happens, which
// matters since several matrices are created for each of the thousands of
// blocks of an image.
// std::array knows how to copy and destroy itself: no special member function
// is written (rule of zero).
class Matrix8 {
public:
    Matrix8() = default;                 // all coefficients are 0
    explicit Matrix8(double value);      // all coefficients are equal to value

    // Access without bounds checking, like operator[] in lab 4 of the course:
    // one overload to write to a mutable object, one to read a const object.
    double& operator()(std::size_t row, std::size_t col);
    const double& operator()(std::size_t row, std::size_t col) const;

    Matrix8 transposed() const;

private:
    std::array<double, block_size * block_size> values_{};
};

// Matrix product, the equivalent of the NumPy @ operator.
Matrix8 operator*(const Matrix8& lhs, const Matrix8& rhs);

// ---------------------------------------------------------------------------
// 2. Image
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 3. Dct
// ---------------------------------------------------------------------------

// Two-dimensional discrete cosine transform (DCT-II) of an 8 x 8 block.
//
// For a block M of intensities centered in [-128, 127]:
//   D_{k,l} = 1/4 C_k C_l  sum_{i,j} M_{i,j} cos((2i+1) k pi / 16) cos((2j+1) l pi / 16)
// with C_0 = 1/sqrt(2) and C_k = 1 for k > 0.
//
// This formula is an orthonormal change of basis, written
//   D = P M P^T     (to the frequency domain: compression)
//   M = P^T D P     (back to intensities: decompression)
// where P_{k,i} = (C_k / 2) cos((2i+1) k pi / 16). Since P is orthogonal, its
// inverse is its transpose: no matrix inversion is needed.
//
// P and P^T are computed once, in the constructor, then reused for every block
// (the Python version passed P as an argument for the same reason).
class Dct {
public:
    Dct();

    const Matrix8& basis() const noexcept;               // the matrix P
    Matrix8 forward(const Matrix8& block) const;         // D = P M P^T
    Matrix8 inverse(const Matrix8& coefficients) const;  // M = P^T D P

private:
    // The declaration order sets the initialization order: p_ must exist
    // before p_transposed_ is computed from it.
    Matrix8 p_;
    Matrix8 p_transposed_;
};

// ---------------------------------------------------------------------------
// 4. QuantizationTable
// ---------------------------------------------------------------------------

// Quantization matrix Q: each DCT coefficient D_{k,l} is divided by Q_{k,l},
// then truncated toward zero. The larger Q_{k,l}, the more likely the
// coefficient is zeroed.
//
// Invariant: all divisors are >= 1.
//  - a zero or negative divisor makes no sense for a quantization;
//  - since |D_{k,l}| <= 8 * 128 = 1024, a divisor >= 1 guarantees that each
//    quantized coefficient fits in a std::int16_t, the type stored in the CSR
//    matrix (like .astype(np.int16) in Python).
// The constructor checks the invariant and no method modifies the table.
class QuantizationTable {
public:
    // Throws std::invalid_argument if a divisor is < 1.
    explicit QuantizationTable(const Matrix8& divisors);

    // Tables studied in the MAM3 project.
    static QuantizationTable standard();            // JPEG standard, known as psychovisual
    static QuantizationTable uniform(double value); // same divisor everywhere
    static QuantizationTable low_frequencies();     // small divisors at the top left
    static QuantizationTable high_frequencies();    // small divisors at the bottom right

    // Quality factor alpha of the report: returns the table alpha * Q.
    // Throws std::invalid_argument if alpha <= 0 or if alpha * Q breaks the invariant.
    QuantizationTable scaled(double alpha) const;

    double operator()(std::size_t k, std::size_t l) const;
    const Matrix8& divisors() const noexcept;

private:
    Matrix8 divisors_;
};

// ---------------------------------------------------------------------------
// 5. SparseMatrix
// ---------------------------------------------------------------------------

// Sparse matrix in CSR (Compressed Sparse Row) format, the equivalent of
// scipy.sparse.csr_matrix in the Python version. After quantization, most
// coefficients are zero: only the others are stored.
//
// Representation (nnz = number of non-zero coefficients):
//   values_          [nnz]     non-zero values, row by row (int16);
//   column_indices_  [nnz]     column of each value (int32);
//   row_pointers_    [rows+1]  row i occupies positions
//                              [row_pointers_[i], row_pointers_[i+1]) of the
//                              two previous arrays (int32).
// Same types as SciPy (int16 data, int32 indices): the memory sizes of both
// versions are directly comparable.
//
// Example:    | 5 0 0 |      values_         = {5, 3, 7}
//             | 0 0 0 |  ->  column_indices_ = {0, 1, 2}
//             | 0 3 7 |      row_pointers_   = {0, 1, 1, 3}
//
// Invariant:
//   row_pointers_.size() == rows + 1, row_pointers_[0] == 0,
//   row_pointers_ non-decreasing and row_pointers_[rows] == nnz;
//   values_.size() == column_indices_.size() == nnz;
//   within a row, columns strictly increasing and in [0, cols);
//   no stored value is zero.
// The three arrays are std::vector: rule of zero.
class SparseMatrix {
public:
    // Builds the CSR matrix from a dense rows x cols matrix stored row by row.
    // Throws std::invalid_argument if the size does not match.
    SparseMatrix(std::size_t rows, std::size_t cols, const std::vector<std::int16_t>& dense);

    // Builds the matrix from the three CSR arrays (when a file is read back).
    // The arrays are taken by value, then moved into the object, without
    // copying the elements.
    // Throws std::invalid_argument if the invariant does not hold.
    SparseMatrix(std::size_t rows, std::size_t cols,
                 std::vector<std::int16_t> values,
                 std::vector<std::int32_t> column_indices,
                 std::vector<std::int32_t> row_pointers);

    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;
    std::size_t nnz() const noexcept;

    // Coefficient (row, col), zero if it is not stored.
    // Throws std::out_of_range outside the matrix.
    std::int16_t at(std::size_t row, std::size_t col) const;

    std::vector<std::int16_t> to_dense() const;

    // Bytes used by the three arrays, like
    // data.nbytes + indices.nbytes + indptr.nbytes in Python.
    std::size_t storage_bytes() const noexcept;

    const std::vector<std::int16_t>& values() const noexcept;
    const std::vector<std::int32_t>& column_indices() const noexcept;
    const std::vector<std::int32_t>& row_pointers() const noexcept;

private:
    void check_invariant() const;

    std::size_t rows_;
    std::size_t cols_;
    std::vector<std::int16_t> values_;
    std::vector<std::int32_t> column_indices_;
    std::vector<std::int32_t> row_pointers_;
};

// ---------------------------------------------------------------------------
// 6. FrequencyMask
// ---------------------------------------------------------------------------

// Abstract interface: tells whether the frequency coefficient (k, l) of a block
// is kept. Rejected coefficients are set to zero after quantization, which
// filters out the high frequencies (and part of the noise).
//
// The compression only knows this interface: it receives a
// const FrequencyMask& and calls keeps() without knowing which strategy is
// behind it (same idea as ScalarFunction and midpoint in lab 6 of the course).
// Adding a new mask shape therefore does not change the compression.
class FrequencyMask {
public:
    virtual bool keeps(std::size_t k, std::size_t l) const = 0;
    virtual std::string name() const = 0;

    // Virtual destructor: destroying a derived object through the base stays correct.
    virtual ~FrequencyMask() = default;
};

// Square truncation of the Python version: D[F:, :] = 0 and D[:, F:] = 0.
// Keeps the coefficients with k < F and l < F.
// Invariant: 1 <= F <= 8 (F = 8 removes nothing).
class SquareMask : public FrequencyMask {
public:
    explicit SquareMask(std::size_t cutoff);   // throws std::invalid_argument

    bool keeps(std::size_t k, std::size_t l) const override;
    std::string name() const override;

private:
    std::size_t cutoff_;
};

// Triangular truncation of the MAM3 assignment: zeroes the coefficients with
// k + l >= F, F being the cutoff frequency.
// Invariant: 1 <= F <= 15 (k + l is at most 14, so F = 15 removes nothing).
class TriangleMask : public FrequencyMask {
public:
    explicit TriangleMask(std::size_t cutoff);   // throws std::invalid_argument

    bool keeps(std::size_t k, std::size_t l) const override;
    std::string name() const override;

private:
    std::size_t cutoff_;
};

// ---------------------------------------------------------------------------
// 7. CompressedImage
// ---------------------------------------------------------------------------

// Result of the compression: everything needed to rebuild the image.
//  - the dimensions of the cropped image (multiples of 8);
//  - the quantization matrix used, required for decompression;
//  - one CSR matrix per R, G, B channel holding the quantized coefficients,
//    laid out like the pixels (coefficient (k, l) of the block whose corner is
//    (top, left) is stored at (top + k, left + l)), like the three csr_matrix
//    objects of the Python version.
//
// Invariant: width and height are non-zero multiples of 8, with exactly 3
// channels of size height x width.
// Composition of types that already manage their resources: rule of zero.
class CompressedImage {
public:
    // Throws std::invalid_argument if the invariant does not hold.
    CompressedImage(std::size_t width, std::size_t height,
                    const QuantizationTable& table,
                    std::vector<SparseMatrix> channels);

    std::size_t width() const noexcept;
    std::size_t height() const noexcept;
    const QuantizationTable& table() const noexcept;
    const SparseMatrix& channel(std::size_t index) const;   // throws std::out_of_range

    std::size_t nnz() const noexcept;                // non-zero coefficients of the 3 channels
    std::size_t coefficient_count() const noexcept;  // width * height * 3
    double conservation_rate() const noexcept;       // nnz / coefficient_count
    std::size_t storage_bytes() const noexcept;      // bytes of the 3 CSR matrices

private:
    std::size_t width_;
    std::size_t height_;
    QuantizationTable table_;
    std::vector<SparseMatrix> channels_;
};

// Binary .csr file, the equivalent of the .npz file of the Streamlit app: it
// shows the gain of sparse storage on disk.
//
// Format (integers and floating-point values in the byte order of the machine):
//   "JCSR"                  signature, 4 bytes
//   width, height           uint32
//   Q                       64 doubles, row by row
//   then for each channel R, G, B:
//     nnz                   uint32
//     row_pointers          (height + 1) int32
//     column_indices        nnz int32
//     values                nnz int16
//
// Both functions throw std::runtime_error if the file cannot be written or read,
// and std::invalid_argument if its content is inconsistent.
void save_compressed(const CompressedImage& image, const std::string& path);
CompressedImage load_compressed(const std::string& path);

// ---------------------------------------------------------------------------
// 8. Compression and decompression
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 9. Reading and writing image files
// ---------------------------------------------------------------------------

// The whole dependency on the stb library is confined to jpeg.cpp: the rest of
// the program only handles the Image class.

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

// ---------------------------------------------------------------------------
// 10. Quality metrics
// ---------------------------------------------------------------------------

// Metrics between a reference image and its approximation (post-processing
// step of the assignment). Both images must have the same dimensions,
// otherwise std::invalid_argument is thrown.

// Relative error in L2 norm: ||reference - approximation|| / ||reference||,
// computed on the [0, 255] intensities of the three channels.
double relative_l2_error(const Image& reference, const Image& approximation);

// Peak signal-to-noise ratio, in dB: 10 log10(255^2 / mean squared error).
// The higher it is, the closer the approximation; it is +infinity when both
// images are identical.
double psnr(const Image& reference, const Image& approximation);

// ---------------------------------------------------------------------------
// 11. Noise
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 12. Command line
// ---------------------------------------------------------------------------

enum class Command { help, compress, decompress };
enum class TableKind { standard, uniform, low_frequencies, high_frequencies };
enum class MaskKind { square, triangle };

// Settings read from the command line. The defaults reproduce the Python app:
// standard Q, alpha = 1, threshold = 2, square truncation F = 6.
//
// A plain aggregate of parameters, deliberately left as a struct: the validity
// of the values (alpha > 0, F in the right range, noise in [0, 1]...) is
// checked by the classes that use them, in their constructors.
struct Options {
    Command command{Command::help};
    std::string input;
    std::string output{"results"};     // directory (compress) or PNG file (decompress)
    TableKind table{TableKind::standard};
    double alpha{1.0};
    int threshold{2};
    MaskKind mask{MaskKind::square};
    std::size_t cutoff{6};
    double noise{0.0};                 // probability of the salt-and-pepper noise
};

// Parses the program arguments.
// Throws std::invalid_argument if the command is malformed.
Options parse_options(int argc, const char* const argv[]);

void print_usage(std::ostream& out);

} // namespace jpeg

#endif

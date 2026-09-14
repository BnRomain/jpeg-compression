#ifndef JPEG_COMPRESSED_IMAGE_HPP
#define JPEG_COMPRESSED_IMAGE_HPP

#include "quantization_table.hpp"
#include "sparse_matrix.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace jpeg {

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

} // namespace jpeg

#endif

#ifndef JPEG_SPARSE_MATRIX_HPP
#define JPEG_SPARSE_MATRIX_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace jpeg {

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

} // namespace jpeg

#endif

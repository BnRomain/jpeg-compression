#include "sparse_matrix.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace jpeg {

namespace {

// Les indices sont stockés sur 32 bits : la matrice doit rester indexable.
void check_dimensions(std::size_t rows, std::size_t cols)
{
    const auto max_index{static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())};
    if (rows > max_index || cols > max_index || (cols != 0 && rows > max_index / cols)) {
        throw std::invalid_argument{"matrice trop grande pour des indices 32 bits"};
    }
}

} // namespace

SparseMatrix::SparseMatrix(std::size_t rows, std::size_t cols, const std::vector<std::int16_t>& dense)
    : rows_{rows},
      cols_{cols}
{
    check_dimensions(rows, cols);
    if (dense.size() != rows * cols) {
        throw std::invalid_argument{"la matrice dense n'a pas la taille rows * cols"};
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
        // Fin de la ligne i = début de la ligne i + 1.
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
        throw std::invalid_argument{"CSR : row_pointers doit avoir rows + 1 éléments et commencer à 0"};
    }
    if (values_.size() != column_indices_.size()
        || static_cast<std::size_t>(row_pointers_.back()) != values_.size()) {
        throw std::invalid_argument{"CSR : tailles de values, column_indices et row_pointers incohérentes"};
    }
    // Première passe : pointeurs croissants. Ils restent alors tous dans
    // [0, nnz], ce qui rend sûrs les accès de la seconde passe.
    for (std::size_t i{0}; i < rows_; ++i) {
        if (row_pointers_[i + 1] < row_pointers_[i]) {
            throw std::invalid_argument{"CSR : row_pointers doit être croissant"};
        }
    }
    for (std::size_t i{0}; i < rows_; ++i) {
        const auto begin{static_cast<std::size_t>(row_pointers_[i])};
        const auto end{static_cast<std::size_t>(row_pointers_[i + 1])};
        for (std::size_t p{begin}; p < end; ++p) {
            const std::int32_t col{column_indices_[p]};
            if (col < 0 || static_cast<std::size_t>(col) >= cols_) {
                throw std::invalid_argument{"CSR : indice de colonne hors de la matrice"};
            }
            if (p > begin && col <= column_indices_[p - 1]) {
                throw std::invalid_argument{"CSR : colonnes non strictement croissantes sur une ligne"};
            }
            if (values_[p] == 0) {
                throw std::invalid_argument{"CSR : une valeur stockée est nulle"};
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
        throw std::out_of_range{"coefficient en dehors de la matrice CSR"};
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

} // namespace jpeg

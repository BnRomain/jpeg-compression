#ifndef JPEG_SPARSE_MATRIX_HPP
#define JPEG_SPARSE_MATRIX_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace jpeg {

// Matrice creuse au format CSR (Compressed Sparse Row), équivalent de
// scipy.sparse.csr_matrix dans la version Python. Après quantification, la
// grande majorité des coefficients est nulle : on ne stocke que les autres.
//
// Représentation (nnz = nombre de coefficients non nuls) :
//   values_          [nnz]     valeurs non nulles, ligne par ligne (int16) ;
//   column_indices_  [nnz]     colonne de chaque valeur (int32) ;
//   row_pointers_    [rows+1]  la ligne i occupe les positions
//                              [row_pointers_[i], row_pointers_[i+1]) des deux
//                              tableaux précédents (int32).
// Mêmes types que scipy (données int16, indices int32) : les tailles mémoire
// des deux versions sont directement comparables.
//
// Exemple :   | 5 0 0 |      values_         = {5, 3, 7}
//             | 0 0 0 |  ->  column_indices_ = {0, 1, 2}
//             | 0 3 7 |      row_pointers_   = {0, 1, 1, 3}
//
// Invariant :
//   row_pointers_.size() == rows + 1, row_pointers_[0] == 0,
//   row_pointers_ croissant et row_pointers_[rows] == nnz ;
//   values_.size() == column_indices_.size() == nnz ;
//   sur une ligne, colonnes strictement croissantes et dans [0, cols) ;
//   aucune valeur stockée n'est nulle.
// Les trois tableaux sont des std::vector : règle de zéro.
class SparseMatrix {
public:
    // Construit la matrice CSR à partir d'une matrice dense rows x cols rangée
    // ligne par ligne. Lance std::invalid_argument si la taille ne correspond pas.
    SparseMatrix(std::size_t rows, std::size_t cols, const std::vector<std::int16_t>& dense);

    // Construit la matrice à partir des trois tableaux CSR (relecture d'un
    // fichier). Les tableaux sont reçus par valeur puis déplacés dans l'objet,
    // sans recopie des éléments.
    // Lance std::invalid_argument si l'invariant n'est pas respecté.
    SparseMatrix(std::size_t rows, std::size_t cols,
                 std::vector<std::int16_t> values,
                 std::vector<std::int32_t> column_indices,
                 std::vector<std::int32_t> row_pointers);

    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;
    std::size_t nnz() const noexcept;

    // Coefficient (row, col), nul s'il n'est pas stocké.
    // Lance std::out_of_range en dehors de la matrice.
    std::int16_t at(std::size_t row, std::size_t col) const;

    std::vector<std::int16_t> to_dense() const;

    // Octets occupés par les trois tableaux, comme
    // data.nbytes + indices.nbytes + indptr.nbytes en Python.
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

#ifndef JPEG_COMPRESSED_IMAGE_HPP
#define JPEG_COMPRESSED_IMAGE_HPP

#include "quantization_table.hpp"
#include "sparse_matrix.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace jpeg {

// Résultat de la compression : tout ce qu'il faut garder pour reconstruire l'image.
//  - les dimensions de l'image rognée (multiples de 8) ;
//  - la matrice de quantification utilisée, indispensable à la décompression ;
//  - une matrice CSR par canal R, G, B contenant les coefficients quantifiés,
//    rangés comme les pixels (le coefficient (k, l) du bloc de coin (top, left)
//    est en (top + k, left + l)), comme les trois csr_matrix de la version Python.
//
// Invariant : largeur et hauteur multiples non nuls de 8, exactement 3 canaux
// de taille height x width.
// Composition de types qui gèrent déjà leurs ressources : règle de zéro.
class CompressedImage {
public:
    // Lance std::invalid_argument si l'invariant n'est pas respecté.
    CompressedImage(std::size_t width, std::size_t height,
                    const QuantizationTable& table,
                    std::vector<SparseMatrix> channels);

    std::size_t width() const noexcept;
    std::size_t height() const noexcept;
    const QuantizationTable& table() const noexcept;
    const SparseMatrix& channel(std::size_t index) const;   // lance std::out_of_range

    std::size_t nnz() const noexcept;                // coefficients non nuls des 3 canaux
    std::size_t coefficient_count() const noexcept;  // width * height * 3
    double conservation_rate() const noexcept;       // nnz / coefficient_count
    std::size_t storage_bytes() const noexcept;      // octets des 3 matrices CSR

private:
    std::size_t width_;
    std::size_t height_;
    QuantizationTable table_;
    std::vector<SparseMatrix> channels_;
};

// Fichier binaire .csr, équivalent du fichier .npz de l'application Streamlit :
// il matérialise sur disque le gain du stockage creux.
//
// Format (entiers et réels écrits dans l'ordre d'octets de la machine) :
//   "JCSR"                  signature, 4 octets
//   width, height           uint32
//   Q                       64 double, ligne par ligne
//   puis pour chaque canal R, G, B :
//     nnz                   uint32
//     row_pointers          (height + 1) int32
//     column_indices        nnz int32
//     values                nnz int16
//
// Lancent std::runtime_error si le fichier ne peut pas être écrit ou lu, et
// std::invalid_argument si son contenu est incohérent.
void save_compressed(const CompressedImage& image, const std::string& path);
CompressedImage load_compressed(const std::string& path);

} // namespace jpeg

#endif

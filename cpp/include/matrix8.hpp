#ifndef JPEG_MATRIX8_HPP
#define JPEG_MATRIX8_HPP

#include <array>
#include <cstddef>

namespace jpeg {

// L'algorithme JPEG travaille sur des blocs de 8 x 8 pixels.
constexpr std::size_t block_size{8};

// Matrice carrée 8 x 8 de réels : bloc de pixels, matrice de passage de la
// DCT ou bloc de coefficients fréquentiels.
//
// Les 64 coefficients sont rangés ligne par ligne dans un std::array. Sa
// taille est fixée à la compilation : contrairement à std::vector, aucune
// allocation dynamique n'a lieu, ce qui compte puisqu'on crée plusieurs
// matrices pour chacun des milliers de blocs d'une image.
// std::array sait se copier et se détruire : aucune opération spéciale n'est
// écrite (règle de zéro).
class Matrix8 {
public:
    Matrix8() = default;                 // tous les coefficients valent 0
    explicit Matrix8(double value);      // tous les coefficients valent value

    // Accès sans vérification des bornes, comme operator[] au TD4 : une
    // surcharge pour écrire dans un objet modifiable, une pour lire un objet const.
    double& operator()(std::size_t row, std::size_t col);
    const double& operator()(std::size_t row, std::size_t col) const;

    Matrix8 transposed() const;

private:
    std::array<double, block_size * block_size> values_{};
};

// Produit matriciel, équivalent de l'opérateur @ de numpy.
Matrix8 operator*(const Matrix8& lhs, const Matrix8& rhs);

} // namespace jpeg

#endif

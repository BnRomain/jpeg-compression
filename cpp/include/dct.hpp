#ifndef JPEG_DCT_HPP
#define JPEG_DCT_HPP

#include "matrix8.hpp"

namespace jpeg {

// Transformée en cosinus discrète bidimensionnelle (DCT-II) d'un bloc 8 x 8.
//
// Pour un bloc M d'intensités centrées dans [-128, 127] :
//   D_{k,l} = 1/4 C_k C_l  sum_{i,j} M_{i,j} cos((2i+1) k pi / 16) cos((2j+1) l pi / 16)
// avec C_0 = 1/sqrt(2) et C_k = 1 pour k > 0.
//
// Cette formule est un changement de base orthonormée qui s'écrit
//   D = P M P^T     (passage en fréquentiel : compression)
//   M = P^T D P     (retour aux intensités : décompression)
// où P_{k,i} = (C_k / 2) cos((2i+1) k pi / 16). P étant orthogonale, son
// inverse est sa transposée : aucune inversion de matrice n'est nécessaire.
//
// P et P^T sont calculées une seule fois, à la construction, puis réutilisées
// pour tous les blocs (en Python, P était passée en argument pour la même raison).
class Dct {
public:
    Dct();

    const Matrix8& basis() const noexcept;               // la matrice P
    Matrix8 forward(const Matrix8& block) const;         // D = P M P^T
    Matrix8 inverse(const Matrix8& coefficients) const;  // M = P^T D P

private:
    // L'ordre de déclaration fixe l'ordre d'initialisation : p_ doit exister
    // avant que p_transposed_ soit calculée à partir d'elle.
    Matrix8 p_;
    Matrix8 p_transposed_;
};

} // namespace jpeg

#endif

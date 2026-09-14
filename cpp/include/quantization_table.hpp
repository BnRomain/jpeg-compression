#ifndef JPEG_QUANTIZATION_TABLE_HPP
#define JPEG_QUANTIZATION_TABLE_HPP

#include "matrix8.hpp"

namespace jpeg {

// Matrice de quantification Q : chaque coefficient DCT D_{k,l} est divisé par
// Q_{k,l} puis tronqué vers zéro. Plus Q_{k,l} est grand, plus le coefficient
// a de chances d'être annulé.
//
// Invariant : tous les diviseurs sont >= 1.
//  - un diviseur nul ou négatif n'a pas de sens pour une quantification ;
//  - comme |D_{k,l}| <= 8 * 128 = 1024, un diviseur >= 1 garantit que chaque
//    coefficient quantifié tient dans un std::int16_t, le type stocké dans le
//    CSR (comme .astype(np.int16) en Python).
// Le constructeur vérifie l'invariant et aucune méthode ne modifie la table.
class QuantizationTable {
public:
    // Lance std::invalid_argument si un diviseur est < 1.
    explicit QuantizationTable(const Matrix8& divisors);

    // Tables étudiées dans le projet MAM3.
    static QuantizationTable standard();            // norme JPEG, dite psychovisuelle
    static QuantizationTable uniform(double value); // même diviseur partout
    static QuantizationTable low_frequencies();     // diviseurs faibles en haut à gauche
    static QuantizationTable high_frequencies();    // diviseurs faibles en bas à droite

    // Facteur de qualité alpha du rapport : renvoie la table alpha * Q.
    // Lance std::invalid_argument si alpha <= 0 ou si alpha * Q casse l'invariant.
    QuantizationTable scaled(double alpha) const;

    double operator()(std::size_t k, std::size_t l) const;
    const Matrix8& divisors() const noexcept;

private:
    Matrix8 divisors_;
};

} // namespace jpeg

#endif

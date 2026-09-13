#include "quantization_table.hpp"

#include <array>
#include <stdexcept>

namespace jpeg {

namespace {

using Rows = std::array<std::array<double, block_size>, block_size>;

// Matrice de la norme JPEG telle qu'elle figure dans le sujet MAM3 et dans
// jpeg_compression.py. La norme donne 14 en position (1, 2) ; on conserve le
// 13 de la version Python pour que les deux programmes produisent exactement
// les mêmes coefficients et puissent être comparés.
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
            // Écrit !(d >= 1) plutôt que d < 1 pour rejeter aussi NaN.
            if (!(divisors(k, l) >= 1.0)) {
                throw std::invalid_argument{"les diviseurs de quantification doivent être >= 1"};
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
    // Matrice Q_filtre_HF du script Python : diviseur 1 pour k + l <= 2,
    // 10 sur la diagonale k + l = 3 et 1000 ailleurs. Seules les toutes
    // premières fréquences survivent : c'est un filtre passe-bas.
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
    // Matrice Q_special du script Python : diviseur 1000 sur le carré des
    // basses fréquences (k, l < 5), sauf la composante continue D_{0,0} gardée
    // à 16, et 1 ailleurs. On conserve surtout les hautes fréquences (contours).
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
        throw std::invalid_argument{"le facteur alpha doit être strictement positif"};
    }
    Matrix8 divisors{divisors_};
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            divisors(k, l) *= alpha;
        }
    }
    // Le constructeur revérifie l'invariant : alpha < 1 peut produire un diviseur < 1.
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

} // namespace jpeg

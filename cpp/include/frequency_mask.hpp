#ifndef JPEG_FREQUENCY_MASK_HPP
#define JPEG_FREQUENCY_MASK_HPP

#include <cstddef>
#include <string>

namespace jpeg {

// Interface abstraite : indique si le coefficient fréquentiel (k, l) d'un bloc
// est conservé. Les coefficients rejetés sont mis à zéro après quantification,
// ce qui filtre les hautes fréquences (et une partie du bruit).
//
// La compression ne connaît que cette interface : elle reçoit un
// const FrequencyMask& et appelle keeps() sans savoir quelle stratégie se
// trouve derrière (même principe que ScalarFunction et midpoint au TD6).
// Ajouter une nouvelle forme de masque ne modifie donc pas la compression.
class FrequencyMask {
public:
    virtual bool keeps(std::size_t k, std::size_t l) const = 0;
    virtual std::string name() const = 0;

    // Destructeur virtuel : détruire un objet dérivé via la base reste correct.
    virtual ~FrequencyMask() = default;
};

// Troncature carrée de la version Python : D[F:, :] = 0 et D[:, F:] = 0.
// On garde les coefficients tels que k < F et l < F.
// Invariant : 1 <= F <= 8 (F = 8 ne supprime rien).
class SquareMask : public FrequencyMask {
public:
    explicit SquareMask(std::size_t cutoff);   // lance std::invalid_argument

    bool keeps(std::size_t k, std::size_t l) const override;
    std::string name() const override;

private:
    std::size_t cutoff_;
};

// Troncature triangulaire du sujet MAM3 : on annule les coefficients tels que
// k + l >= F, F étant la fréquence de coupure.
// Invariant : 1 <= F <= 15 (k + l vaut au plus 14, donc F = 15 ne supprime rien).
class TriangleMask : public FrequencyMask {
public:
    explicit TriangleMask(std::size_t cutoff);   // lance std::invalid_argument

    bool keeps(std::size_t k, std::size_t l) const override;
    std::string name() const override;

private:
    std::size_t cutoff_;
};

} // namespace jpeg

#endif

#ifndef JPEG_IMAGE_HPP
#define JPEG_IMAGE_HPP

#include <cstddef>
#include <vector>

namespace jpeg {

// Image couleur RGB chargée en mémoire.
//
// Représentation : width x height pixels sur 3 canaux (rouge, vert, bleu),
// c'est le tableau nx x ny x 3 du sujet. Les intensités sont des réels de
// [0, 255] rangés ligne par ligne dans un unique std::vector :
//   indice = (row * width + col) * 3 + channel
//
// Invariant : width > 0, height > 0 et pixels_.size() == width * height * 3.
// Le constructeur l'établit, la représentation est privée et aucune méthode
// ne change la taille : le code extérieur ne peut pas casser l'invariant.
//
// Le stockage est confié à std::vector, qui se copie, se déplace et se libère
// seul : aucune opération spéciale n'est écrite (règle de zéro).
class Image {
public:
    static constexpr std::size_t channels{3};

    // Lance std::invalid_argument si une dimension est nulle.
    Image(std::size_t width, std::size_t height, double value = 0.0);

    std::size_t width() const noexcept;
    std::size_t height() const noexcept;
    std::size_t size() const noexcept;   // nombre de valeurs : width * height * 3

    // Accès rapide sans vérification des bornes, pour les boucles de calcul.
    double& operator()(std::size_t row, std::size_t col, std::size_t channel);
    const double& operator()(std::size_t row, std::size_t col, std::size_t channel) const;

    // Accès vérifié : lance std::out_of_range en dehors de l'image.
    double& at(std::size_t row, std::size_t col, std::size_t channel);
    const double& at(std::size_t row, std::size_t col, std::size_t channel) const;

    // Copie rognée aux plus grands multiples de 8 (étape init() de la version
    // Python). L'image courante n'est pas modifiée, d'où le const.
    // Lance std::invalid_argument si l'image fait moins de 8 pixels de côté.
    Image cropped_to_blocks() const;

private:
    std::size_t index(std::size_t row, std::size_t col, std::size_t channel) const noexcept;

    std::size_t width_;
    std::size_t height_;
    std::vector<double> pixels_;
};

} // namespace jpeg

#endif

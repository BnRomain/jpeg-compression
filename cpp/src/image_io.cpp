#include "image_io.hpp"

// stb est une bibliothèque « header-only » : son code n'est compilé que dans
// le fichier qui définit la macro *_IMPLEMENTATION, ici et nulle part ailleurs.
// Le dossier third_party est passé à g++ avec -isystem pour ne pas afficher
// les avertissements internes à stb.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace jpeg {

namespace {

// Tableau de pixels alloué par stbi_load, qui doit être rendu avec
// stbi_image_free. Plutôt que d'appeler stbi_image_free à la main (et de
// l'oublier si une exception survient), la ressource est liée à la durée de
// vie de cet objet : RAII, comme GmshSession dans le TD de synthèse.
class StbPixels {
public:
    explicit StbPixels(const std::string& path);
    ~StbPixels();

    // Deux objets ne doivent pas libérer le même tableau : copie interdite.
    StbPixels(const StbPixels&) = delete;
    StbPixels& operator=(const StbPixels&) = delete;

    std::size_t width() const noexcept;
    std::size_t height() const noexcept;
    unsigned char operator[](std::size_t index) const;

private:
    int width_{};
    int height_{};
    unsigned char* data_{nullptr};
};

StbPixels::StbPixels(const std::string& path)
{
    int channels_in_file{};
    // Le dernier argument impose 3 canaux quel que soit le fichier :
    // stb convertit le gris en RGB et supprime le canal alpha.
    data_ = stbi_load(path.c_str(), &width_, &height_, &channels_in_file,
                      static_cast<int>(Image::channels));
    if (data_ == nullptr) {
        // Si le constructeur lance, le destructeur n'est pas appelé ; ce n'est
        // pas un problème ici puisque rien n'a été alloué.
        throw std::runtime_error{"impossible de lire l'image '" + path + "' ("
                                 + stbi_failure_reason() + ")"};
    }
}

StbPixels::~StbPixels()
{
    stbi_image_free(data_);
}

std::size_t StbPixels::width() const noexcept
{
    return static_cast<std::size_t>(width_);
}

std::size_t StbPixels::height() const noexcept
{
    return static_cast<std::size_t>(height_);
}

unsigned char StbPixels::operator[](std::size_t index) const
{
    return data_[index];
}

} // namespace

Image load_image(const std::string& path)
{
    const StbPixels pixels{path};
    Image image{pixels.width(), pixels.height()};

    // stb range lui aussi les pixels ligne par ligne, canal par canal.
    std::size_t index{0};
    for (std::size_t row{0}; row < image.height(); ++row) {
        for (std::size_t col{0}; col < image.width(); ++col) {
            for (std::size_t channel{0}; channel < Image::channels; ++channel) {
                image(row, col, channel) = pixels[index];
                ++index;
            }
        }
    }
    return image;
}   // pixels est détruit ici : stbi_image_free est appelé automatiquement

void save_png(const Image& image, const std::string& path)
{
    std::vector<unsigned char> bytes(image.size());
    std::size_t index{0};
    for (std::size_t row{0}; row < image.height(); ++row) {
        for (std::size_t col{0}; col < image.width(); ++col) {
            for (std::size_t channel{0}; channel < Image::channels; ++channel) {
                const double value{std::clamp(image(row, col, channel), 0.0, 255.0)};
                bytes[index] = static_cast<unsigned char>(std::lround(value));
                ++index;
            }
        }
    }

    const int width{static_cast<int>(image.width())};
    const int height{static_cast<int>(image.height())};
    const int channels{static_cast<int>(Image::channels)};
    if (stbi_write_png(path.c_str(), width, height, channels, bytes.data(), width * channels) == 0) {
        throw std::runtime_error{"impossible d'écrire l'image '" + path + "'"};
    }
}

} // namespace jpeg

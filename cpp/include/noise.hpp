#ifndef JPEG_NOISE_HPP
#define JPEG_NOISE_HPP

#include "image.hpp"

namespace jpeg {

// Bruit « poivre et sel » : chaque pixel est remplacé, avec la probabilité
// `probability`, par un pixel noir (0) ou blanc (255), les deux cas étant
// équiprobables. Ce bruit est fait de variations très rapides, donc de hautes
// fréquences, que la compression atténue (effet passe-bas étudié dans le rapport).
//
// L'image est modifiée sur place, d'où le passage par référence non constante.
// Le générateur pseudo-aléatoire est initialisé avec `seed` : les expériences
// sont reproductibles.
// Lance std::invalid_argument si probability n'est pas dans [0, 1].
void add_salt_and_pepper(Image& image, double probability, unsigned seed = 42);

} // namespace jpeg

#endif

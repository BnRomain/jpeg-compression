#ifndef JPEG_METRICS_HPP
#define JPEG_METRICS_HPP

#include "image.hpp"

namespace jpeg {

// Indicateurs de qualité entre une image de référence et son approximation
// (étape de post-processing du sujet). Les deux images doivent avoir les mêmes
// dimensions, sinon std::invalid_argument est lancée.

// Erreur relative en norme L2 : ||reference - approximation|| / ||reference||,
// calculée sur les intensités [0, 255] des trois canaux.
double relative_l2_error(const Image& reference, const Image& approximation);

// Rapport signal sur bruit de crête, en dB : 10 log10(255^2 / erreur quadratique
// moyenne). Plus il est élevé, plus l'approximation est fidèle ; il vaut +infini
// si les deux images sont identiques.
double psnr(const Image& reference, const Image& approximation);

} // namespace jpeg

#endif

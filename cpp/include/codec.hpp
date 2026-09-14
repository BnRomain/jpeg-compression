#ifndef JPEG_CODEC_HPP
#define JPEG_CODEC_HPP

#include "compressed_image.hpp"
#include "frequency_mask.hpp"
#include "image.hpp"
#include "quantization_table.hpp"

namespace jpeg {

// Compression de l'image, canal par canal et bloc 8 x 8 par bloc 8 x 8
// (fonction compression() de la version Python) :
//   1. rognage aux multiples de 8, puis centrage : [0, 255] -> [-128, 127] ;
//   2. DCT : D = P M P^T ;
//   3. quantification : D_{k,l} / Q_{k,l} tronqué vers zéro (np.trunc) ;
//   4. seuil : annulation des coefficients de valeur absolue < threshold ;
//   5. masque : annulation des hautes fréquences rejetées par `mask` ;
//   6. stockage des coefficients de chaque canal dans une matrice CSR.
//
// Les arguments sont seulement lus : références constantes, aucune copie.
// `mask` est une référence sur l'interface FrequencyMask : n'importe quelle
// stratégie de troncature convient, sans modifier cette fonction.
// Lance std::invalid_argument si threshold < 0 ou si l'image est trop petite.
CompressedImage compress(const Image& image, const QuantizationTable& table,
                         int threshold, const FrequencyMask& mask);

// Décompression (fonction decompression() de la version Python), bloc par bloc :
// multiplication terme à terme par Q, DCT inverse M = P^T D P, décentrage
// (+128) puis bornage des intensités à [0, 255] (np.clip).
Image decompress(const CompressedImage& compressed);

} // namespace jpeg

#endif

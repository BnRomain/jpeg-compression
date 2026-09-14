#ifndef JPEG_IMAGE_IO_HPP
#define JPEG_IMAGE_IO_HPP

#include "image.hpp"

#include <string>

namespace jpeg {

// Lecture et écriture de fichiers image.
// Toute la dépendance à la bibliothèque stb est confinée dans image_io.cpp :
// le reste du programme ne manipule que la classe Image.

// Lit une image PNG, JPEG, BMP... Une image en niveaux de gris est convertie
// en RGB et un éventuel canal alpha (transparence) est ignoré, comme dans le
// prétraitement de la version Python.
// Lance std::runtime_error si le fichier est illisible.
Image load_image(const std::string& path);

// Écrit l'image au format PNG. Les intensités sont bornées à [0, 255] puis
// arrondies à l'entier le plus proche.
// Lance std::runtime_error en cas d'échec d'écriture.
void save_png(const Image& image, const std::string& path);

} // namespace jpeg

#endif

#ifndef JPEG_OPTIONS_HPP
#define JPEG_OPTIONS_HPP

#include <cstddef>
#include <iosfwd>
#include <string>

namespace jpeg {

enum class Command { help, compress, decompress };
enum class TableKind { standard, uniform, low_frequencies, high_frequencies };
enum class MaskKind { square, triangle };

// Réglages lus sur la ligne de commande. Les valeurs par défaut reproduisent
// l'application Python : Q standard, alpha = 1, seuil = 2, troncature carrée F = 6.
//
// Simple agrégat de paramètres, volontairement laissé en struct : la validité
// des valeurs (alpha > 0, F dans le bon intervalle, bruit dans [0, 1]...) est
// vérifiée par les classes qui les utilisent, dans leurs constructeurs.
struct Options {
    Command command{Command::help};
    std::string input;
    std::string output{"resultats"};   // dossier (compress) ou fichier PNG (decompress)
    TableKind table{TableKind::standard};
    double alpha{1.0};
    int threshold{2};
    MaskKind mask{MaskKind::square};
    std::size_t cutoff{6};
    double noise{0.0};                 // probabilité du bruit poivre et sel
};

// Analyse les arguments du programme.
// Lance std::invalid_argument si la commande est mal formée.
Options parse_options(int argc, const char* const argv[]);

void print_usage(std::ostream& out);

} // namespace jpeg

#endif

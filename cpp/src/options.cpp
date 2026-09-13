#include "options.hpp"

#include <ostream>
#include <stdexcept>
#include <vector>

namespace jpeg {

namespace {

double parse_double(const std::string& text, const std::string& flag)
{
    std::size_t used{0};
    double value{};
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    // On exige que tout le texte soit consommé : "5abc" est refusé.
    if (used == 0 || used != text.size()) {
        throw std::invalid_argument{"valeur numérique invalide pour " + flag + " : '" + text + "'"};
    }
    return value;
}

int parse_int(const std::string& text, const std::string& flag)
{
    std::size_t used{0};
    int value{};
    try {
        value = std::stoi(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used == 0 || used != text.size()) {
        throw std::invalid_argument{"entier invalide pour " + flag + " : '" + text + "'"};
    }
    return value;
}

std::size_t parse_cutoff(const std::string& text)
{
    const int value{parse_int(text, "--cutoff")};
    if (value < 0) {
        throw std::invalid_argument{"--cutoff doit être positif"};
    }
    return static_cast<std::size_t>(value);
}

TableKind parse_table(const std::string& text)
{
    if (text == "standard") {
        return TableKind::standard;
    }
    if (text == "uniform") {
        return TableKind::uniform;
    }
    if (text == "low") {
        return TableKind::low_frequencies;
    }
    if (text == "high") {
        return TableKind::high_frequencies;
    }
    throw std::invalid_argument{"table inconnue : '" + text + "' (standard, uniform, low ou high)"};
}

MaskKind parse_mask(const std::string& text)
{
    if (text == "square") {
        return MaskKind::square;
    }
    if (text == "triangle") {
        return MaskKind::triangle;
    }
    throw std::invalid_argument{"masque inconnu : '" + text + "' (square ou triangle)"};
}

} // namespace

Options parse_options(int argc, const char* const argv[])
{
    // Copie des arguments dans des std::string : comparaisons avec == au lieu de strcmp.
    std::vector<std::string> args;
    for (int i{1}; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }

    Options options{};
    if (args.empty() || args[0] == "help" || args[0] == "--help" || args[0] == "-h") {
        return options;
    }

    if (args[0] == "decompress") {
        if (args.size() != 3) {
            throw std::invalid_argument{"usage : decompress <fichier.csr> <sortie.png>"};
        }
        options.command = Command::decompress;
        options.input = args[1];
        options.output = args[2];
        return options;
    }

    if (args[0] != "compress") {
        throw std::invalid_argument{"commande inconnue : '" + args[0] + "'"};
    }
    if (args.size() < 2) {
        throw std::invalid_argument{"usage : compress <image> [options]"};
    }
    options.command = Command::compress;
    options.input = args[1];

    // Les options vont par paires : --nom valeur.
    for (std::size_t i{2}; i < args.size(); i += 2) {
        const std::string& flag{args[i]};
        if (i + 1 >= args.size()) {
            throw std::invalid_argument{"valeur manquante après " + flag};
        }
        const std::string& value{args[i + 1]};

        if (flag == "--table") {
            options.table = parse_table(value);
        } else if (flag == "--alpha") {
            options.alpha = parse_double(value, flag);
        } else if (flag == "--threshold") {
            options.threshold = parse_int(value, flag);
        } else if (flag == "--mask") {
            options.mask = parse_mask(value);
        } else if (flag == "--cutoff") {
            options.cutoff = parse_cutoff(value);
        } else if (flag == "--noise") {
            options.noise = parse_double(value, flag);
        } else if (flag == "--out") {
            options.output = value;
        } else {
            throw std::invalid_argument{"option inconnue : " + flag};
        }
    }
    return options;
}

void print_usage(std::ostream& out)
{
    out << "Compression d'images par DCT et stockage creux CSR (version C++).\n"
           "\n"
           "Usage :\n"
           "  jpeg_csr compress <image> [options]\n"
           "  jpeg_csr decompress <fichier.csr> <sortie.png>\n"
           "  jpeg_csr help\n"
           "\n"
           "Options de compress (valeurs par défaut identiques à la version Python) :\n"
           "  --table standard|uniform|low|high   matrice de quantification Q     [standard]\n"
           "  --alpha A                           facteur de qualité : Q -> A * Q [1]\n"
           "  --threshold S                       annule les coefficients |c| < S [2]\n"
           "  --mask square|triangle              forme de la troncature          [square]\n"
           "  --cutoff F                          fréquence de coupure            [6]\n"
           "  --noise P                           bruit poivre et sel, proba P    [0]\n"
           "  --out DOSSIER                       dossier des résultats           [resultats]\n";
}

} // namespace jpeg

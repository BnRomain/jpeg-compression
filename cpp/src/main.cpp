// Programme jpeg_csr : compression d'images inspirée de JPEG (DCT, quantification,
// troncature des hautes fréquences) avec stockage creux CSR.
// Version C++ de python/jpeg_compression.py et de l'application python/app.py.

#include "codec.hpp"
#include "compressed_image.hpp"
#include "frequency_mask.hpp"
#include "image.hpp"
#include "image_io.hpp"
#include "metrics.hpp"
#include "noise.hpp"
#include "options.hpp"
#include "quantization_table.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX   // empêche windows.h de définir les macros min et max
#endif
#include <windows.h>
#endif

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double milliseconds_since(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double kilobytes(std::uintmax_t bytes)
{
    return static_cast<double>(bytes) / 1024.0;
}

jpeg::QuantizationTable make_table(const jpeg::Options& options)
{
    switch (options.table) {
    case jpeg::TableKind::uniform:
        return jpeg::QuantizationTable::uniform(100.0).scaled(options.alpha);
    case jpeg::TableKind::low_frequencies:
        return jpeg::QuantizationTable::low_frequencies().scaled(options.alpha);
    case jpeg::TableKind::high_frequencies:
        return jpeg::QuantizationTable::high_frequencies().scaled(options.alpha);
    case jpeg::TableKind::standard:
        break;
    }
    return jpeg::QuantizationTable::standard().scaled(options.alpha);
}

std::string table_name(jpeg::TableKind kind)
{
    switch (kind) {
    case jpeg::TableKind::uniform:
        return "uniforme (100)";
    case jpeg::TableKind::low_frequencies:
        return "basses fréquences";
    case jpeg::TableKind::high_frequencies:
        return "hautes fréquences";
    case jpeg::TableKind::standard:
        break;
    }
    return "standard";
}

void print_quality(const std::string& label, const jpeg::Image& reference,
                   const jpeg::Image& approximation)
{
    std::cout << label << "erreur L2 relative "
              << 100.0 * jpeg::relative_l2_error(reference, approximation) << " %, PSNR "
              << jpeg::psnr(reference, approximation) << " dB\n";
}

// Compression complète avec un masque dont le type réel n'est connu qu'à
// l'exécution : cette fonction ne voit qu'une référence sur l'interface.
int compress_with_mask(const jpeg::Options& options, const jpeg::FrequencyMask& mask)
{
    const jpeg::Image original{jpeg::load_image(options.input)};
    const jpeg::Image clean{original.cropped_to_blocks()};
    const jpeg::QuantizationTable table{make_table(options)};

    const fs::path output_dir{options.output};
    fs::create_directories(output_dir);
    const std::string stem{fs::path{options.input}.stem().string()};

    // Image effectivement compressée : l'image rognée, bruitée si demandé.
    jpeg::Image input{clean};
    if (options.noise > 0.0) {
        jpeg::add_salt_and_pepper(input, options.noise);
        jpeg::save_png(input, (output_dir / (stem + "_bruitee.png")).string());
    }

    Clock::time_point start{Clock::now()};
    const jpeg::CompressedImage compressed{jpeg::compress(input, table, options.threshold, mask)};
    const double compression_ms{milliseconds_since(start)};

    start = Clock::now();
    const jpeg::Image reconstructed{jpeg::decompress(compressed)};
    const double decompression_ms{milliseconds_since(start)};

    const fs::path csr_path{output_dir / (stem + ".csr")};
    const fs::path png_path{output_dir / (stem + "_reconstruite.png")};
    jpeg::save_compressed(compressed, csr_path.string());
    jpeg::save_png(reconstructed, png_path.string());

    // Mêmes indicateurs que l'application Streamlit. « Données RAM » y valait
    // img.nbytes : l'image d'origine dépliée en float64, 8 octets par valeur.
    const std::uintmax_t dense_bytes{original.size() * sizeof(double)};
    const std::uintmax_t csr_bytes{compressed.storage_bytes()};
    const std::uintmax_t csr_file_bytes{fs::file_size(csr_path)};
    const std::uintmax_t source_file_bytes{fs::file_size(options.input)};

    std::cout << std::fixed << std::setprecision(2)
              << "Image          : " << options.input << " (" << original.width() << " x "
              << original.height() << ", rognée à " << clean.width() << " x " << clean.height() << ")\n"
              << "Réglages       : Q " << table_name(options.table) << ", alpha " << options.alpha
              << ", seuil " << options.threshold << ", masque " << mask.name();
    if (options.noise > 0.0) {
        std::cout << ", bruit " << 100.0 * options.noise << " %";
    }
    std::cout << '\n'
              << "Coefficients   : " << compressed.nnz() << " non nuls sur "
              << compressed.coefficient_count() << " (taux de conservation "
              << 100.0 * compressed.conservation_rate() << " %)\n";
    print_quality("Qualité        : ", input, reconstructed);
    if (options.noise > 0.0) {
        // Effet débruitage : on compare à l'image sans bruit l'image bruitée,
        // puis l'image reconstruite. Une erreur plus faible après compression
        // signifie qu'une partie du bruit a été filtrée.
        print_quality("Bruitée / pure : ", clean, input);
        print_quality("Reconst. / pure: ", clean, reconstructed);
    }
    std::cout << "Mémoire dense  : " << kilobytes(dense_bytes) << " Ko (float64, comme img.nbytes)\n"
              << "Mémoire CSR    : " << kilobytes(csr_bytes) << " Ko (valeurs int16, indices int32)\n"
              << "Gain mémoire   : dense / CSR = "
              << static_cast<double>(dense_bytes) / static_cast<double>(csr_bytes) << "x\n"
              << "Fichier .csr   : " << kilobytes(csr_file_bytes) << " Ko, image source "
              << kilobytes(source_file_bytes) << " Ko (source / .csr = "
              << static_cast<double>(source_file_bytes) / static_cast<double>(csr_file_bytes) << "x)\n"
              << "Temps          : compression " << compression_ms << " ms, décompression "
              << decompression_ms << " ms\n"
              << "Fichiers       : " << csr_path.generic_string() << ", " << png_path.generic_string() << '\n';
    return 0;
}

int run_compress(const jpeg::Options& options)
{
    // Le masque est choisi à l'exécution. Seul l'objet réellement demandé est
    // construit, et il vit jusqu'à la fin de compress_with_mask.
    if (options.mask == jpeg::MaskKind::triangle) {
        const jpeg::TriangleMask mask{options.cutoff};
        return compress_with_mask(options, mask);
    }
    const jpeg::SquareMask mask{options.cutoff};
    return compress_with_mask(options, mask);
}

int run_decompress(const jpeg::Options& options)
{
    const Clock::time_point start{Clock::now()};
    const jpeg::CompressedImage compressed{jpeg::load_compressed(options.input)};
    const jpeg::Image image{jpeg::decompress(compressed)};
    const double elapsed_ms{milliseconds_since(start)};

    const fs::path output{options.output};
    if (output.has_parent_path()) {
        fs::create_directories(output.parent_path());
    }
    jpeg::save_png(image, output.string());

    std::cout << std::fixed << std::setprecision(2)
              << "Fichier        : " << options.input << " (" << compressed.width() << " x "
              << compressed.height() << ")\n"
              << "Coefficients   : " << compressed.nnz() << " non nuls (taux de conservation "
              << 100.0 * compressed.conservation_rate() << " %)\n"
              << "Temps          : lecture et décompression " << elapsed_ms << " ms\n"
              << "Image écrite   : " << output.string() << '\n';
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
#ifdef _WIN32
    // La console Windows n'interprète pas l'UTF-8 par défaut (accents des messages).
    SetConsoleOutputCP(CP_UTF8);
#endif
    try {
        const jpeg::Options options{jpeg::parse_options(argc, argv)};
        switch (options.command) {
        case jpeg::Command::compress:
            return run_compress(options);
        case jpeg::Command::decompress:
            return run_decompress(options);
        case jpeg::Command::help:
            break;
        }
        jpeg::print_usage(std::cout);
        return 0;
    } catch (const std::exception& error) {
        // Toutes les erreurs (arguments, fichiers, invariants violés) remontent ici.
        std::cerr << "Erreur : " << error.what() << '\n';
        return 1;
    }
}

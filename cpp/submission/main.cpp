// jpeg_csr: JPEG-inspired image compression (DCT, quantization, high-frequency
// truncation) with CSR sparse storage.
// C++ version of python/jpeg_compression.py and of the python/app.py app.
//
// Build:
//   g++ -std=c++20 -O2 -Wall -Wextra -pedantic -isystem . jpeg.cpp main.cpp -o jpeg_csr
// (stb_image.h and stb_image_write.h must sit next to these files.)

#include "jpeg.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double milliseconds_since(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double kibibytes(std::uintmax_t bytes)
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
        return "uniform (100)";
    case jpeg::TableKind::low_frequencies:
        return "low frequencies";
    case jpeg::TableKind::high_frequencies:
        return "high frequencies";
    case jpeg::TableKind::standard:
        break;
    }
    return "standard";
}

void print_quality(const std::string& label, const jpeg::Image& reference,
                   const jpeg::Image& approximation)
{
    std::cout << label << "relative L2 error "
              << 100.0 * jpeg::relative_l2_error(reference, approximation) << " %, PSNR "
              << jpeg::psnr(reference, approximation) << " dB\n";
}

// Full compression with a mask whose actual type is only known at run time:
// this function only sees a reference to the interface.
int compress_with_mask(const jpeg::Options& options, const jpeg::FrequencyMask& mask)
{
    const jpeg::Image original{jpeg::load_image(options.input)};
    const jpeg::Image clean{original.cropped_to_blocks()};
    const jpeg::QuantizationTable table{make_table(options)};

    const fs::path output_dir{options.output};
    fs::create_directories(output_dir);
    const std::string stem{fs::path{options.input}.stem().string()};

    // Image actually compressed: the cropped image, with noise if requested.
    jpeg::Image input{clean};
    if (options.noise > 0.0) {
        jpeg::add_salt_and_pepper(input, options.noise);
        jpeg::save_png(input, (output_dir / (stem + "_noisy.png")).string());
    }

    Clock::time_point start{Clock::now()};
    const jpeg::CompressedImage compressed{jpeg::compress(input, table, options.threshold, mask)};
    const double compression_ms{milliseconds_since(start)};

    start = Clock::now();
    const jpeg::Image reconstructed{jpeg::decompress(compressed)};
    const double decompression_ms{milliseconds_since(start)};

    const fs::path csr_path{output_dir / (stem + ".csr")};
    const fs::path png_path{output_dir / (stem + "_reconstructed.png")};
    jpeg::save_compressed(compressed, csr_path.string());
    jpeg::save_png(reconstructed, png_path.string());

    // Same metrics as the Streamlit app. "RAM data" there is img.nbytes:
    // the original image unfolded as float64, 8 bytes per value.
    const std::uintmax_t dense_bytes{original.size() * sizeof(double)};
    const std::uintmax_t csr_bytes{compressed.storage_bytes()};
    const std::uintmax_t csr_file_bytes{fs::file_size(csr_path)};
    const std::uintmax_t source_file_bytes{fs::file_size(options.input)};

    std::cout << std::fixed << std::setprecision(2)
              << "Image          : " << options.input << " (" << original.width() << " x "
              << original.height() << ", cropped to " << clean.width() << " x " << clean.height() << ")\n"
              << "Settings       : Q " << table_name(options.table) << ", alpha " << options.alpha
              << ", threshold " << options.threshold << ", mask " << mask.name();
    if (options.noise > 0.0) {
        std::cout << ", noise " << 100.0 * options.noise << " %";
    }
    std::cout << '\n'
              << "Coefficients   : " << compressed.nnz() << " non-zero out of "
              << compressed.coefficient_count() << " (retention rate "
              << 100.0 * compressed.conservation_rate() << " %)\n";
    print_quality("Quality        : ", input, reconstructed);
    if (options.noise > 0.0) {
        // Denoising effect: the noisy image, then the reconstructed image, are
        // compared with the clean image. A lower error after compression means
        // that part of the noise was filtered out.
        print_quality("Noisy / clean  : ", clean, input);
        print_quality("Rebuilt / clean: ", clean, reconstructed);
    }
    std::cout << "Dense memory   : " << kibibytes(dense_bytes) << " KiB (float64, like img.nbytes)\n"
              << "CSR memory     : " << kibibytes(csr_bytes) << " KiB (int16 values, int32 indices)\n"
              << "Memory gain    : dense / CSR = "
              << static_cast<double>(dense_bytes) / static_cast<double>(csr_bytes) << "x\n"
              << ".csr file      : " << kibibytes(csr_file_bytes) << " KiB, source image "
              << kibibytes(source_file_bytes) << " KiB (source / .csr = "
              << static_cast<double>(source_file_bytes) / static_cast<double>(csr_file_bytes) << "x)\n"
              << "Time           : compression " << compression_ms << " ms, decompression "
              << decompression_ms << " ms\n"
              << "Files          : " << csr_path.generic_string() << ", " << png_path.generic_string() << '\n';
    return 0;
}

int run_compress(const jpeg::Options& options)
{
    // The mask is chosen at run time. Only the requested object is built, and
    // it lives until the end of compress_with_mask.
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
              << "File           : " << options.input << " (" << compressed.width() << " x "
              << compressed.height() << ")\n"
              << "Coefficients   : " << compressed.nnz() << " non-zero (retention rate "
              << 100.0 * compressed.conservation_rate() << " %)\n"
              << "Time           : reading and decompression " << elapsed_ms << " ms\n"
              << "Image written  : " << output.string() << '\n';
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
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
        // Every error (arguments, files, violated invariants) ends up here.
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

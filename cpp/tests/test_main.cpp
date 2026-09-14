// Tests unitaires de la version C++, écrits avec assert comme dans les TD.
// Ils reprennent les tests pytest de python/tests/test_compression.py et
// ajoutent la vérification des invariants de chaque classe.
//
// Lancement : make test   (ou make test SANITIZE=1 sous Linux / macOS)

#undef NDEBUG   // assert doit rester actif quelles que soient les options de compilation

#include "codec.hpp"
#include "compressed_image.hpp"
#include "dct.hpp"
#include "frequency_mask.hpp"
#include "image.hpp"
#include "matrix8.hpp"
#include "metrics.hpp"
#include "noise.hpp"
#include "options.hpp"
#include "quantization_table.hpp"
#include "sparse_matrix.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Vérifie que `statement` lance une exception du type attendu. L'instruction
// est passée entre parenthèses pour que ses virgules ne séparent pas les
// arguments de la macro.
#define ASSERT_THROWS(statement, exception_type) \
    do {                                         \
        bool thrown{false};                      \
        try {                                    \
            statement;                           \
        } catch (const exception_type&) {        \
            thrown = true;                       \
        }                                        \
        assert(thrown);                          \
    } while (false)

namespace {

using namespace jpeg;   // acceptable dans un fichier de test, jamais dans un en-tête

Image random_image(std::size_t width, std::size_t height)
{
    std::mt19937 generator{42};
    std::uniform_real_distribution<double> intensity{0.0, 255.0};
    Image image{width, height};
    for (std::size_t row{0}; row < height; ++row) {
        for (std::size_t col{0}; col < width; ++col) {
            for (std::size_t channel{0}; channel < Image::channels; ++channel) {
                image(row, col, channel) = intensity(generator);
            }
        }
    }
    return image;
}

void test_dct_basis_is_orthogonal()
{
    const Dct dct{};
    const Matrix8 product{dct.basis() * dct.basis().transposed()};
    for (std::size_t i{0}; i < block_size; ++i) {
        for (std::size_t j{0}; j < block_size; ++j) {
            const double expected{i == j ? 1.0 : 0.0};
            assert(std::abs(product(i, j) - expected) < 1e-12);
        }
    }
}

void test_dct_round_trip()
{
    const Dct dct{};
    Matrix8 block{};
    for (std::size_t i{0}; i < block_size; ++i) {
        for (std::size_t j{0}; j < block_size; ++j) {
            block(i, j) = static_cast<double>(i * block_size + j) - 32.0;
        }
    }
    const Matrix8 back{dct.inverse(dct.forward(block))};
    for (std::size_t i{0}; i < block_size; ++i) {
        for (std::size_t j{0}; j < block_size; ++j) {
            assert(std::abs(back(i, j) - block(i, j)) < 1e-9);
        }
    }
}

void test_constant_block_has_only_dc_coefficient()
{
    // Un bloc constant v ne varie pas : seul D_{0,0} = 8 v est non nul.
    const Dct dct{};
    const Matrix8 coefficients{dct.forward(Matrix8{10.0})};
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            const double expected{k == 0 && l == 0 ? 80.0 : 0.0};
            assert(std::abs(coefficients(k, l) - expected) < 1e-9);
        }
    }
}

void test_image_is_cropped_to_multiple_of_8()
{
    const Image image{19, 17};
    const Image cropped{image.cropped_to_blocks()};
    assert(cropped.width() == 16);
    assert(cropped.height() == 16);
    assert(image.width() == 19);   // l'original n'est pas modifié
}

void test_image_invariant()
{
    ASSERT_THROWS((Image{0, 8}), std::invalid_argument);
    ASSERT_THROWS((Image{7, 7}.cropped_to_blocks()), std::invalid_argument);

    Image image{8, 8, 1.0};
    image.at(7, 7, 2) = 3.0;
    assert(image(7, 7, 2) == 3.0);
    ASSERT_THROWS((image.at(8, 0, 0)), std::out_of_range);
    ASSERT_THROWS((image.at(0, 0, 3)), std::out_of_range);

    // Limite de taille : au-delà, les calculs de taille de stb pourraient déborder.
    const Image widest{Image::max_dimension, 1};
    assert(widest.width() == Image::max_dimension);
    ASSERT_THROWS((Image{Image::max_dimension + 1, 1}), std::invalid_argument);
}

void test_quantization_tables()
{
    const QuantizationTable q{QuantizationTable::standard()};
    assert(q(0, 0) == 16.0);
    assert(q(7, 7) == 99.0);
    assert(q.scaled(5.0)(7, 7) == 495.0);
    assert(QuantizationTable::low_frequencies()(0, 2) == 1.0);
    assert(QuantizationTable::low_frequencies()(3, 0) == 10.0);
    assert(QuantizationTable::low_frequencies()(4, 4) == 1000.0);
    assert(QuantizationTable::high_frequencies()(0, 0) == 16.0);
    assert(QuantizationTable::high_frequencies()(4, 4) == 1000.0);
    assert(QuantizationTable::high_frequencies()(7, 7) == 1.0);

    ASSERT_THROWS((QuantizationTable::uniform(0.5)), std::invalid_argument);
    ASSERT_THROWS((q.scaled(0.0)), std::invalid_argument);
    ASSERT_THROWS((QuantizationTable::uniform(1.0).scaled(0.5)), std::invalid_argument);
}

void test_frequency_masks()
{
    const SquareMask square{6};
    const TriangleMask triangle{6};
    // Utilisation par référence sur l'interface : l'appel virtuel exécute la
    // version de la classe réelle.
    const FrequencyMask& as_square{square};
    const FrequencyMask& as_triangle{triangle};

    assert(as_square.keeps(5, 5));
    assert(!as_square.keeps(6, 0));
    assert(!as_square.keeps(0, 7));
    assert(as_triangle.keeps(3, 2));
    assert(!as_triangle.keeps(3, 3));
    assert(!as_triangle.keeps(6, 0));

    ASSERT_THROWS((SquareMask{0}), std::invalid_argument);
    ASSERT_THROWS((SquareMask{9}), std::invalid_argument);
    ASSERT_THROWS((TriangleMask{16}), std::invalid_argument);
}

void test_sparse_matrix_from_dense()
{
    // Exemple du commentaire de sparse_matrix.hpp.
    const std::vector<std::int16_t> dense{5, 0, 0,
                                          0, 0, 0,
                                          0, 3, 7};
    const SparseMatrix matrix{3, 3, dense};

    assert(matrix.nnz() == 3);
    assert((matrix.values() == std::vector<std::int16_t>{5, 3, 7}));
    assert((matrix.column_indices() == std::vector<std::int32_t>{0, 1, 2}));
    assert((matrix.row_pointers() == std::vector<std::int32_t>{0, 1, 1, 3}));
    assert(matrix.at(2, 1) == 3);
    assert(matrix.at(1, 1) == 0);
    assert(matrix.to_dense() == dense);
    assert(matrix.storage_bytes() == 3 * 2 + 3 * 4 + 4 * 4);
    ASSERT_THROWS((matrix.at(3, 0)), std::out_of_range);
}

void test_sparse_matrix_invariant()
{
    const SparseMatrix valid{2, 2, {1, 2}, {0, 1}, {0, 2, 2}};
    assert(valid.at(0, 1) == 2);

    ASSERT_THROWS((SparseMatrix{2, 2, {1, 2}, {0, 1}, {0, 3, 2}}), std::invalid_argument);  // pointeurs décroissants
    ASSERT_THROWS((SparseMatrix{2, 2, {1, 2}, {1, 0}, {0, 2, 2}}), std::invalid_argument);  // colonnes non croissantes
    ASSERT_THROWS((SparseMatrix{2, 2, {1, 2}, {0, 2}, {0, 2, 2}}), std::invalid_argument);  // colonne hors matrice
    ASSERT_THROWS((SparseMatrix{2, 2, {1, 0}, {0, 1}, {0, 2, 2}}), std::invalid_argument);  // zéro stocké
    ASSERT_THROWS((SparseMatrix{2, 2, std::vector<std::int16_t>(3, 0)}), std::invalid_argument);  // taille dense
}

void test_compression_removes_high_frequencies()
{
    const Image image{random_image(19, 17)};
    const CompressedImage compressed{compress(image, QuantizationTable::standard(), 2, SquareMask{6})};

    assert(compressed.width() == 16);
    assert(compressed.height() == 16);
    for (std::size_t c{0}; c < Image::channels; ++c) {
        const std::vector<std::int16_t> plane{compressed.channel(c).to_dense()};
        for (std::size_t top{0}; top < 16; top += block_size) {
            for (std::size_t left{0}; left < 16; left += block_size) {
                for (std::size_t k{0}; k < block_size; ++k) {
                    for (std::size_t l{0}; l < block_size; ++l) {
                        if (k >= 6 || l >= 6) {
                            assert(plane[(top + k) * 16 + left + l] == 0);
                        }
                    }
                }
            }
        }
    }
}

void test_threshold_removes_small_coefficients()
{
    const Image image{random_image(16, 16)};
    const CompressedImage compressed{compress(image, QuantizationTable::standard(), 3, SquareMask{8})};
    for (std::size_t c{0}; c < Image::channels; ++c) {
        for (const std::int16_t value : compressed.channel(c).values()) {
            assert(std::abs(value) >= 3);
        }
    }
    ASSERT_THROWS((compress(image, QuantizationTable::standard(), -1, SquareMask{8})), std::invalid_argument);
}

void test_decompression_stays_in_range()
{
    const Image image{random_image(19, 17)};
    const Image result{decompress(compress(image, QuantizationTable::standard(), 2, SquareMask{6}))};

    assert(result.width() == 16 && result.height() == 16);
    for (std::size_t row{0}; row < result.height(); ++row) {
        for (std::size_t col{0}; col < result.width(); ++col) {
            for (std::size_t channel{0}; channel < Image::channels; ++channel) {
                assert(result(row, col, channel) >= 0.0 && result(row, col, channel) <= 255.0);
            }
        }
    }
}

void test_pipeline_on_uniform_image()
{
    const Image image{16, 16, 0.5 * 255.0};
    const Image result{decompress(compress(image, QuantizationTable::standard(), 2, SquareMask{6}))};

    // Test Python : mse < 0.01 pour des intensités dans [0, 1], soit 0.01 * 255^2 ici.
    const double mse{255.0 * 255.0 / std::pow(10.0, psnr(image, result) / 10.0)};
    assert(mse < 0.01 * 255.0 * 255.0);
    assert(relative_l2_error(image, result) < 0.01);
}

void test_csr_file_round_trip()
{
    const Image image{random_image(24, 16)};
    const CompressedImage compressed{
        compress(image, QuantizationTable::standard().scaled(2.0), 1, TriangleMask{10})};

    const std::string path{"test_round_trip.csr"};
    save_compressed(compressed, path);
    const CompressedImage loaded{load_compressed(path)};
    std::filesystem::remove(path);

    assert(loaded.width() == compressed.width());
    assert(loaded.height() == compressed.height());
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            assert(loaded.table()(k, l) == compressed.table()(k, l));
        }
    }
    for (std::size_t c{0}; c < Image::channels; ++c) {
        assert(loaded.channel(c).to_dense() == compressed.channel(c).to_dense());
    }
    assert(relative_l2_error(decompress(compressed), decompress(loaded)) == 0.0);

    ASSERT_THROWS((load_compressed("fichier_absent.csr")), std::runtime_error);

    // Un fichier qui annonce une image trop grande est rejeté avant toute allocation.
    const std::string oversized_path{"test_oversized.csr"};
    {
        std::ofstream out{oversized_path, std::ios::binary};
        const std::uint32_t width{static_cast<std::uint32_t>(Image::max_dimension + block_size)};
        const std::uint32_t height{static_cast<std::uint32_t>(block_size)};
        const double divisor{1.0};
        out.write("JCSR", 4);
        out.write(reinterpret_cast<const char*>(&width), sizeof width);
        out.write(reinterpret_cast<const char*>(&height), sizeof height);
        for (std::size_t i{0}; i < block_size * block_size; ++i) {
            out.write(reinterpret_cast<const char*>(&divisor), sizeof divisor);
        }
    }
    ASSERT_THROWS((load_compressed(oversized_path)), std::invalid_argument);
    std::filesystem::remove(oversized_path);
}

void test_metrics()
{
    const Image image{random_image(8, 8)};
    assert(relative_l2_error(image, image) == 0.0);
    assert(std::isinf(psnr(image, image)));

    const Image ten{8, 8, 10.0};
    const Image twelve{8, 8, 12.0};
    assert(std::abs(relative_l2_error(ten, twelve) - 0.2) < 1e-12);
    ASSERT_THROWS((psnr(image, Image{16, 8})), std::invalid_argument);
}

void test_salt_and_pepper()
{
    Image image{random_image(16, 16)};
    const Image copy{image};

    add_salt_and_pepper(image, 0.0);
    assert(relative_l2_error(copy, image) == 0.0);

    add_salt_and_pepper(image, 1.0);
    for (std::size_t row{0}; row < image.height(); ++row) {
        for (std::size_t col{0}; col < image.width(); ++col) {
            assert(image(row, col, 0) == 0.0 || image(row, col, 0) == 255.0);
            assert(image(row, col, 1) == image(row, col, 0));
            assert(image(row, col, 2) == image(row, col, 0));
        }
    }
    ASSERT_THROWS((add_salt_and_pepper(image, 1.5)), std::invalid_argument);
}

void test_parse_options()
{
    const char* const defaults[]{"jpeg_csr", "compress", "photo.png"};
    const Options basic{parse_options(3, defaults)};
    assert(basic.command == Command::compress);
    assert(basic.input == "photo.png");
    assert(basic.table == TableKind::standard && basic.alpha == 1.0 && basic.threshold == 2);
    assert(basic.mask == MaskKind::square && basic.cutoff == 6 && basic.noise == 0.0);

    const char* const custom[]{"jpeg_csr", "compress", "photo.png", "--alpha", "5",
                               "--mask", "triangle", "--cutoff", "10", "--table", "uniform",
                               "--noise", "0.05", "--threshold", "0", "--out", "sortie"};
    const Options tuned{parse_options(17, custom)};
    assert(tuned.alpha == 5.0 && tuned.mask == MaskKind::triangle && tuned.cutoff == 10);
    assert(tuned.table == TableKind::uniform && tuned.noise == 0.05 && tuned.threshold == 0);
    assert(tuned.output == "sortie");

    const char* const decompress_args[]{"jpeg_csr", "decompress", "a.csr", "a.png"};
    const Options back{parse_options(4, decompress_args)};
    assert(back.command == Command::decompress && back.input == "a.csr" && back.output == "a.png");

    const char* const bad_number[]{"jpeg_csr", "compress", "photo.png", "--alpha", "5abc"};
    ASSERT_THROWS((parse_options(5, bad_number)), std::invalid_argument);
    const char* const missing_value[]{"jpeg_csr", "compress", "photo.png", "--alpha"};
    ASSERT_THROWS((parse_options(4, missing_value)), std::invalid_argument);
    const char* const unknown[]{"jpeg_csr", "compresser"};
    ASSERT_THROWS((parse_options(2, unknown)), std::invalid_argument);

    const char* const help[]{"jpeg_csr"};
    assert(parse_options(1, help).command == Command::help);
}

} // namespace

int main()
{
    test_dct_basis_is_orthogonal();
    test_dct_round_trip();
    test_constant_block_has_only_dc_coefficient();
    test_image_is_cropped_to_multiple_of_8();
    test_image_invariant();
    test_quantization_tables();
    test_frequency_masks();
    test_sparse_matrix_from_dense();
    test_sparse_matrix_invariant();
    test_compression_removes_high_frequencies();
    test_threshold_removes_small_coefficients();
    test_decompression_stays_in_range();
    test_pipeline_on_uniform_image();
    test_csr_file_round_trip();
    test_metrics();
    test_salt_and_pepper();
    test_parse_options();
    std::cout << "Tous les tests passent (17 fonctions de test).\n";
    return EXIT_SUCCESS;
}

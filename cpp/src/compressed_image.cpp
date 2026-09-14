#include "compressed_image.hpp"

#include "image.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace jpeg {

CompressedImage::CompressedImage(std::size_t width, std::size_t height,
                                 const QuantizationTable& table,
                                 std::vector<SparseMatrix> channels)
    : width_{width},
      height_{height},
      table_{table},
      channels_{std::move(channels)}
{
    if (width_ == 0 || height_ == 0 || width_ % block_size != 0 || height_ % block_size != 0
        || width_ > Image::max_dimension || height_ > Image::max_dimension) {
        throw std::invalid_argument{"les dimensions compressées doivent être des multiples non nuls de 8, "
                                    "d'au plus " + std::to_string(Image::max_dimension) + " pixels"};
    }
    if (channels_.size() != Image::channels) {
        throw std::invalid_argument{"une image compressée contient exactement 3 canaux"};
    }
    for (const SparseMatrix& channel : channels_) {
        if (channel.rows() != height_ || channel.cols() != width_) {
            throw std::invalid_argument{"un canal CSR n'a pas les dimensions de l'image"};
        }
    }
}

std::size_t CompressedImage::width() const noexcept
{
    return width_;
}

std::size_t CompressedImage::height() const noexcept
{
    return height_;
}

const QuantizationTable& CompressedImage::table() const noexcept
{
    return table_;
}

const SparseMatrix& CompressedImage::channel(std::size_t index) const
{
    return channels_.at(index);
}

std::size_t CompressedImage::nnz() const noexcept
{
    std::size_t total{0};
    for (const SparseMatrix& channel : channels_) {
        total += channel.nnz();
    }
    return total;
}

std::size_t CompressedImage::coefficient_count() const noexcept
{
    return width_ * height_ * Image::channels;
}

double CompressedImage::conservation_rate() const noexcept
{
    return static_cast<double>(nnz()) / static_cast<double>(coefficient_count());
}

std::size_t CompressedImage::storage_bytes() const noexcept
{
    std::size_t total{0};
    for (const SparseMatrix& channel : channels_) {
        total += channel.storage_bytes();
    }
    return total;
}

namespace {

const char signature[4]{'J', 'C', 'S', 'R'};

// Écriture binaire : reinterpret_cast présente l'objet comme une suite
// d'octets (char) qui sont écrits tels quels. Les surcharges de write et de
// read sont choisies par le compilateur selon le type de l'argument.
void write(std::ostream& out, std::uint32_t value)
{
    out.write(reinterpret_cast<const char*>(&value), sizeof value);
}

void write(std::ostream& out, double value)
{
    out.write(reinterpret_cast<const char*>(&value), sizeof value);
}

void write(std::ostream& out, const std::vector<std::int32_t>& values)
{
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(std::int32_t)));
}

void write(std::ostream& out, const std::vector<std::int16_t>& values)
{
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(std::int16_t)));
}

void read(std::istream& in, std::uint32_t& value)
{
    in.read(reinterpret_cast<char*>(&value), sizeof value);
}

void read(std::istream& in, double& value)
{
    in.read(reinterpret_cast<char*>(&value), sizeof value);
}

// Les deux surcharges suivantes lisent values.size() éléments : le vecteur
// doit avoir été dimensionné avant l'appel.
void read(std::istream& in, std::vector<std::int32_t>& values)
{
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(std::int32_t)));
}

void read(std::istream& in, std::vector<std::int16_t>& values)
{
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(std::int16_t)));
}

void check_stream(const std::ios& stream, const std::string& path)
{
    if (!stream) {
        throw std::runtime_error{"erreur de lecture ou d'écriture du fichier '" + path + "'"};
    }
}

} // namespace

void save_compressed(const CompressedImage& image, const std::string& path)
{
    // Le fichier est fermé par le destructeur de std::ofstream (RAII), même si
    // une exception interrompt la fonction.
    std::ofstream out{path, std::ios::binary};
    if (!out) {
        throw std::runtime_error{"impossible de créer le fichier '" + path + "'"};
    }

    out.write(signature, sizeof signature);
    write(out, static_cast<std::uint32_t>(image.width()));
    write(out, static_cast<std::uint32_t>(image.height()));
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            write(out, image.table()(k, l));
        }
    }
    for (std::size_t c{0}; c < Image::channels; ++c) {
        const SparseMatrix& channel{image.channel(c)};
        write(out, static_cast<std::uint32_t>(channel.nnz()));
        write(out, channel.row_pointers());
        write(out, channel.column_indices());
        write(out, channel.values());
    }
    check_stream(out, path);
}

CompressedImage load_compressed(const std::string& path)
{
    std::ifstream in{path, std::ios::binary};
    if (!in) {
        throw std::runtime_error{"impossible d'ouvrir le fichier '" + path + "'"};
    }

    char header[4]{};
    in.read(header, sizeof header);
    if (!in || !std::equal(header, header + sizeof header, signature)) {
        throw std::runtime_error{"'" + path + "' n'est pas un fichier .csr"};
    }

    std::uint32_t width{};
    std::uint32_t height{};
    read(in, width);
    read(in, height);
    Matrix8 divisors{};
    for (std::size_t k{0}; k < block_size; ++k) {
        for (std::size_t l{0}; l < block_size; ++l) {
            read(in, divisors(k, l));
        }
    }
    check_stream(in, path);

    // Vérification avant toute allocation : un fichier corrompu ne doit pas
    // provoquer la réservation de plusieurs gigaoctets.
    if (width > Image::max_dimension || height > Image::max_dimension) {
        throw std::invalid_argument{"dimensions invalides dans '" + path + "'"};
    }
    const std::size_t coefficient_count{static_cast<std::size_t>(width) * height};

    std::vector<SparseMatrix> channels;
    for (std::size_t c{0}; c < Image::channels; ++c) {
        std::uint32_t nnz{};
        read(in, nnz);
        check_stream(in, path);
        if (nnz > coefficient_count) {
            throw std::invalid_argument{"nombre de coefficients invalide dans '" + path + "'"};
        }

        std::vector<std::int32_t> row_pointers(static_cast<std::size_t>(height) + 1);
        std::vector<std::int32_t> column_indices(nnz);
        std::vector<std::int16_t> values(nnz);
        read(in, row_pointers);
        read(in, column_indices);
        read(in, values);
        check_stream(in, path);

        // Le constructeur de SparseMatrix vérifie l'invariant CSR ; les
        // vecteurs sont déplacés, pas recopiés.
        channels.emplace_back(height, width, std::move(values),
                              std::move(column_indices), std::move(row_pointers));
    }
    // QuantizationTable et CompressedImage vérifient à leur tour leurs invariants.
    return CompressedImage{width, height, QuantizationTable{divisors}, std::move(channels)};
}

} // namespace jpeg

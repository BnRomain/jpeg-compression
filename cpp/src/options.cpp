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
    // The whole text must be consumed: "5abc" is rejected.
    if (used == 0 || used != text.size()) {
        throw std::invalid_argument{"invalid number for " + flag + ": '" + text + "'"};
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
        throw std::invalid_argument{"invalid integer for " + flag + ": '" + text + "'"};
    }
    return value;
}

std::size_t parse_cutoff(const std::string& text)
{
    const int value{parse_int(text, "--cutoff")};
    if (value < 0) {
        throw std::invalid_argument{"--cutoff must be non-negative"};
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
    throw std::invalid_argument{"unknown table: '" + text + "' (standard, uniform, low or high)"};
}

MaskKind parse_mask(const std::string& text)
{
    if (text == "square") {
        return MaskKind::square;
    }
    if (text == "triangle") {
        return MaskKind::triangle;
    }
    throw std::invalid_argument{"unknown mask: '" + text + "' (square or triangle)"};
}

} // namespace

Options parse_options(int argc, const char* const argv[])
{
    // The arguments are copied into std::string: comparisons with == instead of strcmp.
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
            throw std::invalid_argument{"usage: decompress <file.csr> <output.png>"};
        }
        options.command = Command::decompress;
        options.input = args[1];
        options.output = args[2];
        return options;
    }

    if (args[0] != "compress") {
        throw std::invalid_argument{"unknown command: '" + args[0] + "'"};
    }
    if (args.size() < 2) {
        throw std::invalid_argument{"usage: compress <image> [options]"};
    }
    options.command = Command::compress;
    options.input = args[1];

    // Options come in pairs: --name value.
    for (std::size_t i{2}; i < args.size(); i += 2) {
        const std::string& flag{args[i]};
        if (i + 1 >= args.size()) {
            throw std::invalid_argument{"missing value after " + flag};
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
            throw std::invalid_argument{"unknown option: " + flag};
        }
    }
    return options;
}

void print_usage(std::ostream& out)
{
    out << "DCT image compression with CSR sparse storage (C++ version).\n"
           "\n"
           "Usage:\n"
           "  jpeg_csr compress <image> [options]\n"
           "  jpeg_csr decompress <file.csr> <output.png>\n"
           "  jpeg_csr help\n"
           "\n"
           "compress options (the defaults match the Python version):\n"
           "  --table standard|uniform|low|high   quantization matrix Q          [standard]\n"
           "  --alpha A                           quality factor: Q -> A * Q     [1]\n"
           "  --threshold S                       zero coefficients with |c| < S [2]\n"
           "  --mask square|triangle              truncation shape               [square]\n"
           "  --cutoff F                          cutoff frequency               [6]\n"
           "  --noise P                           salt-and-pepper noise, prob. P [0]\n"
           "  --out DIR                           output directory               [results]\n";
}

} // namespace jpeg

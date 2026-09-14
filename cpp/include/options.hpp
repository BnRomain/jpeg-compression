#ifndef JPEG_OPTIONS_HPP
#define JPEG_OPTIONS_HPP

#include <cstddef>
#include <iosfwd>
#include <string>

namespace jpeg {

enum class Command { help, compress, decompress };
enum class TableKind { standard, uniform, low_frequencies, high_frequencies };
enum class MaskKind { square, triangle };

// Settings read from the command line. The defaults reproduce the Python app:
// standard Q, alpha = 1, threshold = 2, square truncation F = 6.
//
// A plain aggregate of parameters, deliberately left as a struct: the validity
// of the values (alpha > 0, F in the right range, noise in [0, 1]...) is
// checked by the classes that use them, in their constructors.
struct Options {
    Command command{Command::help};
    std::string input;
    std::string output{"results"};     // directory (compress) or PNG file (decompress)
    TableKind table{TableKind::standard};
    double alpha{1.0};
    int threshold{2};
    MaskKind mask{MaskKind::square};
    std::size_t cutoff{6};
    double noise{0.0};                 // probability of the salt-and-pepper noise
};

// Parses the program arguments.
// Throws std::invalid_argument if the command is malformed.
Options parse_options(int argc, const char* const argv[]);

void print_usage(std::ostream& out);

} // namespace jpeg

#endif

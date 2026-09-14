# DCT + CSR Image Compression: C++ Version

C++20 port of the project's [Python version](../python): JPEG-inspired image
compression (DCT on 8x8 blocks, quantization, high-frequency truncation) with
the coefficients stored in the CSR sparse format. The `jpeg_csr` program does on
the command line what `jpeg_compression.py` and the Streamlit app do, and
reproduces the analyses of the MAM3 report.

C++ programming project, MAM4 at Polytech Nice Sophia: Romain Ben and Karim Zrig.

- Summary report (2 pages, French): [`docs/report-fr.pdf`](docs/report-fr.pdf)
- Slides (French): [`docs/slides-fr.pdf`](docs/slides-fr.pdf)

## Build

Requirements: a C++20 `g++` (GCC 10 or newer) and `make`. On Windows, MSYS2
provides `g++` and `mingw32-make`, to use instead of `make`.

```bash
make                    # builds ./jpeg_csr
make test               # runs the unit tests
make demo               # compresses images/astronaut.png with several settings
make test SANITIZE=1    # tests under AddressSanitizer and UBSan (Linux, macOS)
```

Compiler flags from the course: `-std=c++20 -Wall -Wextra -pedantic`.

## Usage

```bash
./jpeg_csr compress images/astronaut.png --alpha 5 --out results/alpha5
./jpeg_csr decompress results/alpha5/astronaut.csr results/decoded.png
```

| Option | Purpose | Default |
|---|---|---|
| `--table standard\|uniform\|low\|high` | quantization matrix Q | `standard` |
| `--alpha A` | quality factor: Q becomes A x Q | `1` |
| `--threshold S` | zeroes the quantized coefficients with absolute value < S | `2` |
| `--mask square\|triangle` | square (k, l < F) or triangular (k + l < F) truncation | `square` |
| `--cutoff F` | cutoff frequency | `6` |
| `--noise P` | salt-and-pepper noise with probability P, added before compression | `0` |
| `--out DIR` | output directory | `results` |

The defaults are those of the Python app. `compress` writes `<name>.csr` (the
three CSR matrices, the equivalent of the `.npz` file) and
`<name>_reconstructed.png`, then prints the Streamlit metrics:

```text
Coefficients   : 45566 non-zero out of 786432 (retention rate 5.79 %)
Quality        : relative L2 error 5.43 %, PSNR 30.49 dB
Dense memory   : 6144.00 KiB (float64, like img.nbytes)
CSR memory     : 273.00 KiB (int16 values, int32 indices)
Memory gain    : dense / CSR = 22.51x
.csr file      : 273.52 KiB, source image 773.00 KiB (source / .csr = 2.83x)
Time           : compression 28.23 ms, decompression 21.55 ms
```

## Code organization

```text
cpp/
├── Makefile
├── include/           one commented header per module
├── src/               definitions and main.cpp
├── tests/             unit tests (assert)
├── scripts/           comparison with Python, figure generation
├── images/            test images
├── third_party/       stb_image and stb_image_write (public domain)
└── docs/              report and slides (LaTeX and PDF, French)
```

| Module | Purpose | Course concepts |
|---|---|---|
| `Matrix8` | 8x8 block and matrix product | `std::array`, `operator()` and `operator*` overloads, rule of zero |
| `Dct` | matrix P computed once, D = P M Pᵀ and M = Pᵀ D P | class, member initializer list, `const` methods |
| `QuantizationTable` | standard, uniform, low- or high-frequency Q, alpha factor | invariant (divisors >= 1), `explicit`, exceptions |
| `FrequencyMask` | interface implemented by `SquareMask` (Python) and `TriangleMask` (assignment) | abstract class, `virtual`, `override`, virtual destructor |
| `Image` | RGB image, cropping to multiples of 8 | invariant, `std::vector`, `const` and non-`const` access, checked `at()` |
| `image_io` | PNG, JPEG and BMP reading, PNG writing with stb | RAII (`StbPixels`), deleted copy (`= delete`) |
| `SparseMatrix` | hand-written CSR matrix | checked invariant, move semantics (`std::move`) |
| `CompressedImage` | 3 CSR matrices + Q, binary `.csr` file | composition, `std::ofstream` streams (RAII) |
| `codec` | `compress` and `decompress` | `const T&`, reference to the interface |
| `metrics`, `noise` | relative L2 error, PSNR, salt-and-pepper noise | `T&` to modify, `<random>` |
| `options`, `main` | command line and output | `std::string`, `enum class`, `try` / `catch` |

## Mapping to the Python version

| Python (`jpeg_compression.py`, `app.py`) | C++ |
|---|---|
| `init(img)` | `Image::cropped_to_blocks()` and centering in `compress` |
| `DCT2_P()`, `D_matrix(img_8, P)` | `Dct` class |
| `compression(img, threshold)` | `compress(image, table, threshold, mask)` |
| `decompression(img_compressed)` | `decompress(compressed)` |
| `csr_matrix(channel.astype(np.int16))` | `SparseMatrix` class |
| `np.savez_compressed(...)` | `save_compressed` and `load_compressed` |
| app metrics | output of `jpeg_csr compress` |

## Validation

- `make test` mirrors the pytest tests of the Python version (orthogonality of P,
  cropping, high-frequency removal, bounds, CSR conversion) and also checks the
  invariant of each class, the round trip through a `.csr` file and the
  command-line parsing. The GitHub CI runs them again under sanitizers.
- `python scripts/compare_with_python.py images/astronaut.png` compresses the same
  image with both versions: 786,407 of the 786,432 coefficients are identical
  and the number of non-zero coefficients is the same. The other 25 differ by 1:
  in Python, the image goes through /255 then x255, and D/Q is for instance
  23.999999999999996 instead of 24, which truncation brings down to 23. The C++
  version is about 11 times faster (50 ms versus 543 ms for compression and
  decompression).
- `python scripts/make_figures.py` regenerates the figures in `docs/figures`.

## Security

- Only the PNG, JPEG and BMP decoders of stb are compiled and images are limited
  to 16384 pixels per side (`Image::max_dimension`): the size computations of
  stb, done with `int`, cannot overflow.
- The stb copies compute buffer sizes with `size_t` (CodeQL alerts
  `cpp/integer-multiplication-cast-to-long`): the changes are listed in
  [`third_party/README.md`](third_party/README.md).
- A `.csr` file is fully validated when it is read (dimensions, CSR invariant,
  Q matrix) before any decompression.

## Credits

- [stb](https://github.com/nothings/stb) by Sean Barrett, public domain (modified
  copy, see [`third_party/README.md`](third_party/README.md)).
- Test images from [scikit-image](https://scikit-image.org/):
  `astronaut.png` (NASA, public domain) and `coffee.png` (Rachel Michetti, CC0).

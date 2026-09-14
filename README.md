# 📷 Image Compression via DCT & Sparse Matrices (CSR)

[![Python tests](https://github.com/BnRomain/jpeg-compression/actions/workflows/tests.yml/badge.svg)](https://github.com/BnRomain/jpeg-compression/actions/workflows/tests.yml)
[![C++ tests](https://github.com/BnRomain/jpeg-compression/actions/workflows/cpp-tests.yml/badge.svg)](https://github.com/BnRomain/jpeg-compression/actions/workflows/cpp-tests.yml)
[![CodeQL](https://github.com/BnRomain/jpeg-compression/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/BnRomain/jpeg-compression/actions/workflows/github-code-scanning/codeql)
[![Release](https://img.shields.io/github/v/release/BnRomain/jpeg-compression?sort=semver)](https://github.com/BnRomain/jpeg-compression/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jpeg-csr-compression.streamlit.app/)

This project is a custom implementation of image compression inspired by the **JPEG** standard, using the **Discrete Cosine Transform (DCT)** and storage optimization through the **CSR (Compressed Sparse Row)** format.

The application is interactive and built with **Streamlit**.

The project now comes in two versions:

| Version | Folder | Context | Authors |
|---|---|---|---|
| **Python** (Streamlit app) | [`python/`](python) | Applied mathematics project (MAM3), January 2026 | Romain Ben, Evrard Lecureur, Zouhair Saitout |
| **C++20** (command line) | [`cpp/`](cpp) | C++ programming project (MAM4), September 2026 | Romain Ben, Karim Zrig |

Both versions were developed at Polytech Nice Sophia (Université Côte d'Azur).

## 🚀 Project Overview
The goal is to show how zeroing specific frequencies in the frequency domain (DCT) creates a "sparse" matrix, which can then be stored far more compactly than a raw image.



## 🛠️ How It Works

The algorithm follows the classic image processing steps:
1. **Block splitting**: the image is processed in $8 \times 8$ pixel blocks on the three **RGB** channels.
2. **DCT-II**: conversion from the spatial domain to the frequency domain through a change-of-basis matrix $P$.
3. **Quantization & thresholding**:
   - division by a standard quantization matrix $Q$;
   - an adjustable threshold: coefficients below the threshold are set to zero;
   - removal of high frequencies (truncation of the matrix $D$).
4. **Sparse storage**: the dense matrices are converted to the **CSR** (Compressed Sparse Row) format, which keeps only the non-zero values.
5. **Reconstruction**: the inverse DCT ($P^T D P$) is applied to display the reconstructed image.



## 📊 Compression Analysis
The app displays real-time metrics to compare the efficiency of the algorithm:
* **RAM data**: the size of the "unfolded" image in memory (pixel by pixel).
* **CSR size**: the actual size of the compressed matrices (useful data + indices).
* **Compression ratio**: the reduction factor between the raw volume and the optimized storage.

> **💡 Technical note:** the difference between the original file (e.g. a 200 KB PNG) and the "RAM data" (e.g. 50 MB) is expected. The original file is already compressed by system codecs. My algorithm works on the raw data to show the mathematical gain of the CSR format.

## 🔗 Live Demo
👉 [Compress an image](https://jpeg-csr-compression.streamlit.app/)

To run the app locally:

```bash
cd python
pip install -r requirements.txt
streamlit run app.py
```

## ⚙️ C++ Version

The C++ version reproduces everything the Python version does, without NumPy or SciPy: DCT, quantization with a quality factor $\alpha$, threshold, high-frequency removal, hand-written CSR matrices and a binary `.csr` file (the equivalent of the `.npz` file). It also adds the analyses of the report as options: alternative quantization matrices, the triangular truncation of the assignment and salt-and-pepper noise.

```bash
cd cpp
make test                                            # unit tests
make                                                 # builds ./jpeg_csr
./jpeg_csr compress images/astronaut.png --alpha 5   # compression + metrics
./jpeg_csr decompress results/astronaut.csr decoded.png
```

On Windows with MSYS2, use `mingw32-make` instead of `make`.

On a $512 \times 512$ image, both versions produce the same coefficients (25 floating-point rounding differences out of 786,432) and the C++ version is about **11 times faster**. Architecture, options and results: [`cpp/README.md`](cpp/README.md).

## 🗂️ Repository Structure

```text
jpeg-compression/
├── python/                   Python version (MAM3)
│   ├── app.py                Streamlit app
│   ├── jpeg_compression.py   compression and decompression
│   ├── requirements.txt      dependencies (pinned versions)
│   ├── requirements-dev.txt  test and lint dependencies
│   ├── pyproject.toml        pytest and coverage configuration
│   ├── tests/                pytest tests
│   └── docs/                 report and slides (French)
├── cpp/                      C++ version (MAM4)
│   ├── include/  src/        source code
│   ├── tests/                unit tests
│   ├── third_party/          stb_image and stb_image_write
│   ├── scripts/              comparison with Python, figures
│   └── docs/                 summary report and slides (French)
├── .github/                  workflows, issue and pull request templates, Dependabot
├── CITATION.cff              citation metadata
├── CODE_OF_CONDUCT.md        code of conduct
├── CONTRIBUTING.md           contributing guide
├── LICENSE                   MIT License
├── SECURITY.md               security policy
└── ruff.toml                 Python lint configuration
```

## ✅ Tests and Quality

On every pull request and every push to `main`, GitHub Actions runs:
* **Python tests**: Ruff lint and format check, then `pytest` with a coverage report on the compression functions and the app;
* **C++ tests**: `make test` under AddressSanitizer and UBSan, `make demo`, then the line coverage of the tests and the demo with gcovr;
* **Dependency review**: blocks a pull request that adds a vulnerable dependency;
* **CodeQL**: security analysis of the Python code, the C++ code and the workflows.

The `main` branch is protected: every change goes through a pull request and can only be merged once these checks pass. Coverage reports are published in the summary of each workflow run, and secret scanning with push protection blocks any committed credential.

Versions follow [Semantic Versioning](https://semver.org/) and are published as [GitHub releases](https://github.com/BnRomain/jpeg-compression/releases): see the [contributing guide](CONTRIBUTING.md#versioning-and-releases).

**Dependabot** monitors the Python dependencies and the GitHub Actions. Patch and minor updates are merged automatically once the required checks of `main` have passed. See also the [security policy](SECURITY.md) and the [wiki](https://github.com/BnRomain/jpeg-compression/wiki).

## 📄 Report & Slides

If you are interested in the theory and the full analysis of this project, you can read (in French):

- **📑 Full report**: [read the report](python/docs/report-fr.pdf)
- **📊 Slides**: [view the slides](python/docs/slides-fr.pdf)

These documents cover:
- the DCT & CSR algorithm
- the compression results and metrics
- illustrations and visual comparisons

For the C++ version:

- **📑 Summary report (2 pages)**: [read the report](cpp/docs/report-fr.pdf)
- **📊 Slides**: [view the slides](cpp/docs/slides-fr.pdf)

## 🤝 Contributing

Contributions are welcome. Please read the [contributing guide](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md) before opening an issue or a pull request. Security vulnerabilities must be reported privately, as described in the [security policy](SECURITY.md).

## 📜 License

This project is released under the [MIT License](LICENSE). The vendored stb headers in `cpp/third_party/` keep their own license (public domain or MIT), and the sample images are in the public domain (NASA) or under CC0.

## 📚 Citation

To cite this project, use the metadata in [`CITATION.cff`](CITATION.cff) or the "Cite this repository" button on GitHub.

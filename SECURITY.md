# Security Policy

## Supported Versions

Only the `main` branch is maintained. It contains two implementations, both covered by this policy:

| Component | Path | Supported |
| --- | --- | --- |
| Python implementation and Streamlit app | `python/` | Yes |
| C++ implementation (`jpeg_csr`) | `cpp/` | Yes |
| Vendored third-party code (stb) | `cpp/third_party/` | Yes, see [Third-Party Code](#third-party-code) |
| GitHub Actions workflows and Dependabot configuration | `.github/` | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please do not disclose it publicly through a GitHub issue.

Instead, please report it privately to the project maintainer through GitHub's private vulnerability reporting: [report a vulnerability](https://github.com/BnRomain/jpeg-compression/security/advisories/new).

When reporting a vulnerability, please provide:

* A short description of the vulnerability
* The affected file or component
* The steps required to reproduce the issue
* Any relevant screenshots, logs, or code examples

## Scope

This policy applies to the source code and configuration contained in this repository, in particular:

* the decoding of user-supplied images (upload in `python/app.py`, `load_image` in `cpp/src/image_io.cpp`);
* the reading of compressed `.csr` files (`load_compressed` in `cpp/src/compressed_image.cpp`);
* the GitHub Actions workflows and the Dependabot configuration.

As this is an educational project, security issues should be reported responsibly so they can be reviewed and addressed without unnecessarily exposing other users or contributors.

## Third-Party Code

`cpp/third_party/` contains copies of `stb_image.h` (2.30) and `stb_image_write.h` (1.16) from [nothings/stb](https://github.com/nothings/stb). They are not managed by a package manager, so Dependabot cannot track them: new upstream versions and advisories must be checked manually.

These copies are patched to compute buffer sizes with `size_t` (CodeQL rule `cpp/integer-multiplication-cast-to-long`), only the PNG, JPEG and BMP decoders are compiled, and images are limited to 16384 pixels per side. The patches and the update procedure are listed in [`cpp/third_party/README.md`](cpp/third_party/README.md).

## Security Measures

* **CodeQL** code scanning on Python, C/C++ and GitHub Actions for every pull request and every push to `main`.
* **Dependabot** version updates for the Python dependencies (pinned in `python/requirements*.txt`) and the GitHub Actions, plus Dependabot security updates. Patch and minor updates are merged automatically only once the required checks of `main` have passed.
* **Dependency review** blocks pull requests that introduce dependencies with known vulnerabilities of moderate severity or higher.
* **Secret scanning** with push protection.
* **Sanitizers**: the C++ unit tests run under AddressSanitizer and UndefinedBehaviorSanitizer in CI.
* **Input validation** in the C++ program: every class checks its invariant in its constructor, and `.csr` files are fully validated (dimensions, CSR structure, quantization matrix) before being decompressed.

## Response

Security reports will be reviewed as soon as reasonably possible.

Depending on the nature and severity of the issue, appropriate corrective actions may include:

* Fixing the vulnerability
* Updating dependencies
* Improving the code or configuration
* Documenting the issue and its resolution

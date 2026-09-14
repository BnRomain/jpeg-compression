# Contributing

Thank you for your interest in this project. It started as a student project at
Polytech Nice Sophia, and bug reports, suggestions and pull requests are welcome.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** or **suggest an improvement** with the
  [issue forms](https://github.com/BnRomain/jpeg-compression/issues/new/choose).
- **Report a security vulnerability** privately, as described in the
  [security policy](SECURITY.md). Please do not open a public issue for it.
- **Open a pull request** for a fix, a test or documentation.

For a larger change, please open an issue first so that we can agree on the approach.

## Development setup

```bash
git clone https://github.com/BnRomain/jpeg-compression.git
cd jpeg-compression
```

### Python version (`python/`)

Requires Python 3.12.

```bash
cd python
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

streamlit run app.py               # run the app
python -m pytest -v                # run the tests
```

Lint from the repository root:

```bash
ruff check python cpp/scripts
```

### C++ version (`cpp/`)

Requires a C++20 compiler (GCC 10 or newer) and `make`. On Windows,
[MSYS2](https://www.msys2.org/) provides `g++` and `mingw32-make`.

```bash
cd cpp
make test                          # unit tests
make test SANITIZE=1               # with AddressSanitizer and UBSan (Linux, macOS)
make demo                          # compress the sample image with several settings
```

## Coding guidelines

The repository provides an [`.editorconfig`](.editorconfig) file: most editors
apply its indentation and whitespace settings automatically.

### C++

- C++20, compiled with `-Wall -Wextra -pedantic`: new code must not add warnings.
- One module per header in `include/`, implemented in `src/`.
- Each class establishes its invariant in its constructor and throws a standard
  exception when it does not hold.
- Prefer the rule of zero (`std::vector`, `std::array`) and use RAII for any
  other resource.
- Comments explain the reasoning, not what the code already says.
- Add or update a test in `tests/test_main.cpp` for every change of behavior.

### Python

- The code must pass `ruff check` with the configuration in [`ruff.toml`](ruff.toml).
- Add or update a pytest test in `python/tests/` for every change of behavior.
- Pin new dependencies to an exact version in `requirements.txt` (or
  `requirements-dev.txt` for development tools) so that Dependabot can track them.

### Keeping both versions consistent

The two implementations must keep producing the same coefficients. If you change
the algorithm, update both versions and run the comparison script:

```bash
cd cpp
make
python scripts/compare_with_python.py images/astronaut.png
```

### Third-party code

`cpp/third_party/` contains patched copies of stb. Any change to these files must
be listed in [`cpp/third_party/README.md`](cpp/third_party/README.md).

## Pull request process

1. Create a branch from `main` with a descriptive name, for example
   `fix/csr-reader-bounds` or `docs/usage-examples`.
2. Keep commits focused, with a short summary in the imperative mood
   (for example "Add a test for the triangle mask").
3. Open a pull request against `main` and fill in the template.
4. The `main` branch is protected: a pull request can only be merged once the
   required checks (`python`, `cpp` and `dependency-review`) pass and the branch
   is up to date with `main`. CodeQL also analyzes every pull request.
5. Update the documentation (READMEs, [wiki](https://github.com/BnRomain/jpeg-compression/wiki))
   when the usage or the behavior changes.

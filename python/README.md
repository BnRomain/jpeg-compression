# Python Version (MAM3 project)

Original implementation of the project: the compression module `jpeg_compression.py`
and the Streamlit app `app.py`. The full project description is in the
[main README](../README.md).

Developed by Romain Ben, Evrard Lecureur and Zouhair Saitout (MAM3, Polytech Nice Sophia).

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Live demo: [jpeg-csr-compression.streamlit.app](https://jpeg-csr-compression.streamlit.app/)

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -v
```

- `tests/test_compression.py`: DCT matrix, cropping, compression, decompression, CSR conversion;
- `tests/test_app.py`: Streamlit app startup.

The code is linted with [Ruff](https://docs.astral.sh/ruff/), configured in
[`ruff.toml`](../ruff.toml). From the repository root:

```bash
ruff check python cpp/scripts
```

Dependency versions are pinned in `requirements.txt` and
`requirements-dev.txt`: Dependabot proposes updates every week.

## Documents (French)

- [Report](docs/report-fr.pdf)
- [Slides](docs/slides-fr.pdf)

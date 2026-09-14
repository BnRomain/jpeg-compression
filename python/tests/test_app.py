from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_starts_without_error():
    app = AppTest.from_file(str(APP))
    app.run(timeout=60)

    assert not app.exception
    assert app.title[0].value == "📷 Compression DCT & Matrices CSR"


def test_app_default_threshold_matches_compression_module():
    app = AppTest.from_file(str(APP))
    app.run(timeout=60)

    # Seuil par défaut de l'application = seuil par défaut de compression()
    assert app.sidebar.slider[0].value == 2

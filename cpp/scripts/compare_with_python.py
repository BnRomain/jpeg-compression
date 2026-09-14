"""Compare la version C++ à la version Python sur une même image.

1. Compresse et décompresse l'image avec python/jpeg_compression.py
   (seuil 2, troncature 6x6 : réglages de app.py) et mesure le temps.
2. Lance ./jpeg_csr compress avec ses réglages par défaut, identiques.
3. Relit le fichier .csr écrit par le C++ et compare, coefficient par
   coefficient, les matrices quantifiées des deux versions.

Usage, depuis le dossier cpp/ après `make` :
    python scripts/compare_with_python.py images/astronaut.png
Dépendances : celles de python/requirements.txt (numpy, scipy, opencv).
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix

CPP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CPP_DIR.parent / "python"))
from jpeg_compression import compression, decompression  # noqa: E402

EXECUTABLE = CPP_DIR / ("jpeg_csr.exe" if os.name == "nt" else "jpeg_csr")
REPEAT = 3


def load_rgb(path):
    """Chargement identique à app.py : BGR -> RGB puis réels dans [0, 1]."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0


def read_csr_file(path):
    """Relit le format binaire décrit dans include/compressed_image.hpp."""
    with open(path, "rb") as f:
        assert f.read(4) == b"JCSR", "signature invalide"
        width, height = (int(v) for v in np.frombuffer(f.read(8), dtype="<u4"))
        np.frombuffer(f.read(64 * 8), dtype="<f8")  # matrice Q, non utilisée ici
        planes = []
        for _ in range(3):
            nnz = int(np.frombuffer(f.read(4), dtype="<u4")[0])
            indptr = np.frombuffer(f.read(4 * (height + 1)), dtype="<i4")
            indices = np.frombuffer(f.read(4 * nnz), dtype="<i4")
            values = np.frombuffer(f.read(2 * nnz), dtype="<i2")
            planes.append(csr_matrix((values, indices, indptr), shape=(height, width)).toarray())
    return np.stack(planes, axis=-1)


def time_python(image):
    best = float("inf")
    for _ in range(REPEAT):
        start = time.perf_counter()
        coefficients = compression(image, seuil=2)
        reconstructed = decompression(coefficients)
        best = min(best, time.perf_counter() - start)
    return coefficients, reconstructed, best * 1000


def time_cpp(image_path, output_dir):
    best = float("inf")
    for _ in range(REPEAT):
        run = subprocess.run(
            [str(EXECUTABLE), "compress", str(image_path), "--out", str(output_dir)],
            capture_output=True, text=True, encoding="utf-8", check=True,
        )
        match = re.search(r"compression ([\d.]+) ms, décompression ([\d.]+) ms", run.stdout)
        best = min(best, float(match.group(1)) + float(match.group(2)))
    return best


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    image_path = Path(sys.argv[1])
    output_dir = CPP_DIR / "resultats" / "comparaison"

    image = load_rgb(image_path)
    py_coefficients, py_reconstructed, py_ms = time_python(image)
    cpp_ms = time_cpp(image_path, output_dir)

    cpp_coefficients = read_csr_file(output_dir / f"{image_path.stem}.csr")
    py_coefficients = py_coefficients.astype(np.int16)
    mismatches = np.count_nonzero(py_coefficients != cpp_coefficients)

    cpp_png = cv2.cvtColor(cv2.imdecode(np.fromfile(output_dir / f"{image_path.stem}_reconstruite.png",
                                                    dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    pixel_gap = np.abs(py_reconstructed * 255 - cpp_png.astype(float)).max()

    csr_bytes = sum(
        m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
        for m in (csr_matrix(py_coefficients[:, :, c]) for c in range(3))
    )

    print(f"Image                         : {image_path} {image.shape[1]} x {image.shape[0]}")
    print(f"Coefficients différents       : {mismatches} sur {py_coefficients.size}")
    print(f"Coefficients non nuls         : Python {np.count_nonzero(py_coefficients)}, "
          f"C++ {np.count_nonzero(cpp_coefficients)}")
    print(f"Taille CSR Python (scipy)     : {csr_bytes / 1024:.2f} Ko")
    print(f"Écart max. des pixels         : {pixel_gap:.3f} (sur 255, arrondi PNG compris)")
    print(f"Temps compression + décomp.   : Python {py_ms:.1f} ms, C++ {cpp_ms:.1f} ms "
          f"(C++ {py_ms / cpp_ms:.0f} fois plus rapide, meilleur de {REPEAT} essais)")


if __name__ == "__main__":
    main()

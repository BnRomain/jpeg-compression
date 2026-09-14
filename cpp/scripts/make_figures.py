"""Génère les vignettes du rapport et de la présentation dans docs/figures.

Le script lance ./jpeg_csr avec les réglages étudiés, puis assemble des zooms
alignés sur la grille des blocs 8x8 et agrandis sans lissage, pour que les
artefacts de bloc restent visibles.

Usage, depuis le dossier cpp/ après `make` :
    python scripts/make_figures.py
Dépendances : numpy et opencv (python/requirements.txt).
"""

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

CPP_DIR = Path(__file__).resolve().parents[1]
EXECUTABLE = CPP_DIR / ("jpeg_csr.exe" if os.name == "nt" else "jpeg_csr")
IMAGE = CPP_DIR / "images" / "astronaut.png"
RUNS_DIR = CPP_DIR / "resultats" / "figures"
FIGURES_DIR = CPP_DIR / "docs" / "figures"

RUNS = {
    "alpha1": [],
    "alpha5": ["--alpha", "5"],
    "alpha20": ["--alpha", "20"],
    "uniforme": ["--table", "uniform"],
    "basses": ["--table", "low", "--cutoff", "8"],
    "hautes": ["--table", "high", "--cutoff", "8"],
    "bruit": ["--noise", "0.05", "--mask", "triangle", "--cutoff", "4"],
}

# Zone du visage : coin et taille multiples de 8.
TOP, LEFT, SIZE, ZOOM = 64, 136, 192, 2


def read(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def zoom(image):
    region = image[TOP:TOP + SIZE, LEFT:LEFT + SIZE]
    return cv2.resize(region, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)


def reconstructed(name):
    return zoom(read(RUNS_DIR / name / "astronaut_reconstruite.png"))


def save_strip(images, name):
    gap = np.full((SIZE * ZOOM, 16, 3), 255, dtype=np.uint8)
    parts = [images[0]]
    for image in images[1:]:
        parts += [gap, image]
    ok, buffer = cv2.imencode(".png", np.hstack(parts))
    assert ok
    buffer.tofile(FIGURES_DIR / name)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, options in RUNS.items():
        subprocess.run([str(EXECUTABLE), "compress", str(IMAGE), "--out", str(RUNS_DIR / name), *options],
                       check=True, capture_output=True)

    original = zoom(read(IMAGE))
    save_strip([original, reconstructed("alpha1"), reconstructed("alpha5"), reconstructed("alpha20")], "alpha.png")
    save_strip([reconstructed("alpha1"), reconstructed("uniforme"), reconstructed("basses"),
                reconstructed("hautes")], "tables.png")
    noisy = zoom(read(RUNS_DIR / "bruit" / "astronaut_bruitee.png"))
    save_strip([original, noisy, reconstructed("bruit")], "bruit.png")
    print("Figures écrites dans", FIGURES_DIR)


if __name__ == "__main__":
    main()

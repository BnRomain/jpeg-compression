"""Generate the thumbnails of the report and the slides in docs/figures.

The script runs ./jpeg_csr with the studied settings, then assembles close-ups
aligned with the 8x8 block grid and enlarged without smoothing, so that block
artifacts remain visible.

Usage, from the cpp/ folder after `make` (or after building the three files of
submission/, which the script also finds):
    python scripts/make_figures.py
Dependencies: numpy and opencv (python/requirements.txt).
"""

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CPP_DIR = Path(__file__).resolve().parents[1]


def find_executable():
    """The program built by the Makefile, or the one built in submission/."""
    name = "jpeg_csr.exe" if os.name == "nt" else "jpeg_csr"
    for directory in (CPP_DIR, CPP_DIR / "submission"):
        if (directory / name).exists():
            return directory / name
    sys.exit(f"{name} not found: run make, or build the three files of submission/")


EXECUTABLE = find_executable()
IMAGE = CPP_DIR / "images" / "astronaut.png"
RUNS_DIR = CPP_DIR / "results" / "figures"
FIGURES_DIR = CPP_DIR / "docs" / "figures"

RUNS = {
    "alpha1": [],
    "alpha5": ["--alpha", "5"],
    "alpha20": ["--alpha", "20"],
    "uniform": ["--table", "uniform"],
    "low": ["--table", "low", "--cutoff", "8"],
    "high": ["--table", "high", "--cutoff", "8"],
    "noise": ["--noise", "0.05", "--mask", "triangle", "--cutoff", "4"],
}

# Face area: corner and size are multiples of 8.
TOP, LEFT, SIZE, ZOOM = 64, 136, 192, 2


def read(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def zoom(image):
    region = image[TOP : TOP + SIZE, LEFT : LEFT + SIZE]
    return cv2.resize(region, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)


def reconstructed(name):
    return zoom(read(RUNS_DIR / name / "astronaut_reconstructed.png"))


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
        subprocess.run(
            [str(EXECUTABLE), "compress", str(IMAGE), "--out", str(RUNS_DIR / name), *options],
            check=True,
            capture_output=True,
        )

    original = zoom(read(IMAGE))
    save_strip([original, reconstructed("alpha1"), reconstructed("alpha5"), reconstructed("alpha20")], "alpha.png")
    save_strip(
        [reconstructed("alpha1"), reconstructed("uniform"), reconstructed("low"), reconstructed("high")], "tables.png"
    )
    noisy = zoom(read(RUNS_DIR / "noise" / "astronaut_noisy.png"))
    save_strip([original, noisy, reconstructed("noise")], "noise.png")
    print("Figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()

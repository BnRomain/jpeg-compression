"""Compare the C++ version with the Python version on the same image.

1. Compress and decompress the image with python/jpeg_compression.py
   (threshold 2, 6x6 truncation: the settings of app.py) and time it.
2. Run ./jpeg_csr compress with its default settings, which are identical.
3. Read back the .csr file written by the C++ program and compare the
   quantized matrices of both versions, coefficient by coefficient.

Usage, from the cpp/ folder after `make` (or after building the three files of
submission/, which the script also finds):
    python scripts/compare_with_python.py images/astronaut.png
Dependencies: those of python/requirements.txt (numpy, scipy, opencv).
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

REPEAT = 3


def find_executable():
    """The program built by the Makefile, or the one built in submission/."""
    name = "jpeg_csr.exe" if os.name == "nt" else "jpeg_csr"
    for directory in (CPP_DIR, CPP_DIR / "submission"):
        if (directory / name).exists():
            return directory / name
    sys.exit(f"{name} not found: run make, or build the three files of submission/")


EXECUTABLE = find_executable()


def load_rgb(path):
    """Load the image like app.py: BGR -> RGB, then reals in [0, 1]."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0


def read_csr_file(path):
    """Read the binary format described in include/compressed_image.hpp.

    Same description in section 7 of submission/jpeg.hpp.
    """
    with open(path, "rb") as f:
        assert f.read(4) == b"JCSR", "invalid signature"
        width, height = (int(v) for v in np.frombuffer(f.read(8), dtype="<u4"))
        np.frombuffer(f.read(64 * 8), dtype="<f8")  # Q matrix, not used here
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
        coefficients = compression(image, threshold=2)
        reconstructed = decompression(coefficients)
        best = min(best, time.perf_counter() - start)
    return coefficients, reconstructed, best * 1000


def time_cpp(image_path, output_dir):
    best = float("inf")
    for _ in range(REPEAT):
        run = subprocess.run(
            [str(EXECUTABLE), "compress", str(image_path), "--out", str(output_dir)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        match = re.search(r"compression ([\d.]+) ms, decompression ([\d.]+) ms", run.stdout)
        best = min(best, float(match.group(1)) + float(match.group(2)))
    return best


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    image_path = Path(sys.argv[1])
    output_dir = CPP_DIR / "results" / "comparison"

    image = load_rgb(image_path)
    py_coefficients, py_reconstructed, py_ms = time_python(image)
    cpp_ms = time_cpp(image_path, output_dir)

    cpp_coefficients = read_csr_file(output_dir / f"{image_path.stem}.csr")
    py_coefficients = py_coefficients.astype(np.int16)
    mismatches = np.count_nonzero(py_coefficients != cpp_coefficients)

    cpp_png = cv2.cvtColor(
        cv2.imdecode(
            np.fromfile(output_dir / f"{image_path.stem}_reconstructed.png", dtype=np.uint8), cv2.IMREAD_COLOR
        ),
        cv2.COLOR_BGR2RGB,
    )
    pixel_gap = np.abs(py_reconstructed * 255 - cpp_png.astype(float)).max()

    csr_bytes = sum(
        m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
        for m in (csr_matrix(py_coefficients[:, :, c]) for c in range(3))
    )

    print(f"Image                         : {image_path} {image.shape[1]} x {image.shape[0]}")
    print(f"Different coefficients        : {mismatches} out of {py_coefficients.size}")
    print(
        f"Non-zero coefficients         : Python {np.count_nonzero(py_coefficients)}, "
        f"C++ {np.count_nonzero(cpp_coefficients)}"
    )
    print(f"Python CSR size (SciPy)       : {csr_bytes / 1024:.2f} KiB")
    print(f"Max. pixel gap                : {pixel_gap:.3f} (out of 255, PNG rounding included)")
    print(
        f"Compression + decompression   : Python {py_ms:.1f} ms, C++ {cpp_ms:.1f} ms "
        f"(C++ {py_ms / cpp_ms:.0f} times faster, best of {REPEAT} runs)"
    )


if __name__ == "__main__":
    main()

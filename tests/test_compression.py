import numpy as np
from scipy.sparse import csr_matrix

from jpeg_compression import (
    DCT2_P,
    D_matrix,
    init,
    compression,
    decompression,
)


def test_dct_matrix_shape():
    P = DCT2_P()

    assert P.shape == (8, 8)


def test_dct_matrix_is_orthogonal():
    P = DCT2_P()

    np.testing.assert_allclose(
        P @ P.T,
        np.eye(8),
        atol=1e-12,
    )


def test_init_crops_image_to_multiple_of_8():
    img = np.zeros((17, 19, 3))

    result, x, y = init(img)

    assert x == 16
    assert y == 16
    assert result.shape == (16, 16, 3)


def test_d_matrix():
    rng = np.random.default_rng(42)
    block = rng.random((8, 8))

    P = DCT2_P()
    result = D_matrix(block, P)

    assert result.shape == (8, 8)
    assert np.isfinite(result).all()


def test_compression():
    rng = np.random.default_rng(42)
    img = rng.random((17, 19, 3))

    result = compression(img, seuil=2)

    # Image cropped to multiples of 8
    assert result.shape == (16, 16, 3)

    # No NaN or infinity
    assert np.isfinite(result).all()


def test_compression_removes_high_frequencies():
    rng = np.random.default_rng(42)
    img = rng.random((16, 16, 3))

    result = compression(img, seuil=2)

    for channel in range(3):
        for i in range(0, 16, 8):
            for j in range(0, 16, 8):
                block = result[i:i + 8, j:j + 8, channel]

                # Frequencies >= 6 are explicitly removed
                assert np.all(block[6:, :] == 0)
                assert np.all(block[:, 6:] == 0)


def test_decompression():
    rng = np.random.default_rng(42)
    img = rng.random((17, 19, 3))

    compressed = compression(img, seuil=2)
    result = decompression(compressed)

    assert result.shape == (16, 16, 3)
    assert np.isfinite(result).all()

    # Output must be a valid image between 0 and 1
    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_compression_decompression_pipeline():
    # Simple synthetic image
    img = np.full((16, 16, 3), 0.5)

    compressed = compression(img, seuil=2)
    reconstructed = decompression(compressed)

    assert reconstructed.shape == img.shape
    assert np.isfinite(reconstructed).all()

    # The reconstruction should remain reasonably close
    mse = np.mean((img - reconstructed) ** 2)

    assert mse < 0.01


def test_csr_conversion():
    rng = np.random.default_rng(42)
    img = rng.random((16, 16, 3))

    compressed = compression(img, seuil=2)

    matrices = [
        csr_matrix(compressed[:, :, channel].astype(np.int16))
        for channel in range(3)
    ]

    reconstructed = np.stack(
        [matrix.toarray() for matrix in matrices],
        axis=-1,
    )

    np.testing.assert_array_equal(
        reconstructed,
        compressed.astype(np.int16),
    )

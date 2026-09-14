import numpy as np

# Standard JPEG quantization matrix
Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 13, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def init(img):
    # img is already RGB in [0, 1] (see app.py)
    img_2D = img[:, :, 0]
    x, y = img_2D.shape
    x_new = x - x % 8
    y_new = y - y % 8

    img_new = img[:x_new, :y_new, :3]
    # Back to [0, 255], then centered on 128
    img_new = (img_new * 255) - 128
    return img_new, x_new, y_new


def DCT2_P():
    P = np.zeros((8, 8))
    for i in range(8):
        C = 1 / np.sqrt(2) if i == 0 else 1
        for j in range(8):
            P[i, j] = (1/2) * C * np.cos(((2*j + 1) * i * np.pi) / 16)
    return P


def D_matrix(img_8, P):
    # P is passed as an argument so that it is not recomputed for every block
    D = P @ img_8 @ np.transpose(P)
    return D


def compression(img_input, threshold=2):
    img, x, y = init(img_input)
    img_compressed = np.zeros((x, y, 3))
    P = DCT2_P()

    for channel in range(3):
        for i in range(x // 8):
            for j in range(y // 8):
                img_8 = img[i*8:(i+1)*8, j*8:(j+1)*8, channel]
                D = D_matrix(img_8, P)
                D = np.trunc(D / Q)

                # Frequency removal: small coefficients, then rows and columns >= 6
                D[np.abs(D) < threshold] = 0
                D[6:, :] = 0
                D[:, 6:] = 0

                img_compressed[i*8:(i+1)*8, j*8:(j+1)*8, channel] = D
    return img_compressed


def decompression(img_compressed):
    x, y, _ = img_compressed.shape
    img_uncompressed = np.zeros((x, y, 3))
    P = DCT2_P()

    for channel in range(3):
        for i in range(x // 8):
            for j in range(y // 8):
                img_8 = img_compressed[i*8:(i+1)*8, j*8:(j+1)*8, channel]
                img_8 = img_8 * Q
                img_8_uncompressed = np.transpose(P) @ img_8 @ P
                img_uncompressed[i*8:(i+1)*8, j*8:(j+1)*8, channel] = img_8_uncompressed

    img_uncompressed = img_uncompressed + 128
    img_uncompressed = img_uncompressed / 255
    # Clip to [0, 1] to avoid display errors on overflow
    return np.clip(img_uncompressed, 0, 1)

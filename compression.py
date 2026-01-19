import numpy as np
from scipy.sparse import csr_matrix
from PIL import Image

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
    img = np.array(img)
    x, y, _ = img.shape
    x -= x % 8
    y -= y % 8
    img = img[:x, :y, :3]
    img = img * 255 - 128
    return img, x, y


def DCT2_P():
    P = np.zeros((8, 8))
    for i in range(8):
        C = 1 / np.sqrt(2) if i == 0 else 1
        for j in range(8):
            P[i, j] = 0.5 * C * np.cos(((2*j + 1) * i * np.pi) / 16)
    return P


def D_matrix(img_8):
    P = DCT2_P()
    return P @ img_8 @ P.T


def compression(img, seuil=2):
    img, x, y = init(img)
    img_compressed = np.zeros((x, y, 3))

    for c in range(3):
        for i in range(x // 8):
            for j in range(y // 8):
                bloc = img[i*8:(i+1)*8, j*8:(j+1)*8, c]
                D = D_matrix(bloc)
                D = np.fix(D / Q)
                D[np.abs(D) < seuil] = 0
                D[3:, :] = 0
                D[:, 3:] = 0
                img_compressed[i*8:(i+1)*8, j*8:(j+1)*8, c] = D

    return img_compressed


def to_csr(img_comp):
    csr_channels = []
    for c in range(3):
        csr_channels.append(csr_matrix(img_comp[:, :, c].astype(np.int16)))
    return csr_channels


def decompression(csr_channels):
    P = DCT2_P()
    x, y = csr_channels[0].shape
    img = np.zeros((x, y, 3))

    for c in range(3):
        dense = csr_channels[c].toarray()
        for i in range(x // 8):
            for j in range(y // 8):
                bloc = dense[i*8:(i+1)*8, j*8:(j+1)*8] * Q
                img[i*8:(i+1)*8, j*8:(j+1)*8, c] = P.T @ bloc @ P

    img = (img + 128) / 255
    return np.clip(img, 0, 1)

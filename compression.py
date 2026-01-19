import numpy as np
from scipy.sparse import csr_matrix

# Matrice de quantification JPEG standard (luminance)
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


# =========================
# ESPACES COULEUR
# =========================

def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    Y  = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.1687*R - 0.3313*G + 0.5*B + 128
    Cr = 0.5*R - 0.4187*G - 0.0813*B + 128
    return Y, Cb, Cr


def ycbcr_to_rgb(Y, Cb, Cr):
    R = Y + 1.402*(Cr - 128)
    G = Y - 0.34414*(Cb - 128) - 0.71414*(Cr - 128)
    B = Y + 1.772*(Cb - 128)
    img = np.stack([R, G, B], axis=2)
    return np.clip(img, 0, 255)


# =========================
# DCT
# =========================

def DCT2_P():
    P = np.zeros((8, 8))
    for i in range(8):
        C = 1 / np.sqrt(2) if i == 0 else 1
        for j in range(8):
            P[i, j] = 0.5 * C * np.cos(((2*j + 1) * i * np.pi) / 16)
    return P


P = DCT2_P()


# =========================
# COMPRESSION
# =========================

def compression(img_rgb, seuil=1):
    img = img_rgb.astype(np.float32)
    x, y, _ = img.shape
    x -= x % 8
    y -= y % 8
    img = img[:x, :y]

    Y, Cb, Cr = rgb_to_ycbcr(img)
    Y = Y - 128

    Y_comp = np.zeros_like(Y)

    for i in range(x // 8):
        for j in range(y // 8):
            bloc = Y[i*8:(i+1)*8, j*8:(j+1)*8]
            D = P @ bloc @ P.T
            D = np.round(D / Q)

            # Seuillage + limitation fréquentielle
            D[np.abs(D) < seuil] = 0
            D[5:, :] = 0
            D[:, 5:] = 0

            Y_comp[i*8:(i+1)*8, j*8:(j+1)*8] = D

    # CSR uniquement sur Y (luminance)
    Y_csr = csr_matrix(Y_comp.astype(np.int16))

    return Y_csr, Cb, Cr


# =========================
# DECOMPRESSION
# =========================

def decompression(Y_csr, Cb, Cr):
    Y_q = Y_csr.toarray().astype(np.float32)
    x, y = Y_q.shape

    Y = np.zeros((x, y))

    for i in range(x // 8):
        for j in range(y // 8):
            bloc = Y_q[i*8:(i+1)*8, j*8:(j+1)*8] * Q
            Y[i*8:(i+1)*8, j*8:(j+1)*8] = P.T @ bloc @ P

    Y = Y + 128
    img_rgb = ycbcr_to_rgb(Y, Cb, Cr)
    return img_rgb / 255

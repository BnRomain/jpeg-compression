import numpy as np 
from scipy.sparse import csr_matrix, load_npz
import os

# Matrice de quantification JPEG standard
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
    # img est déjà en RGB (0-1) via l'appli
    img_2D = img[:, :, 0]
    x, y = img_2D.shape
    x_new = x - x % 8
    y_new = y - y % 8

    img_new = img[:x_new, :y_new, :3]
    # On passe en 0-255 puis on centre sur 128 (ton process)
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
    # On passe P en argument pour ne pas le recalculer 1000 fois
    D = P @ img_8 @ np.transpose(P)
    return D 

def compression(img_input, seuil=2):
    img, x, y = init(img_input)
    img_compressed = np.zeros((x, y, 3))
    P = DCT2_P()
    
    for bloc in range(3):
        for i in range(x // 8):
            for j in range(y // 8):
                img_8 = img[i*8:(i+1)*8, j*8:(j+1)*8, bloc]
                D = D_matrix(img_8, P)
                D = np.fix( D / Q ) 
                
                # Ta logique de suppression de fréquences
                D[np.abs(D) < seuil] = 0
                D[6:, :] = 0
                D[:, 6:] = 0 
                
                img_compressed[i*8:(i+1)*8, j*8:(j+1)*8, bloc] = D 
    return img_compressed

def decompression(img_compressed):
    x, y, z = img_compressed.shape
    img_uncompressed = np.zeros((x, y, 3))
    P = DCT2_P() 
    
    for bloc in range(3):
        for i in range(x // 8):
            for j in range(y // 8):
                img_8 = img_compressed[i*8:(i+1)*8, j*8:(j+1)*8, bloc]
                img_8 = img_8 * Q 
                img_8_uncompressed = np.transpose(P) @ img_8 @ P
                img_uncompressed[i*8:(i+1)*8, j*8:(j+1)*8, bloc] = img_8_uncompressed
                
    img_uncompressed = img_uncompressed + 128
    img_uncompressed = img_uncompressed / 255 
    # Clip pour éviter les erreurs d'affichage si dépassement 0-1
    return np.clip(img_uncompressed, 0, 1)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import io
import os

# Import de tes fonctions depuis ton fichier de calcul
from compression.py import compression, decompression, init

st.set_page_config(page_title="Compresseur DCT/CSR", layout="centered")

st.title("📷 Compression RGB via DCT et Matrices CSR")
st.write("Cette application utilise votre algorithme personnalisé de compression par blocs 8x8.")

# 1. Upload
uploaded_file = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg"])

# 2. Paramètres
seuil = st.slider("Seuil de suppression des coefficients", 0, 10, 2)

if uploaded_file is not None:
    # Lecture de l'image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    import cv2
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0 # Normalisation comme plt.imread
    
    col_orig, col_comp = st.columns(2)
    
    with col_orig:
        st.subheader("Originale")
        st.image(img, use_container_width=True)

    if st.button("Lancer la compression"):
        # === ÉTAPE 1 : COMPRESSION (ton code) ===
        # On récupère l'image quantifiée (dense)
        img_comp_dense = compression(img, seuil)
        
        # === ÉTAPE 2 : TRANSFORMATION CSR ET CALCUL TAILLE ===
        taille_csr_totale = 0
        matrices_csr = []
        
        for c in range(3):
            canal_int16 = img_comp_dense[:, :, c].astype(np.int16)
            m_sparse = csr_matrix(canal_int16)
            matrices_csr.append(m_sparse)
            
            # Calcul de la taille réelle en mémoire du format CSR
            # (data + indices + indptr)
            taille_csr_totale += (m_sparse.data.nbytes + 
                                 m_sparse.indices.nbytes + 
                                 m_sparse.indptr.nbytes)

        # === ÉTAPE 3 : DÉCOMPRESSION POUR AFFICHAGE ===
        img_final = decompression(img_comp_dense)
        
        with col_comp:
            st.subheader("Décompressée")
            st.image(img_final, use_container_width=True)

        # === ÉTAPE 4 : COMPARAISON DES TAILLES ===
        st.divider()
        st.subheader("📊 Comparaison des tailles")
        
        # Taille originale (données brutes non compressées)
        taille_brute = img.nbytes 
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Format Brut (RGB)", f"{taille_brute / 1024:.1f} Ko")
        c2.metric("Format CSR", f"{taille_csr_totale / 1024:.1f} Ko")
        
        ratio = taille_brute / taille_csr_totale
        c3.metric("Ratio de Gain", f"{ratio:.2f}x")

        st.info(f"Le format CSR est ici **{ratio:.1f} fois plus léger** que l'image brute en mémoire grâce au seuillage (Seuil = {seuil}).")

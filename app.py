import streamlit as st
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from compression import compression, decompression

st.set_page_config(page_title="JPEG + CSR Compression", layout="centered")

st.title("📷 Analyse de Compression DCT & CSR")
st.write("Visualisation de la compression par blocs 8x8 sur les canaux RGB.")

# 1. Barre latérale pour les réglages
with st.sidebar:
    st.header("Réglages")
    seuil = st.slider("Seuil de quantification", 0, 10, 2)
    st.info("Un seuil plus haut augmente la sparsité et réduit la taille du fichier.")

# 2. Upload de l'image
uploaded_file = st.file_uploader("Importer une image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Lecture et conversion
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    # Affichage Image Originale
    st.subheader("🖼️ Image Originale")
    st.image(img, use_container_width=True)

    if st.button("🚀 Lancer la compression", use_container_width=True):
        with st.spinner("Calcul des matrices DCT..."):
            # === COMPRESSION ===
            img_comp_dense = compression(img, seuil)
            
            # === CALCUL TAILLE CSR ===
            taille_csr_totale = 0
            nnz_total = 0
            pixels_total = img_comp_dense.shape[0] * img_comp_dense.shape[1] * 3
            
            for c in range(3):
                canal_int16 = img_comp_dense[:, :, c].astype(np.int16)
                m_sparse = csr_matrix(canal_int16)
                # Taille : data (valeurs) + indices (colonnes) + indptr (lignes)
                taille_csr_totale += (m_sparse.data.nbytes + m_sparse.indices.nbytes + m_sparse.indptr.nbytes)
                nnz_total += m_sparse.nnz

            # === DECOMPRESSION ===
            img_final = decompression(img_comp_dense)

        # Affichage Image Compressée
        st.subheader("📉 Image après Compression/Décompression")
        st.image(img_final, use_container_width=True)

        # === STATISTIQUES ===
        st.divider()
        st.subheader("📊 Résultats de l'algorithme")
        
        taille_brute = img.nbytes
        ratio = taille_brute / taille_csr_totale
        sparsite = (1 - (nnz_total / pixels_total)) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Taille Initiale", f"{taille_brute / 1024:.1f} Ko")
        col2.metric("Taille CSR", f"{taille_csr_totale / 1024:.1f} Ko")
        col3.metric("Ratio de Gain", f"{ratio:.2f}x")

        st.write(f"**Sparsité globale :** `{sparsite:.2f} %` de zéros dans les matrices DCT.")
        
        if seuil > 4:
            st.warning("⚠️ Un seuil élevé provoque un effet de pixelisation (blocs 8x8 visibles).")

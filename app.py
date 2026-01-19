import streamlit as st
import numpy as np
import cv2
import io
from scipy.sparse import csr_matrix
from compression import compression, decompression

st.set_page_config(page_title="JPEG + CSR Compression", layout="centered")

# === TITRE ET SIGNATURE ===
st.title("📷 Compression DCT & Matrices CSR")
st.sidebar.title("Réglages")
seuil = st.sidebar.slider("Seuil de quantification", 0, 10, 2)
st.sidebar.divider()
st.sidebar.markdown("### 👨‍💻 Crédits")
st.sidebar.caption("Made by Romain Ben")

# === UPLOAD ===
uploaded_file = st.file_uploader("Importer une image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Lecture image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    st.subheader("🖼️ Image Originale")
    st.image(img, use_container_width=True)

    # === BOUTON DE LANCEMENT ===
    if st.button("🚀 Lancer la compression", use_container_width=True):
        with st.spinner("Compression en cours..."):
            # 1. Compression dense (ton code)
            img_comp_dense = compression(img, seuil)
            
            # 2. Transformation CSR et stockage dans la "Session State"
            st.session_state['matrices_csr'] = []
            nnz_total = 0
            taille_csr_totale = 0
            
            for c in range(3):
                canal_int16 = img_comp_dense[:, :, c].astype(np.int16)
                m_sparse = csr_matrix(canal_int16)
                st.session_state['matrices_csr'].append(m_sparse)
                
                nnz_total += m_sparse.nnz
                taille_csr_totale += (m_sparse.data.nbytes + m_sparse.indices.nbytes + m_sparse.indptr.nbytes)

            # 3. Décompression pour affichage
            st.session_state['img_final'] = decompression(img_comp_dense)
            
            # 4. Sauvegarde des stats
            st.session_state['taille_brute'] = img.nbytes
            st.session_state['taille_csr'] = taille_csr_totale
            st.session_state['nnz'] = nnz_total

    # === AFFICHAGE DES RÉSULTATS (si la compression a déjà été faite) ===
    if 'img_final' in st.session_state:
        st.subheader("📉 Image Reconstruite")
        st.image(st.session_state['img_final'], use_container_width=True)

        # Statistiques
        st.divider()
        col1, col2, col3 = st.columns(3)
        ratio = st.session_state['taille_brute'] / st.session_state['taille_csr']
        
        col1.metric("Données RAM", f"{st.session_state['taille_brute'] / 1024:.1f} Ko")
        col2.metric("Taille CSR", f"{st.session_state['taille_csr'] / 1024:.1f} Ko")
        col3.metric("Ratio de Gain", f"{ratio:.2f}x")

        # === TÉLÉCHARGEMENTS ===
        st.subheader("📥 Téléchargements")
        c_dl1, c_dl2 = st.columns(2)

        # Download PNG
        img_out = (st.session_state['img_final'] * 255).astype(np.uint8)
        _, buffer_img = cv2.imencode('.png', cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
        c_dl1.download_button(
            label="🖼️ Rendu visuel (PNG)",
            data=buffer_img.tobytes(),
            file_name="rendu_compression.png",
            mime="image/png",
            use_container_width=True
        )

        # Download NPZ
        buf_npz = io.BytesIO()
        dict_csr = {
            'canal_R': st.session_state['matrices_csr'][0],
            'canal_G': st.session_state['matrices_csr'][1],
            'canal_B': st.session_state['matrices_csr'][2]
        }
        np.savez_compressed(buf_npz, **dict_csr)
        c_dl2.download_button(
            label="💾 Matrices Sparse (NPZ)",
            data=buf_npz.getvalue(),
            file_name="donnees_compression.npz",
            mime="application/octet-stream",
            use_container_width=True
        )

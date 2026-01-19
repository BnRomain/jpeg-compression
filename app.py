import streamlit as st
import numpy as np
from PIL import Image
import io

from compression import compression, decompression

st.set_page_config(
    page_title="JPEG + CSR Compression",
    layout="centered"
)

st.title("📷 Compression JPEG avec DCT et matrices CSR")
st.write(
    "Démo interactive : compression avec perte basée sur la DCT, "
    "quantification JPEG et stockage sparse (CSR) sur la luminance."
)

# Upload image
uploaded_file = st.file_uploader(
    "Importer une image",
    type=["png", "jpg", "jpeg"]
)

# Paramètre de compression
seuil = st.slider(
    "Seuil de quantification (coefficients DCT négligeables)",
    min_value=0,
    max_value=6,
    value=1
)

if uploaded_file is not None:
    # Chargement image
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img_pil, dtype=np.float32)

    st.subheader("Image originale")
    st.image(img_pil, use_container_width=True)

    if st.button("Lancer la compression"):
        with st.spinner("Compression en cours..."):
            # === COMPRESSION ===
            Y_csr, Cb, Cr = compression(img_np, seuil)

            # === DECOMPRESSION ===
            img_decomp = decompression(Y_csr, Cb, Cr)

        st.subheader("Image après compression + décompression")
        st.image(img_decomp, use_container_width=True)

        # Infos compression
        nnz = Y_csr.nnz
        total = Y_csr.shape[0] * Y_csr.shape[1]
        sparsity = 100 * (1 - nnz / total)

        st.markdown(
            f"""
            **Informations sur la compression :**
            - Taille image : `{Y_csr.shape[0]} × {Y_csr.shape[1]}`
            - Coefficients non nuls (Y) : `{nnz}`
            - Sparsité de la luminance : `{sparsity:.2f} %`
            """
        )

        st.markdown(
            f"""
            # === STATISTIQUES DE COMPRESSION ===
            st.subheader("📊 Analyse de la compression")
            
            # 1. Taille originale (en octets) : 3 canaux (RGB), 8 bits par canal
            taille_origine = img_np.shape[0] * img_np.shape[1] * 3
            
            # 2. Taille compressée (estimation des données stockées)
            # On compte les coefficients Y (CSR), et les canaux Cb, Cr complets
            taille_y_csr = Y_csr.data.nbytes + Y_csr.indices.nbytes + Y_csr.indptr.nbytes
            taille_cb_cr = Cb.nbytes + Cr.nbytes
            taille_totale_comp = taille_y_csr + taille_cb_cr
            
            # Calcul du ratio et gain
            ratio = taille_origine / taille_totale_comp
            gain = (1 - (taille_totale_comp / taille_origine)) * 100
    
            # Affichage des métriques
            col1, col2, col3 = st.columns(3)
            col1.metric("Taille Originale", f"{taille_origine / 1024:.1f} KB")
            col2.metric("Taille Estimée", f"{taille_totale_comp / 1024:.1f} KB", f"-{gain:.1f}%", delta_color="normal")
            col3.metric("Ratio", f"{ratio:.2f}:1")
    
            st.info(f"💡 Grâce au format **CSR**, nous ne stockons que `{nnz}` coefficients non nuls sur `{total}` pour la luminance.")
            """

        # Téléchargement
        img_out = Image.fromarray((img_decomp * 255).astype(np.uint8))
        buf = io.BytesIO()
        img_out.save(buf, format="PNG")

        st.download_button(
            label="📥 Télécharger l'image reconstruite",
            data=buf.getvalue(),
            file_name="image_compressee.png",
            mime="image/png"
        )

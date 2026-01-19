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

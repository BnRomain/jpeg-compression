import streamlit as st
from PIL import Image
import numpy as np
import io

from compression import compression, to_csr, decompression

st.set_page_config(page_title="JPEG + CSR Compression", layout="centered")

st.title("📷 JPEG Compression avec matrices CSR")
st.write("Projet académique – Compression avec perte basée sur DCT et stockage sparse")

uploaded = st.file_uploader("Importer une image", type=["png", "jpg", "jpeg"])
seuil = st.slider("Seuil de quantification", 0, 10, 2)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Image originale", use_container_width=True)

    if st.button("Lancer la compression"):
        img_comp = compression(img, seuil)
        csr = to_csr(img_comp)
        img_decomp = decompression(csr)

        st.image(img_decomp, caption="Image après compression + décompression", use_container_width=True)

        img_out = Image.fromarray((img_decomp * 255).astype(np.uint8))
        buf = io.BytesIO()
        img_out.save(buf, format="PNG")

        st.download_button(
            "Télécharger l'image reconstruite",
            buf.getvalue(),
            "image_compressee.png",
            "image/png"
        )

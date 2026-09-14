import io

import cv2
import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix

from jpeg_compression import compression, decompression

st.set_page_config(page_title="JPEG + CSR Compression", layout="centered")

# === TITLE AND CREDITS ===
st.title("📷 DCT & CSR Image Compression")
st.sidebar.title("Settings")
threshold = st.sidebar.slider("Quantization threshold", 0, 10, 2)
st.sidebar.divider()
st.sidebar.markdown("### 👨‍💻 Credits")
st.sidebar.caption("Made by Romain Ben, Evrard Lecureur, Zouhair Saitout and Karim Zrig")

# === UPLOAD ===
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    st.subheader("🖼️ Original image")
    st.image(img, use_container_width=True)

    # === RUN BUTTON ===
    if st.button("🚀 Run compression", use_container_width=True):
        with st.spinner("Compressing..."):
            # 1. Dense compression
            img_comp_dense = compression(img, threshold)

            # 2. CSR conversion, stored in the session state
            st.session_state['csr_matrices'] = []
            nnz_total = 0
            total_csr_size = 0

            for c in range(3):
                channel_int16 = img_comp_dense[:, :, c].astype(np.int16)
                m_sparse = csr_matrix(channel_int16)
                st.session_state['csr_matrices'].append(m_sparse)

                nnz_total += m_sparse.nnz
                total_csr_size += (m_sparse.data.nbytes + m_sparse.indices.nbytes + m_sparse.indptr.nbytes)

            # 3. Decompression for display
            st.session_state['img_final'] = decompression(img_comp_dense)

            # 4. Save the statistics
            st.session_state['raw_size'] = img.nbytes
            st.session_state['csr_size'] = total_csr_size
            st.session_state['nnz'] = nnz_total

    # === RESULTS (once the compression has run) ===
    if 'img_final' in st.session_state:
        st.subheader("📉 Reconstructed image")
        st.image(st.session_state['img_final'], use_container_width=True)

        # Statistics
        st.divider()
        col1, col2, col3 = st.columns(3)
        ratio = st.session_state['raw_size'] / st.session_state['csr_size']

        col1.metric("RAM data", f"{st.session_state['raw_size'] / 1024:.1f} KiB")
        col2.metric("CSR size", f"{st.session_state['csr_size'] / 1024:.1f} KiB")
        col3.metric("Compression ratio", f"{ratio:.2f}x")

        # === DOWNLOADS ===
        st.subheader("📥 Downloads")
        c_dl1, c_dl2 = st.columns(2)

        # Download PNG
        img_out = (st.session_state['img_final'] * 255).astype(np.uint8)
        _, buffer_img = cv2.imencode('.png', cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
        c_dl1.download_button(
            label="🖼️ Visual result (PNG)",
            data=buffer_img.tobytes(),
            file_name="compression_result.png",
            mime="image/png",
            use_container_width=True
        )

        # Download NPZ
        buf_npz = io.BytesIO()
        csr_dict = {
            'channel_R': st.session_state['csr_matrices'][0],
            'channel_G': st.session_state['csr_matrices'][1],
            'channel_B': st.session_state['csr_matrices'][2]
        }
        np.savez_compressed(buf_npz, **csr_dict)
        c_dl2.download_button(
            label="💾 Sparse matrices (NPZ)",
            data=buf_npz.getvalue(),
            file_name="compression_data.npz",
            mime="application/octet-stream",
            use_container_width=True
        )

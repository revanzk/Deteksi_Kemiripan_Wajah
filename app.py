import streamlit as st
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Kemiripan Wajah",
    page_icon="🧬",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: #f8fafc;
}

/* Header card */
.header-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}
.header-card h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-card p {
    font-size: 0.95rem;
    margin: 0.4rem 0 0 0;
    opacity: 0.85;
}

/* Section card */
.section-card {
    background: white;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border: 1px solid #e8edf3;
}
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #1e3a5f;
    margin: 0 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Result area */
.result-mirip {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border: 2px solid #34d399;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-tidak {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 2px solid #f87171;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-label {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-score {
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1;
}
.result-sub {
    font-size: 0.85rem;
    opacity: 0.75;
    margin-top: 0.3rem;
}

/* Image caption */
.img-caption {
    text-align: center;
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.4rem;
    font-weight: 500;
}

/* Score bar */
.score-bar-wrap {
    background: #e2e8f0;
    border-radius: 99px;
    height: 10px;
    margin: 0.8rem 0;
    overflow: hidden;
}
.score-bar-fill-green {
    height: 10px;
    border-radius: 99px;
    background: linear-gradient(90deg, #34d399, #059669);
    transition: width 0.6s ease;
}
.score-bar-fill-red {
    height: 10px;
    border-radius: 99px;
    background: linear-gradient(90deg, #f87171, #dc2626);
    transition: width 0.6s ease;
}

/* Divider */
.vs-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    font-weight: 700;
    color: #94a3b8;
    padding: 0 0.5rem;
    height: 100%;
}

/* Streamlit overrides */
div[data-testid="stFileUploader"] > label {
    font-size: 0.9rem;
    color: #374151;
}
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 0.95rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.88;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1>🧬 Deteksi Kemiripan Wajah</h1>
    <p>Berbasis PCA · SVD · Eigenfaces · Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

IMG_SIZE = (100, 100)

@st.cache_resource
def get_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

def bytes_to_cv2(uploaded_file):
    """Konversi UploadedFile → numpy array (BGR)."""
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    uploaded_file.seek(0)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def extract_face_vector(img_bgr, detector):
    """
    Deteksi wajah → crop → grayscale → resize → flatten.
    Return None jika wajah tidak terdeteksi.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_crop, IMG_SIZE)
    return face_resized.flatten() / 255.0, (x, y, w, h)

def cv2_to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def draw_box(img_bgr, box):
    """Gambar kotak biru di sekitar wajah."""
    out = img_bgr.copy()
    x, y, w, h = box
    cv2.rectangle(out, (x, y), (x+w, y+h), (37, 99, 235), 2)
    return out

# ══════════════════════════════════════════════════════════════
# STEP 1 – Upload Dataset
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="section-card">
    <p class="section-title">📁 Langkah 1 — Upload Foto Dataset (Referensi)</p>
</div>
""", unsafe_allow_html=True)

dataset_files = st.file_uploader(
    "Pilih beberapa foto wajah sebagai data latih (minimal 3 foto disarankan).",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="dataset_upload",
)

# ══════════════════════════════════════════════════════════════
# STEP 2 – Upload Gambar Baru
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="section-card">
    <p class="section-title">🆕 Langkah 2 — Upload Foto Baru untuk Dicocokkan</p>
</div>
""", unsafe_allow_html=True)

new_file = st.file_uploader(
    "Pilih 1 foto wajah baru yang ingin dicocokkan dengan dataset.",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    key="new_upload",
)

# ── Sidebar settings ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    n_components = st.slider("Jumlah Komponen PCA (k)", min_value=1, max_value=50, value=8)
    threshold = st.slider("Threshold Kemiripan", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    use_face_detection = st.toggle("Gunakan Deteksi Wajah Otomatis", value=True)
    st.markdown("---")
    st.markdown("""
    **Cara penggunaan:**
    1. Upload beberapa foto dataset
    2. Upload 1 foto baru
    3. Klik **Analisis**
    """)
    st.markdown("---")
    st.caption("PCA · SVD · Eigenfaces")

# ══════════════════════════════════════════════════════════════
# STEP 3 – Analisis
# ══════════════════════════════════════════════════════════════

run = st.button("🔍 Analisis Kemiripan", use_container_width=True)

if run:
    if not dataset_files:
        st.error("❌ Harap upload minimal 1 foto dataset terlebih dahulu.")
        st.stop()
    if new_file is None:
        st.error("❌ Harap upload foto baru untuk dicocokkan.")
        st.stop()

    detector = get_detector()

    # ── Proses dataset ────────────────────────────────────────
    with st.spinner("Memproses dataset dan melatih model PCA…"):
        features = []
        dataset_imgs = []         # PIL images asli
        dataset_face_boxes = []   # bounding box
        failed = []

        for f in dataset_files:
            img_bgr = bytes_to_cv2(f)
            if img_bgr is None:
                failed.append(f.name)
                continue

            if use_face_detection:
                vec, box = extract_face_vector(img_bgr, detector)
            else:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, IMG_SIZE)
                vec = resized.flatten() / 255.0
                box = None

            dataset_imgs.append((f.name, img_bgr, box))

            if vec is not None:
                features.append(vec)
            else:
                failed.append(f.name)

        if failed:
            st.warning(f"⚠️ Wajah tidak terdeteksi pada: {', '.join(failed)}")

        if len(features) < 2:
            st.error("❌ Terlalu sedikit wajah yang berhasil dideteksi dari dataset. "
                     "Pastikan foto wajah jelas dan menghadap ke depan, atau matikan deteksi otomatis.")
            st.stop()

        X = np.array(features)
        k = min(n_components, len(features) - 1)
        pca = PCA(n_components=max(1, k), whiten=True)
        X_pca = pca.fit_transform(X)
        mean_profile = X_pca.mean(axis=0).reshape(1, -1)

    # ── Proses gambar baru ────────────────────────────────────
    with st.spinner("Menganalisis foto baru…"):
        new_img_bgr = bytes_to_cv2(new_file)

        if use_face_detection:
            new_vec, new_box = extract_face_vector(new_img_bgr, detector)
        else:
            gray_new = cv2.cvtColor(new_img_bgr, cv2.COLOR_BGR2GRAY)
            new_vec = cv2.resize(gray_new, IMG_SIZE).flatten() / 255.0
            new_box = None

        if new_vec is None:
            st.error("❌ Wajah tidak terdeteksi pada foto baru. "
                     "Coba foto lain atau matikan deteksi otomatis.")
            st.stop()

        new_pca = pca.transform(new_vec.reshape(1, -1))

        # Cosine similarity terhadap mean profile dataset
        similarity_raw = float(cosine_similarity(new_pca, mean_profile)[0][0])
        score = max(0.0, min(1.0, (similarity_raw + 1) / 2))

        # Temukan gambar dataset paling mirip dengan gambar baru
        sims_all = cosine_similarity(new_pca, X_pca)[0]
        best_idx = int(np.argmax(sims_all))
        best_name, best_img_bgr, best_box = dataset_imgs[best_idx]

    # ──────────────────────────────────────────────────────────
    # TAMPILKAN HASIL
    # ──────────────────────────────────────────────────────────

    is_mirip = score >= threshold
    pct = score * 100

    st.markdown("---")
    st.markdown("### 📊 Hasil Analisis")

    # ── Skor kemiripan ────────────────────────────────────────
    bar_color = "green" if is_mirip else "red"
    status_text = "✅ COCOK / MIRIP" if is_mirip else "❌ TIDAK COCOK"

    result_class = "result-mirip" if is_mirip else "result-tidak"
    bar_class = f"score-bar-fill-{bar_color}"

    st.markdown(f"""
    <div class="{result_class}">
        <div class="result-label">{status_text}</div>
        <div class="result-score">{pct:.1f}%</div>
        <div class="score-bar-wrap">
            <div class="{bar_class}" style="width:{min(pct,100):.1f}%"></div>
        </div>
        <div class="result-sub">Threshold: {threshold*100:.0f}% &nbsp;|&nbsp; Komponen PCA: {k}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Perbandingan gambar ───────────────────────────────────
    st.markdown("#### 🖼️ Perbandingan Gambar")

    col_l, col_mid, col_r = st.columns([5, 1, 5])

    with col_l:
        # Dataset image (paling mirip)
        best_display = draw_box(best_img_bgr, best_box) if best_box is not None else best_img_bgr
        st.image(cv2_to_pil(best_display), use_container_width=True)
        st.markdown(f'<div class="img-caption">📁 Dari Dataset<br><b>{best_name}</b></div>',
                    unsafe_allow_html=True)

    with col_mid:
        st.markdown('<div class="vs-divider" style="padding-top:4rem">VS</div>',
                    unsafe_allow_html=True)

    with col_r:
        # New image
        new_display = draw_box(new_img_bgr, new_box) if new_box is not None else new_img_bgr
        st.image(cv2_to_pil(new_display), use_container_width=True)
        st.markdown(f'<div class="img-caption">🆕 Foto Baru<br><b>{new_file.name}</b></div>',
                    unsafe_allow_html=True)

    # ── Info tambahan ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️ Detail Teknis"):
        st.markdown(f"""
        | Parameter | Nilai |
        |---|---|
        | Jumlah foto dataset | {len(dataset_files)} foto |
        | Wajah berhasil diproses | {len(features)} wajah |
        | Komponen PCA (k) | {k} |
        | Cosine Similarity (raw) | `{similarity_raw:.6f}` |
        | Skor ternormalisasi | `{score:.6f}` |
        | Threshold | `{threshold}` |
        | Foto dataset paling mirip | `{best_name}` |
        | Deteksi wajah otomatis | {'Aktif' if use_face_detection else 'Nonaktif'} |
        """)

    # ── Semua foto dataset ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📂 Semua Foto Dataset yang Diproses"):
        n_cols = 4
        rows = [dataset_imgs[i:i+n_cols] for i in range(0, len(dataset_imgs), n_cols)]
        for row in rows:
            cols = st.columns(n_cols)
            for col, (name, img_bgr, box) in zip(cols, row):
                disp = draw_box(img_bgr, box) if box is not None else img_bgr
                col.image(cv2_to_pil(disp), use_container_width=True)
                col.caption(name)

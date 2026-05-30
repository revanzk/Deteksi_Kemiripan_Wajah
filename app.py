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

/* ── Font global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ────────────────────────────────────────────────────────────
   APP BACKGROUND: #f8fafc  → putih muda  → teks GELAP
   ──────────────────────────────────────────────────────────── */
.stApp {
    background: #f8fafc;
    color: #1e293b;
}

/* Streamlit default text elements di atas bg putih → gelap */
.stApp p,
.stApp span,
.stApp label,
.stApp li,
.stApp td,
.stApp th,
.stApp caption,
[data-testid="stMarkdownContainer"] *,
[data-testid="stCaptionContainer"] * {
    color: #1e293b;
}

/* ────────────────────────────────────────────────────────────
   HEADER CARD: bg gradien BIRU TUA → teks PUTIH
   ──────────────────────────────────────────────────────────── */
.header-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}
.header-card h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
    color: #ffffff !important;   /* bg gelap → teks putih */
}
.header-card p {
    font-size: 0.95rem;
    margin: 0.4rem 0 0 0;
    color: rgba(255,255,255,0.88) !important;   /* bg gelap → teks putih */
}

/* ────────────────────────────────────────────────────────────
   SECTION CARD: bg WHITE → teks GELAP
   ──────────────────────────────────────────────────────────── */
.section-card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border: 1px solid #e2e8f0;
}
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #1e3a5f;   /* bg putih → teks navy gelap */
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ────────────────────────────────────────────────────────────
   RESULT MIRIP: bg gradien HIJAU MUDA → teks HIJAU TUA
   ──────────────────────────────────────────────────────────── */
.result-mirip {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border: 2px solid #34d399;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-mirip .result-label,
.result-mirip .result-score,
.result-mirip .result-sub {
    color: #064e3b;   /* bg hijau muda → teks hijau tua */
}

/* ────────────────────────────────────────────────────────────
   RESULT TIDAK: bg gradien MERAH MUDA → teks MERAH TUA
   ──────────────────────────────────────────────────────────── */
.result-tidak {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 2px solid #f87171;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-tidak .result-label,
.result-tidak .result-score,
.result-tidak .result-sub {
    color: #7f1d1d;   /* bg merah muda → teks merah tua */
}

/* Typography result (shared) */
.result-label {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-score {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
}
.result-sub {
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

/* ────────────────────────────────────────────────────────────
   SCORE BAR: bg abu muda, isi gradien → tidak perlu teks
   ──────────────────────────────────────────────────────────── */
.score-bar-wrap {
    background: #cbd5e1;
    border-radius: 99px;
    height: 10px;
    margin: 0.8rem 0;
    overflow: hidden;
}
.score-bar-fill-green {
    height: 10px;
    border-radius: 99px;
    background: linear-gradient(90deg, #34d399, #059669);
}
.score-bar-fill-red {
    height: 10px;
    border-radius: 99px;
    background: linear-gradient(90deg, #f87171, #dc2626);
}

/* ────────────────────────────────────────────────────────────
   VS DIVIDER: bg #e2e8f0 abu muda → teks GELAP
   ──────────────────────────────────────────────────────────── */
.vs-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    font-weight: 800;
    color: #334155;   /* bg abu muda → teks slate gelap */
    background: #e2e8f0;
    border-radius: 50%;
    width: 2.5rem;
    height: 2.5rem;
    margin: 5rem auto 0 auto;
}

/* ────────────────────────────────────────────────────────────
   IMAGE CAPTION: bg transparan (inherits putih) → teks GELAP
   ──────────────────────────────────────────────────────────── */
.img-caption {
    text-align: center;
    font-size: 0.85rem;
    color: #1e293b;   /* bg putih → teks gelap */
    margin-top: 0.5rem;
    font-weight: 600;
    line-height: 1.6;
}

/* ────────────────────────────────────────────────────────────
   ANALYSIS BOX: bg WHITE → teks GELAP
   ──────────────────────────────────────────────────────────── */
.analysis-box {
    background: white;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    padding: 1.2rem 1.6rem;
    margin-top: 1rem;
}
.analysis-box .analysis-title {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.5rem;
    color: #1e3a5f;   /* bg putih → teks navy gelap */
}
.analysis-box ul {
    margin: 0.4rem 0 0 1.2rem;
    padding: 0;
}
.analysis-box li {
    margin-bottom: 0.3rem;
    color: #1e293b;   /* bg putih → teks gelap */
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ────────────────────────────────────────────────────────────
   TOMBOL: bg gradien BIRU TUA → teks PUTIH
   ──────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #2563eb);
    color: #ffffff !important;   /* bg gelap → teks putih */
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

/* ────────────────────────────────────────────────────────────
   SIDEBAR: bg putih/abu muda (Streamlit default) → teks GELAP
   ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] li {
    color: #1e293b;   /* bg putih → teks gelap */
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #1e3a5f;   /* bg putih → headings navy gelap */
}

/* ── File uploader: bg putih → teks gelap ── */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: #1e293b;
}

/* ── Expander: bg putih → teks gelap ── */
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] p,
[data-testid="stExpander"] td,
[data-testid="stExpander"] th {
    color: #1e293b;
}

/* ── Slider: bg putih → teks gelap ── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] span {
    color: #1e293b;
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
    """Load Haar Cascade untuk deteksi wajah frontal."""
    frontal = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    frontal_alt2 = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    )
    profile = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )
    return frontal, frontal_alt2, profile

def bytes_to_cv2(uploaded_file):
    """Konversi UploadedFile → numpy array (BGR)."""
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    uploaded_file.seek(0)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def extract_face_vector(img_bgr, detectors):
    """
    Deteksi seluruh area wajah → crop → grayscale → resize → flatten.
    Mencoba beberapa cascade (frontal default, alt2, profile) agar
    seluruh bagian wajah (dahi hingga dagu) ikut terdeteksi.
    Return (vector, bounding_box) atau (None, None) jika tidak terdeteksi.
    """
    frontal, frontal_alt2, profile = detectors
    h_img, w_img = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Equalize histogram agar pencahayaan tidak mempengaruhi deteksi
    gray_eq = cv2.equalizeHist(gray)

    faces = []

    # Coba frontal default
    det = frontal.detectMultiScale(
        gray_eq, scaleFactor=1.05, minNeighbors=4,
        minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(det) > 0:
        faces = list(det)

    # Fallback: frontal alt2
    if not faces:
        det2 = frontal_alt2.detectMultiScale(
            gray_eq, scaleFactor=1.05, minNeighbors=4,
            minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(det2) > 0:
            faces = list(det2)

    # Fallback: profile
    if not faces:
        det3 = profile.detectMultiScale(
            gray_eq, scaleFactor=1.05, minNeighbors=4,
            minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(det3) > 0:
            faces = list(det3)

    if not faces:
        return None, None

    # Ambil wajah terbesar
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Tambah padding 15% ke atas (dahi) dan 5% ke sisi & bawah
    # agar seluruh wajah ikut tercakup dalam crop fitur
    pad_top    = int(h * 0.15)
    pad_side   = int(w * 0.05)
    pad_bottom = int(h * 0.05)

    x1 = max(0, x - pad_side)
    y1 = max(0, y - pad_top)
    x2 = min(w_img, x + w + pad_side)
    y2 = min(h_img, y + h + pad_bottom)

    face_crop = gray[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, IMG_SIZE)
    return face_resized.flatten() / 255.0, (x1, y1, x2 - x1, y2 - y1)

def cv2_to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def draw_box(img_bgr, box):
    """
    Gambar kotak di sekitar SELURUH wajah (sudah termasuk padding).
    Warna: biru (#2563eb), ketebalan proporsional dengan ukuran gambar.
    """
    out = img_bgr.copy()
    h_img, w_img = out.shape[:2]
    x, y, w, h = box
    thickness = max(2, int(min(w_img, h_img) * 0.005))
    # Kotak utama wajah
    cv2.rectangle(out, (x, y), (x + w, y + h), (37, 99, 235), thickness)
    # Label kecil "Wajah" di pojok kiri atas kotak
    label_y = max(y - 6, 14)
    cv2.putText(
        out, "Wajah", (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.4, min(w_img, h_img) * 0.0012),
        (37, 99, 235), thickness,
    )
    return out

def get_analysis_text(score, threshold, is_mirip):
    """
    Mengembalikan (judul, poin-poin analisis) berdasarkan rentang skor.
    """
    pct = score * 100
    gap = abs(pct - threshold * 100)

    if pct >= 90:
        title = "🟢 Kemiripan Sangat Tinggi"
        points = [
            f"Skor {pct:.1f}% jauh melampaui batas threshold ({threshold*100:.0f}%).",
            "Pola tekstur, kontur, dan distribusi piksel wajah pada ruang PCA sangat berdekatan.",
            "Vektor representasi kedua wajah memiliki arah yang hampir identik dalam ruang Eigenfaces.",
            "Kemungkinan besar ini adalah orang yang sama atau wajah dengan karakteristik yang sangat serupa.",
        ]
    elif pct >= 75:
        title = "🟢 Kemiripan Tinggi"
        points = [
            f"Skor {pct:.1f}% berada di atas threshold ({threshold*100:.0f}%) dengan selisih {gap:.1f}%.",
            "Fitur-fitur dominan wajah (jarak mata, bentuk hidung, kontur wajah) terproyeksi ke arah yang serupa.",
            "Eigenfaces menangkap kesamaan struktural yang signifikan antara kedua gambar.",
            "Sistem menilai kedua wajah memiliki karakteristik yang mirip secara matematis.",
        ]
    elif is_mirip:
        title = "🟡 Kemiripan Cukup (Batas Atas)"
        points = [
            f"Skor {pct:.1f}% melewati threshold ({threshold*100:.0f}%) namun dengan selisih tipis ({gap:.1f}%).",
            "Ada kesamaan pada beberapa komponen utama PCA, namun tidak semua fitur wajah selaras.",
            "Perbedaan pencahayaan, sudut, atau ekspresi bisa mempengaruhi hasil di rentang skor ini.",
            "Disarankan menambah foto dataset atau menyesuaikan threshold untuk hasil lebih akurat.",
        ]
    elif pct >= 40:
        title = "🟠 Kemiripan Rendah (Di Bawah Threshold)"
        points = [
            f"Skor {pct:.1f}% berada di bawah threshold ({threshold*100:.0f}%) dengan selisih {gap:.1f}%.",
            "Vektor wajah dalam ruang PCA menunjukkan arah yang cukup berbeda antar kedua gambar.",
            "Fitur-fitur seperti distribusi cahaya, proporsi wajah, atau tekstur kulit berbeda secara signifikan.",
            "Kemungkinan beda orang, atau foto diambil dalam kondisi yang sangat berbeda (sudut, pencahayaan).",
        ]
    else:
        title = "🔴 Kemiripan Sangat Rendah"
        points = [
            f"Skor {pct:.1f}% sangat jauh dari threshold ({threshold*100:.0f}%).",
            "Cosine similarity mendekati nol atau negatif — arah vektor kedua wajah di ruang PCA sangat berbeda.",
            "Pola piksel dan struktur wajah yang ditangkap Eigenfaces tidak memiliki kesamaan yang berarti.",
            "Sistem menyimpulkan dengan keyakinan tinggi bahwa kedua gambar bukan wajah yang sama.",
        ]

    return title, points

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

    detectors = get_detector()

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
                vec, box = extract_face_vector(img_bgr, detectors)
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
            new_vec, new_box = extract_face_vector(new_img_bgr, detectors)
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
        st.markdown(
            f'<div class="img-caption">📁 Dari Dataset<br><b>{best_name}</b></div>',
            unsafe_allow_html=True
        )

    with col_mid:
        st.markdown(
            '<div class="vs-divider">VS</div>',
            unsafe_allow_html=True
        )

    with col_r:
        # New image
        new_display = draw_box(new_img_bgr, new_box) if new_box is not None else new_img_bgr
        st.image(cv2_to_pil(new_display), use_container_width=True)
        st.markdown(
            f'<div class="img-caption">🆕 Foto Baru<br><b>{new_file.name}</b></div>',
            unsafe_allow_html=True
        )

    # ── Analisis Naratif ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    analysis_title, analysis_points = get_analysis_text(score, threshold, is_mirip)
    points_html = "".join(f"<li>{p}</li>" for p in analysis_points)
    st.markdown(f"""
    <div class="analysis-box">
        <div class="analysis-title">💡 Analisis: {analysis_title}</div>
        <ul>{points_html}</ul>
    </div>
    """, unsafe_allow_html=True)

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

import streamlit as st
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kemiripan Wajah — PCA & SVD",
    page_icon="🔬",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────
# Warna background → teks yang dipakai:
#
#  #0f172a  (navy sangat gelap)  → #f8fafc  (putih muda)
#  #1e293b  (navy gelap)         → #f1f5f9  (abu sangat muda)
#  #f1f5f9  (abu sangat muda)    → #0f172a  (navy sangat gelap)
#  #ffffff  (putih)              → #1e293b  (navy gelap)
#  #dcfce7  (hijau muda)         → #14532d  (hijau sangat gelap)
#  #fee2e2  (merah muda)         → #7f1d1d  (merah sangat gelap)
#  #2563eb  (biru)               → #ffffff  (putih)
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ══ BASE: App background = #f1f5f9 (abu muda) ══ */
html, body { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}

/* ══ HEADER BANNER: bg #0f172a (navy gelap) → teks #f8fafc (putih muda) ══ */
.hdr {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #2563eb 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.8rem;
    text-align: center;
}
.hdr-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;       /* bg gelap → teks terang */
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hdr-sub {
    font-size: 0.9rem;
    color: #94a3b8;       /* bg gelap → teks abu muda (lebih soft) */
    margin: 0;
}

/* ══ CARD: bg #ffffff → teks #1e293b ══ */
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1e3a5f;       /* bg putih → teks navy */
    margin: 0 0 0.3rem 0;
}
.card-sub {
    font-size: 0.82rem;
    color: #64748b;       /* bg putih → teks abu medium */
    margin: 0;
}

/* ══ SIDEBAR: bg #ffffff → SEMUA teks #1e293b (gelap) ══
   Pakai wildcard * agar tidak ada elemen yang terlewat.
   Exception: elemen dengan bg gelap di dalam sidebar (jika ada)
   harus di-override secara terpisah di bawah. ══ */
[data-testid="stSidebar"] {
    background: #ffffff;
}
[data-testid="stSidebar"] * {
    color: #1e293b !important;      /* bg putih → semua teks gelap */
    font-family: 'Inter', sans-serif;
}
/* Heading sidebar sedikit lebih gelap */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #0f172a !important;
    font-weight: 700;
}
/* Sub-teks / caption sidebar */
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: #475569 !important;      /* abu medium — masih terbaca di bg putih */
}

/* ══ FILE UPLOADER: bg #f8fafc → teks #1e293b ══ */
[data-testid="stFileUploaderDropzone"] {
    background: #f8fafc;
    border: 1.5px dashed #cbd5e1;
    border-radius: 10px;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span {
    color: #475569;       /* bg abu sangat muda → teks abu medium */
}

/* ══ BUTTON: bg #2563eb (biru) → teks #ffffff ══ */
.stButton > button {
    background: #2563eb;
    color: #ffffff !important;   /* bg biru → WAJIB putih */
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    font-family: 'Inter', sans-serif;
    width: 100%;
    transition: background 0.2s, transform 0.1s;
}
.stButton > button:hover {
    background: #1d4ed8;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ══ RESULT — COCOK: bg #dcfce7 (hijau muda) → teks #14532d (hijau gelap) ══ */
.res-ok {
    background: #dcfce7;
    border: 2px solid #86efac;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    text-align: center;
}
.res-ok .res-status  { color: #14532d; font-size: 1.1rem; font-weight: 700; margin: 0 0 0.2rem; }
.res-ok .res-pct     { color: #166534; font-size: 3.2rem; font-weight: 800; line-height: 1; margin: 0; }
.res-ok .res-bar-bg  { background: #bbf7d0; border-radius: 99px; height: 10px; margin: 0.9rem 0 0.5rem; overflow: hidden; }
.res-ok .res-bar-fg  { background: linear-gradient(90deg, #22c55e, #15803d); height: 10px; border-radius: 99px; }
.res-ok .res-note    { color: #166534; font-size: 0.82rem; margin: 0; }

/* ══ RESULT — TIDAK: bg #fee2e2 (merah muda) → teks #7f1d1d (merah gelap) ══ */
.res-no {
    background: #fee2e2;
    border: 2px solid #fca5a5;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    text-align: center;
}
.res-no .res-status  { color: #7f1d1d; font-size: 1.1rem; font-weight: 700; margin: 0 0 0.2rem; }
.res-no .res-pct     { color: #991b1b; font-size: 3.2rem; font-weight: 800; line-height: 1; margin: 0; }
.res-no .res-bar-bg  { background: #fecaca; border-radius: 99px; height: 10px; margin: 0.9rem 0 0.5rem; overflow: hidden; }
.res-no .res-bar-fg  { background: linear-gradient(90deg, #f87171, #dc2626); height: 10px; border-radius: 99px; }
.res-no .res-note    { color: #991b1b; font-size: 0.82rem; margin: 0; }

/* ══ VS BADGE: bg #e2e8f0 (abu muda) → teks #334155 (gelap) ══ */
.vs-badge {
    width: 3rem;
    height: 3rem;
    background: #e2e8f0;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 800;
    color: #334155;       /* bg abu muda → teks slate gelap */
    margin: 6rem auto 0;
}

/* ══ IMG CAPTION: bg transparan/putih → teks #1e293b ══ */
.img-cap {
    text-align: center;
    font-size: 0.82rem;
    font-weight: 600;
    color: #1e293b;       /* bg putih → teks gelap */
    margin-top: 0.5rem;
    line-height: 1.5;
    font-family: 'Inter', sans-serif;
}
.img-cap small {
    display: block;
    color: #64748b;       /* bg putih → sub-teks abu medium */
    font-weight: 400;
    font-size: 0.75rem;
}

/* ══ ANALYSIS BOX: bg #1e293b (navy gelap) → teks #f1f5f9 (putih muda) ══ */
.ana-box {
    background: #1e293b;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-top: 1.2rem;
}
.ana-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #7dd3fc;       /* bg navy gelap → teks biru muda (aksen) */
    margin: 0 0 0.7rem 0;
}
.ana-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.ana-list li {
    color: #e2e8f0;       /* bg navy gelap → teks abu sangat muda */
    font-size: 0.88rem;
    line-height: 1.6;
    padding: 0.25rem 0;
    border-bottom: 1px solid #334155;
    display: flex;
    gap: 0.5rem;
}
.ana-list li:last-child { border-bottom: none; }
.ana-dot {
    color: #38bdf8;       /* aksen biru */
    flex-shrink: 0;
}

/* ══ EXPANDER: bg #ffffff → teks #1e293b ══ */
[data-testid="stExpander"] {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}
[data-testid="stExpander"] summary span {
    color: #1e293b;       /* bg putih → teks gelap */
    font-weight: 600;
    font-family: 'Inter', sans-serif;
}
[data-testid="stExpander"] p,
[data-testid="stExpander"] td,
[data-testid="stExpander"] th {
    color: #1e293b;       /* bg putih → teks gelap */
    font-family: 'Inter', sans-serif;
}

/* ══ SLIDER & TOGGLE: label di bg #f1f5f9 → teks #1e293b ══ */
[data-testid="stSlider"] label p,
[data-testid="stSlider"] span {
    color: #1e293b;
}
div[data-testid="stToggle"] label {
    color: #1e293b;
}

/* ══ STEP BADGE ══ */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #dbeafe;   /* bg biru sangat muda */
    color: #1e40af;         /* teks biru gelap — kontras di bg biru muda */
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.25rem 0.7rem;
    border-radius: 99px;
    margin-bottom: 0.5rem;
}

/* ══ Streamlit label di atas uploader ══ */
[data-testid="stFileUploader"] label p {
    color: #374151;
    font-weight: 500;
}

/* ══ Caption gambar Streamlit native ══ */
[data-testid="caption"] {
    color: #475569;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
IMG_SIZE = (100, 100)

@st.cache_resource
def get_detectors():
    """Load 3 Haar Cascade: frontal default, alt2, dan profile."""
    return (
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"),
    )

def file_to_bgr(f):
    data = np.frombuffer(f.read(), np.uint8)
    f.seek(0)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def detect_face(img_bgr, detectors):
    """
    Deteksi wajah dengan 3 cascade (frontal → alt2 → profile).
    Mengembalikan (vektor_fitur, bounding_box) atau (None, None).
    Bounding box sudah termasuk padding 15% atas dan 5% sisi/bawah.
    """
    h_img, w_img = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)   # equalize agar tahan perubahan cahaya

    faces = []
    for cas in detectors:
        det = cas.detectMultiScale(
            gray_eq, scaleFactor=1.05, minNeighbors=4,
            minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(det) > 0:
            faces = list(det)
            break   # pakai hasil cascade pertama yang berhasil

    if not faces:
        return None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Padding proporsional agar seluruh wajah (dahi–dagu) tercakup
    pt = int(h * 0.15)   # atas (dahi)
    ps = int(w * 0.05)   # samping
    pb = int(h * 0.05)   # bawah (dagu)

    x1, y1 = max(0, x - ps), max(0, y - pt)
    x2, y2 = min(w_img, x + w + ps), min(h_img, y + h + pb)

    face_vec = cv2.resize(gray[y1:y2, x1:x2], IMG_SIZE).flatten() / 255.0
    return face_vec, (x1, y1, x2 - x1, y2 - y1)

def fallback_vec(img_bgr):
    """Tanpa deteksi wajah: seluruh gambar dijadikan vektor."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, IMG_SIZE).flatten() / 255.0

def draw_face_box(img_bgr, box):
    """Gambar kotak biru + label 'Wajah' di sekitar area wajah."""
    out = img_bgr.copy()
    x, y, w, h = box
    h_img, w_img = out.shape[:2]
    thick = max(2, int(min(w_img, h_img) * 0.005))
    cv2.rectangle(out, (x, y), (x + w, y + h), (37, 99, 235), thick)
    cv2.putText(
        out, "Wajah", (x, max(y - 6, 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.4, min(w_img, h_img) * 0.0012),
        (37, 99, 235), thick,
    )
    return out

def analysis_text(score, threshold, is_ok):
    """Kembalikan (judul, list poin) berdasarkan rentang skor."""
    pct = score * 100
    gap = abs(pct - threshold * 100)
    if pct >= 90:
        return "Kemiripan Sangat Tinggi 🟢", [
            f"Skor {pct:.1f}% jauh melampaui threshold {threshold*100:.0f}%.",
            "Vektor wajah di ruang PCA hampir identik arahnya.",
            "Pola piksel, kontur, dan tekstur sangat berdekatan.",
            "Kemungkinan besar ini orang yang sama.",
        ]
    elif pct >= 75:
        return "Kemiripan Tinggi 🟢", [
            f"Skor {pct:.1f}% berada {gap:.1f}% di atas threshold.",
            "Fitur dominan (kontur wajah, jarak fitur) selaras di ruang Eigenfaces.",
            "Sistem menilai wajah memiliki kemiripan struktural yang nyata.",
        ]
    elif is_ok:
        return "Kemiripan Cukup 🟡", [
            f"Skor {pct:.1f}% melewati threshold tipis ({gap:.1f}%).",
            "Ada kesamaan pada sebagian komponen PCA, namun tidak semua.",
            "Pencahayaan atau sudut foto bisa mempengaruhi hasil.",
            "Tambah foto dataset atau sesuaikan threshold untuk hasil lebih pasti.",
        ]
    elif pct >= 40:
        return "Kemiripan Rendah 🟠", [
            f"Skor {pct:.1f}% berada {gap:.1f}% di bawah threshold.",
            "Arah vektor wajah di ruang PCA cukup berbeda.",
            "Distribusi cahaya, proporsi, atau tekstur wajah berbeda signifikan.",
            "Kemungkinan beda orang, atau foto diambil dalam kondisi sangat berbeda.",
        ]
    else:
        return "Kemiripan Sangat Rendah 🔴", [
            f"Skor {pct:.1f}% sangat jauh dari threshold.",
            "Cosine similarity mendekati nol — arah vektor hampir tegak lurus.",
            "Tidak ada kesamaan struktural berarti yang ditangkap Eigenfaces.",
            "Sistem sangat yakin kedua gambar bukan wajah yang sama.",
        ]

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    n_comp   = st.slider("Komponen PCA (k)", 1, 50, 8)
    thr      = st.slider("Threshold kemiripan", 0.0, 1.0, 0.55, 0.01,
                         format="%.2f")
    use_det  = st.toggle("Deteksi wajah otomatis", value=True)
    st.markdown("---")
    st.markdown("""
**Cara pakai:**
1. Upload foto‑foto dataset
2. Upload 1 foto baru
3. Klik **Analisis**
    """)
    st.markdown("---")
    st.caption("PCA · SVD · Eigenfaces · Cosine Similarity")

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
    <p class="hdr-title">🔬 Deteksi Kemiripan Wajah</p>
    <p class="hdr-sub">PCA &nbsp;·&nbsp; SVD &nbsp;·&nbsp; Eigenfaces &nbsp;·&nbsp; Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# UPLOAD SECTION (2 kolom sejajar)
# ─────────────────────────────────────────────────────────────
col_up1, col_up2 = st.columns(2, gap="large")

with col_up1:
    st.markdown("""
    <div class="card">
        <div class="step-badge">Langkah 1</div>
        <p class="card-title">📁 Foto Dataset (Referensi)</p>
        <p class="card-sub">Upload beberapa foto wajah sebagai data latih. Minimal 3 foto.</p>
    </div>
    """, unsafe_allow_html=True)
    dataset_files = st.file_uploader(
        "Pilih foto dataset",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="ds",
        label_visibility="collapsed",
    )

with col_up2:
    st.markdown("""
    <div class="card">
        <div class="step-badge">Langkah 2</div>
        <p class="card-title">🆕 Foto Baru untuk Dicocokkan</p>
        <p class="card-sub">Upload 1 foto wajah baru yang ingin dibandingkan dengan dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    new_file = st.file_uploader(
        "Pilih foto baru",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="nf",
        label_visibility="collapsed",
    )

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("🔍 Analisis Kemiripan", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ANALISIS
# ─────────────────────────────────────────────────────────────
if run:
    # Validasi input
    if not dataset_files:
        st.error("❌ Upload minimal 1 foto dataset terlebih dahulu.")
        st.stop()
    if new_file is None:
        st.error("❌ Upload foto baru untuk dicocokkan.")
        st.stop()

    detectors = get_detectors()

    # ── Proses dataset ──
    with st.spinner("Memproses dataset & melatih PCA…"):
        features, ds_imgs, failed = [], [], []

        for f in dataset_files:
            bgr = file_to_bgr(f)
            if bgr is None:
                failed.append(f.name); continue

            if use_det:
                vec, box = detect_face(bgr, detectors)
            else:
                vec, box = fallback_vec(bgr), None

            ds_imgs.append((f.name, bgr, box))
            if vec is not None:
                features.append(vec)
            else:
                failed.append(f.name)

        if failed:
            st.warning(f"⚠️ Wajah tidak terdeteksi pada: {', '.join(failed)}")

        if len(features) < 2:
            st.error("❌ Terlalu sedikit wajah yang terdeteksi. "
                     "Coba foto lebih jelas atau matikan deteksi otomatis.")
            st.stop()

        X    = np.array(features)
        k    = max(1, min(n_comp, len(features) - 1))
        pca  = PCA(n_components=k, whiten=True)
        Xpca = pca.fit_transform(X)
        mean_v = Xpca.mean(axis=0).reshape(1, -1)

    # ── Proses foto baru ──
    with st.spinner("Menganalisis foto baru…"):
        new_bgr = file_to_bgr(new_file)

        if use_det:
            new_vec, new_box = detect_face(new_bgr, detectors)
        else:
            new_vec, new_box = fallback_vec(new_bgr), None

        if new_vec is None:
            st.error("❌ Wajah tidak terdeteksi pada foto baru. "
                     "Coba foto lain atau matikan deteksi otomatis.")
            st.stop()

        new_pca = pca.transform(new_vec.reshape(1, -1))

        # Cosine similarity vs mean dataset
        sim_raw = float(cosine_similarity(new_pca, mean_v)[0][0])
        score   = max(0.0, min(1.0, (sim_raw + 1) / 2))

        # Gambar dataset paling mirip
        sims_all = cosine_similarity(new_pca, Xpca)[0]
        bi = int(np.argmax(sims_all))
        best_name, best_bgr, best_box = ds_imgs[bi]

    is_ok  = score >= thr
    pct    = score * 100
    bar_w  = f"{min(pct, 100):.1f}%"

    st.markdown("---")

    # ── Tampilkan hasil ──
    left, right = st.columns([2, 3], gap="large")

    # Kolom kiri: kartu hasil
    with left:
        if is_ok:
            st.markdown(f"""
            <div class="res-ok">
                <p class="res-status">✅ COCOK / MIRIP</p>
                <p class="res-pct">{pct:.1f}%</p>
                <div class="res-bar-bg">
                    <div class="res-bar-fg" style="width:{bar_w}"></div>
                </div>
                <p class="res-note">Threshold: {thr*100:.0f}% &nbsp;|&nbsp; Komponen PCA: {k}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="res-no">
                <p class="res-status">❌ TIDAK COCOK</p>
                <p class="res-pct">{pct:.1f}%</p>
                <div class="res-bar-bg">
                    <div class="res-bar-fg" style="width:{bar_w}"></div>
                </div>
                <p class="res-note">Threshold: {thr*100:.0f}% &nbsp;|&nbsp; Komponen PCA: {k}</p>
            </div>
            """, unsafe_allow_html=True)

        # Analisis naratif
        ana_title, ana_pts = analysis_text(score, thr, is_ok)
        pts_html = "".join(
            f'<li><span class="ana-dot">▸</span>{p}</li>'
            for p in ana_pts
        )
        st.markdown(f"""
        <div class="ana-box">
            <p class="ana-title">💡 {ana_title}</p>
            <ul class="ana-list">{pts_html}</ul>
        </div>
        """, unsafe_allow_html=True)

    # Kolom kanan: perbandingan gambar
    with right:
        st.markdown("#### 🖼️ Perbandingan Gambar")
        c1, c2, c3 = st.columns([5, 1, 5])

        with c1:
            disp1 = draw_face_box(best_bgr, best_box) if best_box else best_bgr
            st.image(bgr_to_pil(disp1), use_container_width=True)
            st.markdown(
                f'<div class="img-cap">📁 Dari Dataset<small>{best_name}</small></div>',
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                '<div class="vs-badge">VS</div>',
                unsafe_allow_html=True
            )

        with c3:
            disp2 = draw_face_box(new_bgr, new_box) if new_box else new_bgr
            st.image(bgr_to_pil(disp2), use_container_width=True)
            st.markdown(
                f'<div class="img-cap">🆕 Foto Baru<small>{new_file.name}</small></div>',
                unsafe_allow_html=True
            )

    # ── Detail teknis ──
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️ Detail Teknis"):
        st.markdown(f"""
| Parameter | Nilai |
|---|---|
| Foto dataset diunggah | {len(dataset_files)} |
| Wajah berhasil diproses | {len(features)} |
| Komponen PCA (k) | {k} |
| Cosine similarity (raw) | `{sim_raw:.6f}` |
| Skor ternormalisasi | `{score:.6f}` |
| Threshold | `{thr}` |
| Foto dataset paling mirip | `{best_name}` |
| Mode deteksi | {'Haar Cascade (otomatis)' if use_det else 'Tanpa deteksi'} |
        """)

    # ── Grid semua foto dataset ──
    with st.expander("📂 Semua Foto Dataset"):
        cols = st.columns(5)
        for i, (nm, bgr, box) in enumerate(ds_imgs):
            d = draw_face_box(bgr, box) if box else bgr
            cols[i % 5].image(bgr_to_pil(d), use_container_width=True)
            cols[i % 5].caption(nm)

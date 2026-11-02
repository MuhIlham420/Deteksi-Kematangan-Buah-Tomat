"""Website Streamlit untuk UTS Praktikum Visi Komputer, Kelompok 5."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import (
    CFG_DEFAULT,
    CFG_TUNED,
    GESER,
    KELAS,
    LABEL,
    WARNA,
    buat_cfg,
    buat_demo,
    gambar_hasil,
    jalankan_pipeline,
    muat_upload,
    tabel_hasil,
)

ROOT = Path(__file__).resolve().parent
KOLOM_HASIL = ["nomor", "kelas", "hue_median", "rasio_rg", "luas_px", "x", "y", "lebar", "tinggi"]
TIM = [
    {"Nama": "Rijaluddin Abdul Ghani", "NPM": "2408107010008"},
    {"Nama": "M. Ilham", "NPM": "2408107010043"},
    {"Nama": "Fathurrahman Pahmiputra", "NPM": "2408107010069"},
    {"Nama": "Muhammad Shidqi Hanif", "NPM": "2408107010096"},
]


def png_bytes(bgr):
    ok, data = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("Gagal menyiapkan gambar untuk diunduh.")
    return data.tobytes()


def pengaturan():
    with st.sidebar:
        st.markdown("## 🍅 Tomat / Vision")
        st.caption("KELOMPOK 05 · VISI KOMPUTER B")
        st.divider()
        st.subheader("Pengaturan analisis")
        preset = st.selectbox("Konfigurasi", ["Tuned — sesuai laporan", "Default notebook"])
        base = CFG_TUNED if preset.startswith("Tuned") else CFG_DEFAULT
        ruang = st.selectbox("Metode klasifikasi", ["HSV", "BGR"])
        st.caption("Segmentasi tetap HSV pada kedua metode, sesuai notebook.")
        with st.form("parameter"):
            eq_v = st.checkbox("Equalization channel V", value=False)
            pisah = st.checkbox("Pisahkan buah berdempetan", value=True)
            with st.expander("Parameter lanjutan"):
                s_min = st.slider("Saturasi minimum", 0, 255, base["s_min"])
                v_rel = st.slider("Batas V relatif terhadap median", 0.50, 1.00, base["v_rel"], 0.01)
                min_area = st.slider("Luas minimum (% citra)", 0.1, 2.0, base["min_area"] * 100, 0.1)
                h_matang = st.slider("Batas atas Hue matang", 0, 16, base["h_matang"])
                h_setengah = st.slider("Batas atas Hue setengah matang", 1, 40, base["h_setengah"])
                bgr_r = st.slider("Batas rasio R/(R+G) untuk BGR", 0.0, 1.0, base["bgr_r"], 0.01)
            st.form_submit_button("Terapkan parameter", width="stretch", type="primary")
        if h_setengah <= h_matang or bgr_r[0] >= bgr_r[1]:
            st.error("Batas kedua harus lebih besar daripada batas pertama.")
            st.stop()
        st.divider()
        st.markdown("**Tips foto**")
        st.caption("Gunakan latar putih/abu terang, cahaya merata, dan beri jarak antarbuah. Ambil foto dari atas.")
        st.caption("Metode berbasis warna, bukan model AI pengenal objek.")
    return buat_cfg(base, ruang=ruang.lower(), eq_v=eq_v, pisah_dempet=pisah,
                    s_min=s_min, v_rel=v_rel, min_area=min_area / 100,
                    h_matang=h_matang, h_setengah=h_setengah, bgr_r=bgr_r)


def histogram(p):
    fig, ax = plt.subplots(figsize=(10, 3.3))
    x = np.arange(180) - GESER
    for kelas in KELAS:
        hist = np.zeros(180, dtype=np.int64)
        for buah in p["hasil"]:
            if buah["kelas"] == kelas:
                hist += buah["h_hist"]
        warna = np.array(WARNA[kelas][::-1]) / 255
        ax.plot(x, hist, color=warna, label=LABEL[kelas], linewidth=1.5)
        ax.fill_between(x, hist, color=warna, alpha=0.14)
    for batas in (p["cfg"]["h_matang"], p["cfg"]["h_setengah"]):
        ax.axvline(batas, color="#77837b", linestyle="--", linewidth=1)
    ax.set(xlabel="Hue OpenCV (170–179 ditampilkan sebagai −10 sampai −1)",
           ylabel="Jumlah piksel", xlim=(-10, 169))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Histogram piksel dikelompokkan menurut prediksi kelas buah pada gambar ini, bukan ground truth. "
               "Garis putus-putus adalah batas klasifikasi HSV; tidak berlaku untuk keputusan BGR.")


def tampilkan_hasil(p, nama):
    st.subheader("Ringkasan deteksi")
    kolom = st.columns(4)
    kolom[0].metric("Total tomat", p["total"])
    for tempat, kelas in zip(kolom[1:], KELAS):
        tempat.metric(LABEL[kelas], p["hitung"][kelas])
    if p["total"] == 0:
        st.warning("Tidak ada kandidat tomat yang terdeteksi. Coba latar lebih terang/polos, "
                   "naikkan batas V relatif, atau turunkan luas minimum pada parameter lanjutan.")
    anotasi = gambar_hasil(p)
    kiri, kanan = st.columns(2)
    with kiri:
        st.markdown("#### 01 / Gambar asli")
        st.image(p["asli"], channels="BGR", width="stretch")
    with kanan:
        st.markdown("#### 02 / Hasil deteksi")
        st.image(anotasi, channels="BGR", width="stretch")
    tinggi, lebar = p["asli"].shape[:2]
    st.caption(f"Resolusi analisis {lebar} × {tinggi} px · Kotak hijau: mentah · "
               "Oranye: setengah matang · Merah: matang. Gambar asli hanya diperkecil, tanpa blur/equalization.")
    df = pd.DataFrame(tabel_hasil(p), columns=KOLOM_HASIL)
    with st.expander("Detail setiap buah dan unduhan", expanded=True):
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption("Koordinat dan luas mengacu pada resolusi analisis. Hue negatif berarti merah yang melingkar (170–179).")
        a, b, c = st.columns(3)
        a.download_button("Unduh gambar hasil", png_bytes(anotasi), f"{nama}_hasil.png", "image/png",
                          width="stretch")
        b.download_button("Unduh detail CSV", df.to_csv(index=False).encode("utf-8-sig"),
                          f"{nama}_detail.csv", "text/csv", width="stretch")
        ringkasan = pd.DataFrame([{"total": p["total"], **p["hitung"], **p["cfg"]}])
        c.download_button("Unduh ringkasan CSV", ringkasan.to_csv(index=False).encode("utf-8-sig"),
                          f"{nama}_ringkasan.csv", "text/csv", width="stretch")
    with st.expander("Lihat proses segmentasi dan histogram Hue"):
        kiri, kanan = st.columns(2)
        kiri.image(p["bgr"], channels="BGR", caption="Gaussian blur + equalization V (jika aktif)",
                   width="stretch")
        kanan.image(p["mask"], caption="Mask buah setelah morfologi (putih = kandidat objek)",
                    width="stretch")
        histogram(p)
        st.json(p["cfg"], expanded=False)


def analisis(cfg):
    st.markdown("### Dari warna, kenali kematangannya.")
    st.write("Unggah foto tomat untuk mendeteksi buah, memberi label kematangan, dan menghitung setiap kelas.")
    sumber = st.radio("Sumber gambar", ["Upload gambar", "Demo sintetis"], horizontal=True)
    img, nama = None, "tomat"
    if sumber == "Upload gambar":
        unggah = st.file_uploader("Pilih foto tomat", type=["jpg", "jpeg", "png", "webp"],
                                  help="Maksimal 10 MB dan 24 megapiksel.")
        st.caption("Gambar diproses di memori server dan tidak ditulis ke disk oleh aplikasi. "
                   "JPG, PNG, WEBP · Maks. 10 MB / 24 MP")
        if unggah is not None:
            try:
                img = muat_upload(unggah.getvalue())
                nama = Path(unggah.name).stem
            except ValueError as exc:
                st.error(str(exc))
    else:
        st.info("Demo ini berisi tiga lingkaran berwarna sintetis, bukan foto tomat asli "
                "dan bukan bagian dataset evaluasi. Gunakan untuk mencoba alur aplikasi.")
        img, nama = buat_demo(), "demo_sintetis"
    if img is None:
        st.info("Mulai dengan mengunggah foto, atau pilih Demo sintetis untuk mencoba tanpa dataset.")
        return
    with st.spinner("Menganalisis warna dan kontur buah…"):
        p = jalankan_pipeline(img, cfg)
    tampilkan_hasil(p, nama)
    if st.checkbox("Bandingkan empat skenario pada gambar ini"):
        st.caption("Perbandingan langsung memakai parameter sidebar yang sama; tidak melakukan tuning ulang. "
                   "Ambang BGR awal 0,50/0,75 bukan ambang tuned eksperimen laporan. "
                   "Tanpa ground truth, hasil ini adalah jumlah prediksi, bukan akurasi.")
        baris = []
        with st.spinner("Menjalankan empat skenario…"):
            for ruang, eq in [("hsv", False), ("bgr", False), ("hsv", True), ("bgr", True)]:
                skenario = ruang.upper() + (" + eq V" if eq else "")
                hasil = jalankan_pipeline(img, buat_cfg(cfg, ruang=ruang, eq_v=eq))
                baris.append({"Skenario": skenario, "Total": hasil["total"],
                              **{LABEL[k]: hasil["hitung"][k] for k in KELAS}})
        st.dataframe(baris, hide_index=True, width="stretch")
    st.warning("Hasil adalah estimasi berdasarkan warna. Latar berwarna, bayangan, atau buah menumpuk "
               "dapat menyebabkan salah hitung/label; objek berwarna lain juga bisa terdeteksi.")


def eksperimen():
    st.subheader("Hasil evaluasi kelompok")
    st.info("Angka di bawah disalin dari laporan dan keluaran notebook, bukan dihitung ulang oleh website. "
            "Foto dataset asli belum disertakan; evaluasi ulang memerlukan 20 foto tersebut.")
    st.write("Dataset: **20 foto / 100 kemunculan buah**, terdiri atas 14 mentah, 42 setengah matang, dan 44 matang. "
             "Kalibrasi hanya pada 8 foto ideal; 12 foto cahaya/sulit digunakan sebagai data uji.")
    st.dataframe(pd.DataFrame([
        ["Ideal", 8, "100,0%", "62,5%", 0.00, 0.25],
        ["Cahaya", 8, "100,0%", "37,5%", 0.00, 0.67],
        ["Sulit", 4, "75,0%", "0,0%", 1.00, 1.17],
        ["Semua", 20, "95,0%", "40,0%", 0.20, 0.60],
    ], columns=["Kondisi", "Foto", "Akurasi total", "Akurasi semua kelas", "MAE total", "MAE kelas"]),
        hide_index=True, width="stretch")
    st.markdown("#### Eksperimen pembanding — ambang tuned")
    st.dataframe(pd.DataFrame([
        ["HSV", "95,0%", "25,0%", "40,0%", 0.60],
        ["BGR", "95,0%", "0,0%", "10,0%", 1.37],
        ["HSV + eq V", "95,0%", "33,3%", "45,0%", 0.50],
        ["BGR + eq V", "95,0%", "25,0%", "35,0%", 0.87],
    ], columns=["Skenario", "Akurasi total", "Semua kelas (uji)", "Semua kelas (semua)", "MAE kelas"]),
        hide_index=True, width="stretch")
    st.markdown("""
**Cara membaca metrik**
- **Akurasi total:** persentase foto dengan jumlah total buah tepat.
- **Akurasi semua kelas:** persentase foto dengan jumlah ketiga kelas sekaligus tepat; bukan akurasi per buah.
- **MAE total / kelas:** rata-rata selisih absolut jumlah total / per kelas.

HSV lebih baik untuk klasifikasi pada dataset ini. Equalization V menaikkan hasil HSV
dari 40% menjadi 45%, tetapi selisihnya hanya satu foto dan belum membuktikan signifikansi statistik.
Segmentasi pada skenario BGR **tetap memakai HSV**; yang dibandingkan adalah fitur klasifikasi.
""")


def tentang():
    st.subheader("Tentang proyek")
    st.write("UTS Praktikum Visi Komputer B · Kelompok 5 · Informatika, FMIPA, Universitas Syiah Kuala.")
    st.dataframe(TIM, hide_index=True, width="stretch")
    st.markdown("#### Pipeline")
    st.code("Foto → resize (maks. 1000 px) → Gaussian blur 5×5 → HSV\n"
            "→ mask saturasi & V relatif → morfologi open/close\n"
            "→ kontur → pisah dempet → filter bentuk\n"
            "→ median Hue / rasio R:(R+G) → label & hitung", language=None)
    st.write("Konfigurasi awal aplikasi menggunakan hasil tuning notebook: S ≥ 40, "
             "batas V = 0,80 × median V, dan luas minimum 0,4% citra. "
             "Batas Hue OpenCV: matang 0–7 atau 170–179; setengah matang 8–11; sisanya mentah. "
             "Keputusan diterapkan pada median Hue yang digeser +10 untuk menangani merah melingkar.")
    st.markdown("#### Keterbatasan")
    st.write("Dataset kecil dan buah yang sama digunakan berulang. Ground truth hanya berupa jumlah kelas, "
             "bukan label tiap buah. Batas Hue sempit dan sensitif terhadap pencahayaan; "
             "erosi belum dapat memisahkan seluruh buah yang menumpuk rapat.")
    laporan = ROOT / "Laporan_Kelompok05.pdf"
    if laporan.is_file():
        st.download_button("Unduh laporan kelompok", laporan.read_bytes(), laporan.name, "application/pdf")
    st.caption("Sumber metode: notebook kelompok dan dokumentasi OpenCV. "
               "Adaptasi website/README dibantu asisten AI; detail atribusi tersedia di README.")


def main():
    st.set_page_config(page_title="Deteksi Kematangan Tomat | Kelompok 5", page_icon="🍅", layout="wide")
    st.markdown("""<style>
        .block-container {max-width: 1200px; padding-top: 2.5rem;}
        h1, h2, h3 {letter-spacing: -0.03em;}
        [data-testid="stMetric"] {background: #f0eee8; border: 1px solid #e1e2d8;
            border-radius: 14px; padding: 16px;}
        [data-testid="stMetricValue"] {color: #284f3b;}
    </style>""", unsafe_allow_html=True)
    cfg = pengaturan()
    st.caption("COMPUTER VISION / PENGOLAHAN CITRA DIGITAL")
    st.title("🍅 Deteksi Kematangan Tomat")
    st.caption("Dari foto ke informasi · Mentah, setengah matang, dan matang")
    tab_analisis, tab_eksperimen, tab_tentang = st.tabs(["Analisis gambar", "Hasil & eksperimen", "Tentang & pipeline"])
    with tab_analisis:
        analisis(cfg)
    with tab_eksperimen:
        eksperimen()
    with tab_tentang:
        tentang()
    st.divider()
    st.caption("Kelompok 05 · UTS Praktikum Visi Komputer B · Universitas Syiah Kuala")


if __name__ == "__main__":
    main()

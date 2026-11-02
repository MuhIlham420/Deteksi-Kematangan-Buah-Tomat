"""Pipeline OpenCV yang diadaptasi dari deteksi-kematangan-tomat.ipynb.

Tidak memerlukan model terlatih, Google Drive, atau dataset untuk inferensi.
Semua array citra berwarna di modul ini menggunakan urutan BGR.
"""

import warnings
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

KELAS = ["mentah", "setengah_matang", "matang"]
LABEL = {"mentah": "Mentah", "setengah_matang": "Setengah matang", "matang": "Matang"}
WARNA = {"mentah": (0, 180, 0), "setengah_matang": (0, 165, 255), "matang": (0, 0, 255)}
GESER = 10
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000

CFG_DEFAULT = {
    "ruang": "hsv", "sisi_max": 1000, "blur_k": 5, "eq_v": False,
    "s_min": 50, "v_rel": 0.85, "morph_k": 5, "min_area": 0.004,
    "pisah_dempet": True, "seed_min": 0.05, "aspek_max": 2.5, "ekstent_min": 0.5,
    "h_matang": 7, "h_setengah": 12, "bgr_r": (0.50, 0.75),
}
# Hasil tuning pada delapan foto ideal, tersimpan dalam notebook/laporan.
CFG_TUNED = {**CFG_DEFAULT, "s_min": 40, "v_rel": 0.80, "h_setengah": 11}


def buat_cfg(base=None, **ubah):
    cfg = dict(CFG_TUNED if base is None else base)
    if set(ubah) - set(CFG_DEFAULT):
        raise ValueError("Parameter konfigurasi tidak dikenal.")
    cfg.update(ubah)
    return cfg


def muat_upload(data):
    """Validasi file, koreksi orientasi EXIF, dan komposit transparansi ke putih."""
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Gambar harus berisi data dan berukuran maksimal 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as gambar:
                if gambar.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Format yang didukung: JPG, PNG, dan WEBP.")
                if gambar.width * gambar.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Resolusi maksimal 24 megapiksel. Perkecil gambar terlebih dahulu.")
                gambar = ImageOps.exif_transpose(gambar).convert("RGBA")
                latar = Image.new("RGBA", gambar.size, "white")
                rgb = np.array(Image.alpha_composite(latar, gambar).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("File tidak dapat dibaca sebagai gambar. Coba file lain.") from exc
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Resolusi gambar terlalu besar. Perkecil gambar terlebih dahulu.") from exc


def resize_citra(img, sisi_max=1000):
    skala = sisi_max / max(img.shape[:2])
    if skala < 1:
        return cv2.resize(img, None, fx=skala, fy=skala, interpolation=cv2.INTER_AREA)
    return img.copy()


def praproses(img, cfg):
    bgr = cv2.GaussianBlur(img, (cfg["blur_k"],) * 2, 0) if cfg["blur_k"] > 1 else img.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if cfg["eq_v"]:
        hsv[..., 2] = cv2.equalizeHist(np.ascontiguousarray(hsv[..., 2]))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr, hsv


def segmentasi(hsv, cfg):
    v_ref = int(np.median(hsv[..., 2]))
    mask = cv2.inRange(hsv, (0, cfg["s_min"], 0), (179, 255, int(cfg["v_rel"] * v_ref)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["morph_k"],) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def deteksi_kontur(mask, cfg):
    kontur, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in kontur if cv2.contourArea(c) >= cfg["min_area"] * mask.size]


def pisah_blob(blob, seed_min):
    """Heuristik erosi bertahap milik notebook; bukan detektor berbasis AI."""
    kernel = np.ones((3, 3), np.uint8)
    luas = cv2.countNonZero(blob)
    erosi, terbaik, iter_terbaik = blob.copy(), None, 0
    for it in range(1, 80):
        erosi = cv2.erode(erosi, kernel)
        kontur, _ = cv2.findContours(erosi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        seed = [c for c in kontur if cv2.contourArea(c) >= seed_min * luas]
        if not seed:
            break
        if len(seed) >= 2 and (terbaik is None or len(seed) > len(terbaik)):
            terbaik, iter_terbaik = seed, it
    if terbaik is None:
        return [blob]
    hasil = []
    for c in terbaik:
        s = np.zeros_like(blob)
        cv2.drawContours(s, [c], -1, 255, -1)
        s = cv2.dilate(s, kernel, iterations=iter_terbaik + 1)
        hasil.append(cv2.bitwise_and(s, blob))
    return hasil


def pisah_kontur(daftar_kontur, shape, cfg):
    hasil = []
    for c in daftar_kontur:
        blob = np.zeros(shape[:2], np.uint8)
        cv2.drawContours(blob, [c], -1, 255, -1)
        bagian = pisah_blob(blob, cfg["seed_min"]) if cfg["pisah_dempet"] else [blob]
        for b in bagian:
            kontur, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if kontur:
                hasil.append(max(kontur, key=cv2.contourArea))
    return hasil


def filter_bentuk(daftar_kontur, shape, cfg):
    tinggi, lebar = shape[:2]
    hasil = []
    for c in daftar_kontur:
        _, (w, h), _ = cv2.minAreaRect(c)
        luas = cv2.contourArea(c)
        if (luas >= cfg["min_area"] * tinggi * lebar
                and max(w, h) / max(min(w, h), 1) <= cfg["aspek_max"]
                and luas / max(w * h, 1) >= cfg["ekstent_min"]):
            hasil.append(c)
    return hasil


def mask_kontur(shape, kontur):
    m = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(m, [kontur], -1, 255, -1)
    dalam = cv2.erode(m, np.ones((5, 5), np.uint8))
    return dalam if cv2.countNonZero(dalam) else m


def ekstrak_fitur(bgr, hsv, kontur, cfg):
    m = mask_kontur(hsv.shape, kontur)
    jenuh = cv2.inRange(hsv, (0, cfg["s_min"], 0), (179, 255, 255))
    piksel = cv2.bitwise_and(m, jenuh)
    if cv2.countNonZero(piksel) == 0:
        piksel = m
    idx = piksel > 0
    hue = (hsv[..., 0][idx].astype(int) + GESER) % 180
    _b, g, r = bgr[idx].astype(float).mean(axis=0)
    return {
        "kontur": kontur, "bbox": cv2.boundingRect(kontur), "luas": cv2.contourArea(kontur),
        "h_med": float(np.median(hue)), "h_hist": np.bincount(hue, minlength=180),
        "rasio_rg": float(r / (r + g + 1e-6)),
        "hsv_med": tuple(int(np.median(hsv[..., i][m > 0])) for i in range(3)),
    }


def klasifikasi(fitur, cfg):
    if cfg["ruang"] == "bgr":
        t1, t2 = cfg["bgr_r"]
        q = fitur["rasio_rg"]
        return "mentah" if q < t1 else ("setengah_matang" if q < t2 else "matang")
    h = fitur["h_med"]
    if h <= cfg["h_matang"] + GESER:
        return "matang"
    return "setengah_matang" if h <= cfg["h_setengah"] + GESER else "mentah"


def jalankan_pipeline(img, cfg=None):
    cfg = buat_cfg(cfg)
    asli = resize_citra(img, cfg["sisi_max"])
    bgr, hsv = praproses(asli, cfg)
    mask = segmentasi(hsv, cfg)
    kontur = deteksi_kontur(mask, cfg)
    kontur = pisah_kontur(kontur, mask.shape, cfg)
    kontur = filter_bentuk(kontur, mask.shape, cfg)
    # Urutan stabil: atas ke bawah, lalu kiri ke kanan untuk penomoran di UI/CSV.
    kontur.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    hasil, hitung = [], dict.fromkeys(KELAS, 0)
    for c in kontur:
        fitur = ekstrak_fitur(bgr, hsv, c, cfg)
        kelas = klasifikasi(fitur, cfg)
        hitung[kelas] += 1
        hasil.append({**fitur, "kelas": kelas})
    return {"asli": asli, "bgr": bgr, "hsv": hsv, "mask": mask, "hasil": hasil,
            "hitung": hitung, "total": len(hasil), "cfg": cfg}


def gambar_hasil(p):
    out = p["asli"].copy()
    for nomor, b in enumerate(p["hasil"], 1):
        x, y, w, h = b["bbox"]
        warna = WARNA[b["kelas"]]
        cv2.rectangle(out, (x, y), (x + w, y + h), warna, 2)
        teks = f"{nomor}. {LABEL[b['kelas']]}"
        font, skala = cv2.FONT_HERSHEY_SIMPLEX, 0.45
        (tw, th), _ = cv2.getTextSize(teks, font, skala, 1)
        tx = max(0, min(x, out.shape[1] - tw - 8))
        ty = max(th + 8, y - 5)
        cv2.rectangle(out, (tx, ty - th - 6), (tx + tw + 8, ty + 4), (255, 255, 255), -1)
        cv2.putText(out, teks, (tx + 4, ty), font, skala, warna, 1, cv2.LINE_AA)
    return out


def tabel_hasil(p):
    return [{
        "nomor": i, "kelas": LABEL[b["kelas"]], "hue_median": round(b["h_med"] - GESER, 2),
        "rasio_rg": round(b["rasio_rg"], 4), "luas_px": round(b["luas"], 1),
        "x": b["bbox"][0], "y": b["bbox"][1], "lebar": b["bbox"][2], "tinggi": b["bbox"][3],
    } for i, b in enumerate(p["hasil"], 1)]


def buat_demo():
    """Ilustrasi sintetis, bukan foto dataset atau bukti akurasi."""
    hsv = np.full((480, 800, 3), (0, 0, 245), dtype=np.uint8)
    for pusat, hue in [((160, 240), 45), ((400, 240), 9), ((640, 240), 3)]:
        cv2.circle(hsv, pusat, 74, (hue, 210, 170), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

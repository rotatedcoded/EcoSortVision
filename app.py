from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
IMAGE_SIZE = 224

DEVICE = 0 if torch.cuda.is_available() else "cpu"

CLASS_INFO = {
    "battery": {
        "name": "Baterai",
        "icon": "🔋",
        "category": "Limbah khusus",
        "guide": (
            "Pisahkan dari sampah biasa. Simpan dalam kondisi kering dan "
            "serahkan ke fasilitas pengumpulan baterai atau limbah elektronik."
        ),
    },
    "biological": {
        "name": "Sampah Organik",
        "icon": "🍃",
        "category": "Organik",
        "guide": (
            "Pisahkan untuk pengomposan apabila memungkinkan. Hindari "
            "mencampurnya dengan plastik, kaca, atau bahan berbahaya."
        ),
    },
    "cardboard": {
        "name": "Kardus",
        "icon": "📦",
        "category": "Dapat didaur ulang",
        "guide": (
            "Pastikan kardus bersih dan kering. Lipat atau pipihkan sebelum "
            "dimasukkan ke tempat pengumpulan daur ulang."
        ),
    },
    "clothes": {
        "name": "Pakaian",
        "icon": "👕",
        "category": "Guna ulang atau tekstil",
        "guide": (
            "Pakaian yang masih layak dapat digunakan kembali atau "
            "didonasikan. Pakaian rusak dapat dikumpulkan sebagai limbah tekstil."
        ),
    },
    "glass": {
        "name": "Kaca",
        "icon": "🍾",
        "category": "Dapat didaur ulang",
        "guide": (
            "Pisahkan dari sampah lain. Bersihkan wadah kaca dan bungkus "
            "pecahan kaca dengan aman sebelum dibuang."
        ),
    },
    "metal": {
        "name": "Logam",
        "icon": "🥫",
        "category": "Dapat didaur ulang",
        "guide": (
            "Bersihkan sisa isi pada kaleng atau benda logam, lalu kumpulkan "
            "pada tempat daur ulang logam."
        ),
    },
    "paper": {
        "name": "Kertas",
        "icon": "📄",
        "category": "Dapat didaur ulang",
        "guide": (
            "Pastikan kertas tetap bersih dan kering. Pisahkan kertas yang "
            "terkontaminasi makanan atau cairan."
        ),
    },
    "plastic": {
        "name": "Plastik",
        "icon": "🧴",
        "category": "Periksa jenis plastik",
        "guide": (
            "Kosongkan dan bersihkan kemasan. Pisahkan sesuai jenis plastik "
            "dan aturan fasilitas daur ulang di daerahmu."
        ),
    },
    "shoes": {
        "name": "Sepatu",
        "icon": "👟",
        "category": "Guna ulang atau tekstil",
        "guide": (
            "Sepatu yang masih layak dapat digunakan kembali atau didonasikan. "
            "Sepatu rusak dapat diserahkan ke pengelola limbah tekstil."
        ),
    },
    "trash": {
        "name": "Sampah Campuran",
        "icon": "🗑️",
        "category": "Residu",
        "guide": (
            "Periksa kembali apakah ada bagian yang masih dapat dipisahkan. "
            "Buang residu yang tidak dapat didaur ulang ke tempat sampah umum."
        ),
    },
}


st.set_page_config(
    page_title="EcoSort Vision",
    page_icon="♻️",
    layout="wide",
)


@st.cache_resource(show_spinner="Memuat model EcoSort Vision...")
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan pada lokasi: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


def predict_image(
    model: YOLO,
    image: Image.Image,
) -> tuple[str, float, pd.DataFrame]:
    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
    )

    result = results[0]

    if result.probs is None:
        raise RuntimeError("Model tidak menghasilkan probabilitas klasifikasi.")

    names = result.names
    top1_id = int(result.probs.top1)
    top1_class = names[top1_id]
    top1_confidence = float(result.probs.top1conf.item())

    top_ids = [int(class_id) for class_id in result.probs.top5[:3]]
    top_confidences = [
        float(value)
        for value in result.probs.top5conf[:3].detach().cpu().tolist()
    ]

    rows = []

    for rank, (class_id, confidence) in enumerate(
        zip(top_ids, top_confidences),
        start=1,
    ):
        class_name = names[class_id]
        info = CLASS_INFO.get(
            class_name,
            {
                "name": class_name.title(),
                "icon": "♻️",
            },
        )

        rows.append(
            {
                "Peringkat": rank,
                "Prediksi": f"{info['icon']} {info['name']}",
                "Confidence": confidence,
            }
        )

    return top1_class, top1_confidence, pd.DataFrame(rows)


def display_prediction(
    model: YOLO,
    image_file,
) -> None:
    try:
        image = Image.open(image_file).convert("RGB")
    except Exception as error:
        st.error(f"Gambar tidak dapat dibuka: {error}")
        return

    image_column, result_column = st.columns([1, 1])

    with image_column:
        st.subheader("Gambar yang dianalisis")
        st.image(
            image,
            caption="Gambar masukan",
            use_container_width=True,
        )

    with st.spinner("Model sedang menganalisis gambar..."):
        try:
            predicted_class, confidence, top_predictions = predict_image(
                model,
                image,
            )
        except Exception as error:
            st.error(f"Prediksi gagal: {error}")
            return

    info = CLASS_INFO.get(
        predicted_class,
        {
            "name": predicted_class.title(),
            "icon": "♻️",
            "category": "Belum tersedia",
            "guide": "Periksa aturan pengelolaan sampah di daerahmu.",
        },
    )

    with result_column:
        st.subheader("Hasil prediksi")

        st.markdown(
            f"## {info['icon']} {info['name']}"
        )

        metric_left, metric_right = st.columns(2)

        with metric_left:
            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%",
            )

        with metric_right:
            st.metric(
                "Kategori",
                info["category"],
            )

        st.progress(
            min(max(int(confidence * 100), 0), 100)
        )

        if confidence >= 0.80:
            st.success("Model memiliki keyakinan tinggi terhadap prediksi ini.")
        elif confidence >= 0.60:
            st.warning(
                "Keyakinan model sedang. Coba gunakan foto yang lebih jelas."
            )
        else:
            st.error(
                "Keyakinan model rendah. Ambil ulang foto dengan pencahayaan "
                "dan latar belakang yang lebih baik."
            )

        st.markdown("### Saran pengelolaan")
        st.info(info["guide"])

    st.markdown("---")
    st.subheader("Tiga prediksi tertinggi")

    top_predictions["Confidence"] = top_predictions[
        "Confidence"
    ].map(lambda value: f"{value * 100:.2f}%")

    st.dataframe(
        top_predictions,
        hide_index=True,
        use_container_width=True,
    )


def main() -> None:
    st.title("♻️ EcoSort Vision")
    st.write(
        "Aplikasi Computer Vision untuk mengenali 10 kategori sampah "
        "menggunakan model YOLO11 Classification."
    )

    with st.sidebar:
        st.header("Tentang model")
        st.metric("Test accuracy", "94,24%")
        st.metric("Jumlah kelas", "10")
        st.metric("Gambar test", "1.233")

        runtime_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        )

        st.write(f"**Perangkat:** {runtime_name}")
        st.write(f"**Model:** `{MODEL_PATH.name}`")

        st.markdown("---")
        st.caption(
            "Hasil prediksi merupakan bantuan klasifikasi awal. "
            "Ikuti aturan pengelolaan sampah di daerah setempat."
        )

    try:
        model = load_model()
    except Exception as error:
        st.error(str(error))
        st.stop()

    upload_tab, camera_tab = st.tabs(
        ["📁 Upload gambar", "📷 Gunakan kamera"]
    )

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Pilih gambar sampah",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Gunakan satu objek utama dengan pencahayaan yang jelas.",
        )

        if uploaded_file is not None:
            display_prediction(model, uploaded_file)
        else:
            st.info("Unggah gambar untuk mulai melakukan prediksi.")

    with camera_tab:
        camera_file = st.camera_input(
            "Ambil foto objek sampah"
        )

        if camera_file is not None:
            display_prediction(model, camera_file)
        else:
            st.info("Aktifkan kamera dan ambil satu foto objek sampah.")


if __name__ == "__main__":
    main()

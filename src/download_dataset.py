from pathlib import Path
import sys

import kagglehub


DATASET_HANDLE = "sumn2u/garbage-classification-v2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DOWNLOAD DATASET ECO SORT VISION")
    print("=" * 70)
    print(f"Dataset : {DATASET_HANDLE}")
    print(f"Tujuan  : {OUTPUT_DIR}")
    print()

    try:
        downloaded_path = kagglehub.dataset_download(
            DATASET_HANDLE,
            output_dir=str(OUTPUT_DIR),
        )
    except Exception as error:
        print("\nDownload gagal.")
        print(f"Jenis error: {type(error).__name__}")
        print(f"Pesan      : {error}")
        sys.exit(1)

    images = [
        path
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print()
    print("=" * 70)
    print("DOWNLOAD SELESAI")
    print("=" * 70)
    print(f"Lokasi hasil : {downloaded_path}")
    print(f"Jumlah gambar: {len(images):,}")
    print()

    print("Folder yang ditemukan:")
    folders = sorted(
        path.relative_to(OUTPUT_DIR)
        for path in OUTPUT_DIR.rglob("*")
        if path.is_dir()
    )

    for folder in folders[:30]:
        print(f"- {folder}")

    if len(folders) > 30:
        print(f"- dan {len(folders) - 30} folder lainnya")


if __name__ == "__main__":
    main()

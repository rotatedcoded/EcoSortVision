import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "prepared"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
MODEL_DIR = PROJECT_ROOT / "models"

BASE_MODEL = "yolo11n-cls.pt"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training model klasifikasi EcoSort Vision."
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Menjalankan pengujian training selama 1 epoch.",
    )

    return parser.parse_args()


def validate_environment() -> None:
    required_directories = [
        DATASET_DIR / "train",
        DATASET_DIR / "val",
        DATASET_DIR / "test",
    ]

    missing = [
        directory
        for directory in required_directories
        if not directory.exists()
    ]

    if missing:
        print("ERROR: Struktur dataset belum lengkap.")

        for directory in missing:
            print(f"- Tidak ditemukan: {directory}")

        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA tidak aktif.")
        print("Training dihentikan agar tidak berjalan lambat menggunakan CPU.")
        sys.exit(1)


def count_images(directory: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def main() -> None:
    args = parse_arguments()
    validate_environment()

    smoke_test = args.smoke

    epochs = 1 if smoke_test else 30
    batch_size = 16 if smoke_test else 32
    run_name = "smoke_test" if smoke_test else "ecosort_yolo11n"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("TRAINING ECO SORT VISION")
    print("=" * 76)
    print(f"Mode          : {'SMOKE TEST' if smoke_test else 'TRAINING PENUH'}")
    print(f"Dataset       : {DATASET_DIR}")
    print(f"Model awal    : {BASE_MODEL}")
    print(f"Epoch         : {epochs}")
    print(f"Batch size    : {batch_size}")
    print(f"Image size    : 224")
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA          : {torch.version.cuda}")
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print()
    print(f"Train images  : {count_images(DATASET_DIR / 'train'):,}")
    print(f"Val images    : {count_images(DATASET_DIR / 'val'):,}")
    print(f"Test images   : {count_images(DATASET_DIR / 'test'):,}")
    print("=" * 76)
    print()

    model = YOLO(BASE_MODEL)

    try:
        model.train(
            data=str(DATASET_DIR),
            epochs=epochs,
            imgsz=224,
            batch=batch_size,
            device=0,
            workers=2,
            project=str(OUTPUT_DIR),
            name=run_name,
            exist_ok=True,
            pretrained=True,
            optimizer="auto",
            patience=8,
            seed=42,
            deterministic=True,
            amp=True,
            cache=False,
            plots=True,
            verbose=True,
        )
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            print("\nCUDA kehabisan memori.")
            print("Turunkan batch size dari 32 menjadi 16 pada file train.py.")
        raise

    best_model = Path(model.trainer.best)
    last_model = Path(model.trainer.last)

    print()
    print("=" * 76)
    print("TRAINING SELESAI")
    print("=" * 76)
    print(f"Folder hasil  : {model.trainer.save_dir}")
    print(f"Best model    : {best_model}")
    print(f"Last model    : {last_model}")

    if not smoke_test and best_model.exists():
        destination = MODEL_DIR / "best.pt"
        shutil.copy2(best_model, destination)

        print(f"Model deploy  : {destination}")
    elif smoke_test:
        print("Smoke test berhasil.")
        print("Model belum disalin ke folder models karena ini hanya pengujian.")


if __name__ == "__main__":
    main()

import csv
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "standardized_256"
OUTPUT_DIR = PROJECT_ROOT / "data" / "prepared"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "dataset_split_summary.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def link_or_copy(source: Path, destination: Path) -> str:
    """
    Mencoba membuat hard link agar tidak memakai ruang disk tambahan.
    Jika gagal, file akan disalin seperti biasa.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def split_files(files: list[Path]) -> dict[str, list[Path]]:
    random_generator = random.Random(SEED)
    shuffled = files.copy()
    random_generator.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def main() -> None:
    print("=" * 76)
    print("PERSIAPAN DATASET ECO SORT VISION")
    print("=" * 76)
    print(f"Sumber        : {SOURCE_DIR}")
    print(f"Tujuan        : {OUTPUT_DIR}")
    print(f"Seed          : {SEED}")
    print("Pembagian     : train 70% | val 20% | test 10%")
    print()

    if not SOURCE_DIR.exists():
        print(f"ERROR: Folder sumber tidak ditemukan: {SOURCE_DIR}")
        sys.exit(1)

    class_directories = sorted(
        folder
        for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    )

    if not class_directories:
        print("ERROR: Tidak ada folder kelas di dataset.")
        sys.exit(1)

    if OUTPUT_DIR.exists():
        print("Menghapus hasil pembagian lama...")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    totals = defaultdict(int)
    invalid_images = []
    operation_counts = defaultdict(int)

    for class_directory in class_directories:
        class_name = class_directory.name

        candidates = sorted(
            path
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        valid_images = []

        for image_path in candidates:
            if is_valid_image(image_path):
                valid_images.append(image_path)
            else:
                invalid_images.append(image_path)

        splits = split_files(valid_images)

        print(f"Kelas: {class_name}")
        print(f"  Gambar valid : {len(valid_images):,}")

        for split_name, split_images in splits.items():
            destination_class = OUTPUT_DIR / split_name / class_name
            destination_class.mkdir(parents=True, exist_ok=True)

            for index, source_path in enumerate(split_images, start=1):
                destination_name = (
                    f"{index:05d}_{source_path.stem}{source_path.suffix.lower()}"
                )
                destination_path = destination_class / destination_name

                operation = link_or_copy(source_path, destination_path)
                operation_counts[operation] += 1

            count = len(split_images)
            totals[split_name] += count

            summary_rows.append(
                {
                    "class": class_name,
                    "train" if split_name == "train" else split_name: count,
                }
            )

            print(f"  {split_name:<5}        : {count:>5,}")

        print()

    class_summary = []

    for class_directory in class_directories:
        class_name = class_directory.name

        train_count = len(list((OUTPUT_DIR / "train" / class_name).glob("*")))
        val_count = len(list((OUTPUT_DIR / "val" / class_name).glob("*")))
        test_count = len(list((OUTPUT_DIR / "test" / class_name).glob("*")))

        class_summary.append(
            {
                "class": class_name,
                "train": train_count,
                "val": val_count,
                "test": test_count,
                "total": train_count + val_count + test_count,
            }
        )

    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["class", "train", "val", "test", "total"],
        )
        writer.writeheader()
        writer.writerows(class_summary)

    print("=" * 76)
    print("PEMBAGIAN DATASET SELESAI")
    print("=" * 76)
    print(f"Train         : {totals['train']:,} gambar")
    print(f"Validation    : {totals['val']:,} gambar")
    print(f"Test          : {totals['test']:,} gambar")
    print(f"Total         : {sum(totals.values()):,} gambar")
    print(f"Gambar rusak  : {len(invalid_images):,}")
    print(f"Hard link     : {operation_counts['hardlink']:,}")
    print(f"File disalin  : {operation_counts['copy']:,}")
    print(f"Ringkasan CSV : {SUMMARY_PATH}")

    if invalid_images:
        print("\nDaftar gambar yang tidak valid:")

        for image_path in invalid_images:
            print(f"- {image_path}")


if __name__ == "__main__":
    main()

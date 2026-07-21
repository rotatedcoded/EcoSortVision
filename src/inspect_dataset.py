from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

VERSIONS = [
    RAW_DIR / "original",
    RAW_DIR / "standardized_256",
    RAW_DIR / "standardized_384",
]


def count_images(folder: Path) -> Counter:
    counts = Counter()

    if not folder.exists():
        return counts

    for class_dir in sorted(folder.iterdir()):
        if not class_dir.is_dir():
            continue

        total = sum(
            1
            for file in class_dir.rglob("*")
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        if total > 0:
            counts[class_dir.name] = total

    return counts


def main() -> None:
    print("=" * 72)
    print("PEMERIKSAAN DATASET ECO SORT VISION")
    print("=" * 72)

    for version_dir in VERSIONS:
        print(f"\nVERSI: {version_dir.name}")
        print("-" * 72)

        counts = count_images(version_dir)

        if not counts:
            print("Folder tidak ditemukan atau tidak memiliki gambar.")
            continue

        for class_name, total in sorted(counts.items()):
            print(f"{class_name:<15} : {total:>6,} gambar")

        print("-" * 72)
        print(f"Jumlah kelas   : {len(counts)}")
        print(f"Jumlah gambar  : {sum(counts.values()):,}")
        print(f"Kelas terbesar : {max(counts, key=counts.get)} ({max(counts.values()):,})")
        print(f"Kelas terkecil : {min(counts, key=counts.get)} ({min(counts.values()):,})")

    selected_dir = RAW_DIR / "standardized_256"

    print("\n" + "=" * 72)
    print("DATASET YANG AKAN DIGUNAKAN")
    print("=" * 72)
    print(selected_dir)

    if not selected_dir.exists():
        raise FileNotFoundError(
            f"Folder dataset pilihan tidak ditemukan: {selected_dir}"
        )


if __name__ == "__main__":
    main()

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
TEST_DIR = PROJECT_ROOT / "data" / "prepared" / "test"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

IMAGE_SIZE = 224
BATCH_SIZE = 32


def get_class_names(model: YOLO) -> list[str]:
    names = model.names

    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]

    return list(names)


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    normalized: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=axis,
        xticks_rotation=45,
        values_format=".2f" if normalized else "d",
        colorbar=False,
    )

    title = (
        "Normalized Confusion Matrix — EcoSort Vision"
        if normalized
        else "Confusion Matrix — EcoSort Vision"
    )

    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    print("=" * 78)
    print("EVALUASI MODEL ECO SORT VISION")
    print("=" * 78)
    print(f"Model       : {MODEL_PATH}")
    print(f"Test data   : {TEST_DIR}")
    print(f"Output      : {OUTPUT_DIR}")
    print(f"Image size  : {IMAGE_SIZE}")
    print(f"Batch size  : {BATCH_SIZE}")
    print()

    if not MODEL_PATH.exists():
        print(f"ERROR: Model tidak ditemukan: {MODEL_PATH}")
        sys.exit(1)

    if not TEST_DIR.exists():
        print(f"ERROR: Dataset test tidak ditemukan: {TEST_DIR}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA tidak aktif.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in TEST_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print("ERROR: Tidak ada gambar di folder test.")
        sys.exit(1)

    print(f"Jumlah gambar test: {len(image_paths):,}")
    print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print()

    model = YOLO(str(MODEL_PATH))
    class_names = get_class_names(model)
    name_to_id = {
        class_name: class_id
        for class_id, class_name in enumerate(class_names)
    }

    dataset_classes = sorted(
        folder.name
        for folder in TEST_DIR.iterdir()
        if folder.is_dir()
    )

    if set(dataset_classes) != set(class_names):
        print("ERROR: Nama kelas model dan dataset tidak sama.")
        print(f"Kelas model   : {class_names}")
        print(f"Kelas dataset : {dataset_classes}")
        sys.exit(1)

    print(f"Jumlah kelas: {len(class_names)}")
    print(f"Kelas       : {', '.join(class_names)}")
    print()
    print("Menjalankan prediksi test...")

    rows = []
    y_true = []
    y_pred = []
    top5_correct = []

    start_time = time.perf_counter()

    for start_index in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[start_index : start_index + BATCH_SIZE]

        results = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=0,
            verbose=False,
        )

        for source_path, result in zip(batch_paths, results):
            if result.probs is None:
                raise RuntimeError(
                    f"Hasil klasifikasi tidak tersedia untuk {result.path}"
                )

            result_path = source_path
            true_name = source_path.parent.name

            if true_name not in name_to_id:
                raise ValueError(
                    f"Kelas ground truth tidak dikenal: {true_name}"
                )

            true_id = name_to_id[true_name]
            predicted_id = int(result.probs.top1)
            predicted_name = class_names[predicted_id]
            confidence = float(result.probs.top1conf.item())

            top5_ids = [int(value) for value in result.probs.top5]
            top5_names = [class_names[value] for value in top5_ids]
            top5_confidences = [
                float(value)
                for value in result.probs.top5conf.detach().cpu().tolist()
            ]

            is_correct = true_id == predicted_id
            is_top5_correct = true_id in top5_ids

            y_true.append(true_id)
            y_pred.append(predicted_id)
            top5_correct.append(is_top5_correct)

            rows.append(
                {
                    "file": str(result_path.relative_to(PROJECT_ROOT)),
                    "true_class": true_name,
                    "predicted_class": predicted_name,
                    "confidence": confidence,
                    "correct": is_correct,
                    "top5_correct": is_top5_correct,
                    "top5_classes": " | ".join(top5_names),
                    "top5_confidences": " | ".join(
                        f"{value:.6f}"
                        for value in top5_confidences
                    ),
                }
            )

        completed = min(start_index + BATCH_SIZE, len(image_paths))

        print(
            f"\rDiproses: {completed:,}/{len(image_paths):,}",
            end="",
            flush=True,
        )

    elapsed_seconds = time.perf_counter() - start_time
    print("\n")

    predictions = pd.DataFrame(rows)

    top1_accuracy = float(accuracy_score(y_true, y_pred))
    top5_accuracy = float(np.mean(top5_correct))
    incorrect_count = int((~predictions["correct"]).sum())

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_rows = []

    for class_name in class_names:
        class_metrics = report[class_name]

        report_rows.append(
            {
                "class": class_name,
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "f1_score": class_metrics["f1-score"],
                "support": int(class_metrics["support"]),
            }
        )

    per_class_report = pd.DataFrame(report_rows)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    row_totals = matrix.sum(axis=1, keepdims=True)

    normalized_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    mistakes_path = OUTPUT_DIR / "misclassified_predictions.csv"
    report_path = OUTPUT_DIR / "per_class_metrics.csv"
    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    matrix_path = OUTPUT_DIR / "confusion_matrix.png"
    normalized_matrix_path = (
        OUTPUT_DIR / "confusion_matrix_normalized.png"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    predictions[
        ~predictions["correct"]
    ].sort_values(
        by="confidence",
        ascending=False,
    ).to_csv(
        mistakes_path,
        index=False,
        encoding="utf-8-sig",
    )

    per_class_report.to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    save_confusion_matrix(
        matrix,
        class_names,
        matrix_path,
        normalized=False,
    )

    save_confusion_matrix(
        normalized_matrix,
        class_names,
        normalized_matrix_path,
        normalized=True,
    )

    summary = {
        "model": str(MODEL_PATH),
        "test_directory": str(TEST_DIR),
        "number_of_images": len(image_paths),
        "number_of_classes": len(class_names),
        "classes": class_names,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "correct_predictions": len(image_paths) - incorrect_count,
        "incorrect_predictions": incorrect_count,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "elapsed_seconds": elapsed_seconds,
        "average_ms_per_image": (
            elapsed_seconds / len(image_paths)
        ) * 1000,
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("=" * 78)
    print("HASIL EVALUASI TEST")
    print("=" * 78)
    print(f"Top-1 accuracy       : {top1_accuracy:.4%}")
    print(f"Top-5 accuracy       : {top5_accuracy:.4%}")
    print(f"Prediksi benar       : {len(image_paths) - incorrect_count:,}")
    print(f"Prediksi salah       : {incorrect_count:,}")
    print(f"Macro F1-score       : {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-score    : {report['weighted avg']['f1-score']:.4f}")
    print(f"Total waktu          : {elapsed_seconds:.2f} detik")
    print(
        "Rata-rata per gambar : "
        f"{summary['average_ms_per_image']:.2f} ms"
    )

    print("\nHASIL PER KELAS")
    print("-" * 78)

    for row in report_rows:
        print(
            f"{row['class']:<12} "
            f"precision={row['precision']:.4f}  "
            f"recall={row['recall']:.4f}  "
            f"f1={row['f1_score']:.4f}  "
            f"support={row['support']}"
        )

    print("\nFILE HASIL")
    print("-" * 78)
    print(f"Ringkasan JSON       : {summary_path}")
    print(f"Semua prediksi       : {predictions_path}")
    print(f"Prediksi salah       : {mistakes_path}")
    print(f"Metrik per kelas     : {report_path}")
    print(f"Confusion matrix     : {matrix_path}")
    print(f"Confusion normalized : {normalized_matrix_path}")


if __name__ == "__main__":
    main()


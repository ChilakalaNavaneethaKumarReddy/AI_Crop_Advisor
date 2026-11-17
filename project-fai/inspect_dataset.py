"""
================================================================================
AI Crop Advisor – Dataset Inspection Script
================================================================================
Scans and verifies all three supported crop datasets:
1. Paddy (10 classes)
2. Cotton (4 classes)
3. Chilli (5 classes)
Total: 19 classes / 12,648 images
================================================================================
"""

import os
from pathlib import Path
from PIL import Image

def inspect_crop_datasets(data_dir=None):
    if data_dir is None:
        workspace_root = Path(__file__).resolve().parent.parent
        data_dir = workspace_root / "data" / "images"
    else:
        data_dir = Path(data_dir)

    print("=" * 65)
    print("AI CROP ADVISOR: DATASET INSPECTION REPORT")
    print("=" * 65)
    print(f"Data Directory: {data_dir}\n")

    paddy_dir = data_dir / "paddy" / "train_images"
    cotton_dir = data_dir / "cotton" / "cotton"
    chilli_dir = data_dir / "chilli" / "cropped" / "cropped"

    chilli_map = {
        "cercospora": "Chilli___cercospora",
        "healthy": "Chilli___healthy",
        "murda complex(mites,trips)": "Chilli___murda_complex",
        "nutritional": "Chilli___nutritional",
        "powdery mildew": "Chilli___powdery_mildew"
    }

    grand_total_images = 0
    all_formats = set()
    corrupted_files = []
    crop_reports = {}

    crops_to_scan = [
        ("Paddy", paddy_dir, lambda f_name: f"Paddy___{f_name}"),
        ("Cotton", cotton_dir, lambda f_name: f"Cotton___{f_name}"),
        ("Chilli", chilli_dir, lambda f_name: chilli_map.get(f_name, None))
    ]

    for crop_name, root_path, naming_fn in crops_to_scan:
        print("-" * 65)
        print(f"[*] Scanning {crop_name} Dataset: {root_path}")
        print("-" * 65)

        if not root_path.exists():
            print(f"  [ERROR] Directory not found: {root_path}")
            continue

        class_counts = {}
        crop_images = 0

        for folder in sorted(root_path.iterdir()):
            if folder.is_dir():
                target_class = naming_fn(folder.name)
                if not target_class:
                    continue
                count = 0
                for img_file in folder.glob("*.*"):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".mpo"]:
                        try:
                            with Image.open(img_file) as img:
                                img.verify()
                                all_formats.add(img.format if img.format else img_file.suffix.upper())
                                count += 1
                        except Exception as e:
                            corrupted_files.append((str(img_file), str(e)))

                class_counts[target_class] = count
                crop_images += count
                print(f"  - {target_class:<35} : {count:>5} images")

        crop_reports[crop_name] = {
            "classes": len(class_counts),
            "images": crop_images,
            "details": class_counts
        }
        grand_total_images += crop_images
        print(f"\n  -> Total {crop_name}: {len(class_counts)} classes | {crop_images} images\n")

    print("=" * 65)
    print("FINAL DATASET SUMMARY")
    print("=" * 65)
    total_classes = sum(r["classes"] for r in crop_reports.values())
    for crop_name, r in crop_reports.items():
        print(f"- {crop_name:<10}: {r['classes']:>2} classes / {r['images']:>6} images")
    print("-" * 65)
    print(f"TOTAL      : {total_classes:>2} classes / {grand_total_images:>6} images")
    print(f"Formats    : {', '.join(sorted(all_formats))}")
    print(f"Corrupted  : {len(corrupted_files)} unreadable files")
    print("=" * 65)

if __name__ == "__main__":
    inspect_crop_datasets()

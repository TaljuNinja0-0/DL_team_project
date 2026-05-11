"""
Step 1. Waldo 필터링 + 재라벨링
- 클래스 1(Waldo) 라인만 추출
- 클래스 ID → 0으로 리맵핑
- Waldo 있는 이미지/라벨만 남기기
- train / val / test 전부 처리

폴더 구조 가정:
processed/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

결과 폴더 구조 (dataset_waldo):
dataset_waldo/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
"""

import shutil
from pathlib import Path


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATASET_ROOT   = Path(r"C:\Users\bibin\Downloads\where's_waldo_archive\processed")       # 원본 데이터셋 경로 (수정)
OUTPUT_ROOT    = Path(r"C:\Users\bibin\Downloads\where's_waldo_archive\processed_waldo")   # 결과 저장 경로
WALDO_CLASS_ID = 1                       # 원본에서 Waldo 클래스 번호
SPLITS = ["train", "val", "test"]


def filter_waldo(label_path: Path) -> list[str]:
    """
    라벨 파일에서 Waldo(클래스 1) 라인만 추출하고
    클래스 ID를 0으로 리맵핑해서 반환.
    Waldo 없으면 빈 리스트 반환.
    """
    waldo_lines = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            if cls_id == WALDO_CLASS_ID:
                # 클래스 ID를 0으로 리맵핑
                parts[0] = "0"
                waldo_lines.append(" ".join(parts))
    return waldo_lines


def process_split(split: str):
    img_src_dir   = DATASET_ROOT / split / "images"
    label_src_dir = DATASET_ROOT / split / "labels"

    img_dst_dir   = OUTPUT_ROOT / split / "images"
    label_dst_dir = OUTPUT_ROOT / split / "labels"

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)

    # 지원 이미지 확장자
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    total = kept = skipped = 0

    label_files = sorted(label_src_dir.glob("*.txt"))
    for label_path in label_files:
        total += 1
        waldo_lines = filter_waldo(label_path)

        if not waldo_lines:
            skipped += 1
            continue  # Waldo 없는 이미지 제외

        # 대응 이미지 찾기
        img_path = None
        for ext in img_exts:
            candidate = img_src_dir / (label_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"  [경고] 이미지 없음: {label_path.stem}")
            skipped += 1
            continue

        # 이미지 복사
        shutil.copy2(img_path, img_dst_dir / img_path.name)

        # 라벨 저장 (Waldo 라인만)
        dst_label = label_dst_dir / label_path.name
        with open(dst_label, "w") as f:
            f.write("\n".join(waldo_lines) + "\n")

        kept += 1

    print(f"[{split:5s}] 전체 {total}장 → Waldo 있음 {kept}장 / 제외 {skipped}장")
    return total, kept, skipped


def main():
    print("=" * 50)
    print("  Step 1. Waldo 필터링 + 재라벨링")
    print("=" * 50)

    grand_total = grand_kept = grand_skip = 0

    for split in SPLITS:
        img_dir   = DATASET_ROOT / split / "images"
        label_dir = DATASET_ROOT / split / "labels"

        if not img_dir.exists() or not label_dir.exists():
            print(f"[{split}] 폴더 없음 → 스킵")
            continue

        t, k, s = process_split(split)
        grand_total += t
        grand_kept  += k
        grand_skip  += s

    print("=" * 50)
    print(f"  총계: {grand_total}장 → 유지 {grand_kept}장 / 제외 {grand_skip}장")
    print(f"  결과 저장 위치: {OUTPUT_ROOT.resolve()}")
    print("=" * 50)

    # data.yaml 생성
    yaml_path = OUTPUT_ROOT / "data.yaml"
    yaml_content = f"""\
path: {OUTPUT_ROOT.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: 1
names: ['waldo']
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  data.yaml 생성 완료: {yaml_path}")


if __name__ == "__main__":
    main()
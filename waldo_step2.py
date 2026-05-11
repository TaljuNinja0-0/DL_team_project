"""
Step 2. 데이터 검수
- 이미지-라벨 1:1 매칭 확인
- bbox 경계 벗어나는 거 클리핑 (0~1 범위 강제)
- 빈 라벨 / 이상한 bbox 감지
- 샘플 시각화 (랜덤 30장)

dataset_waldo/ (step1 결과) 기준으로 실행
"""

import os
import random
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATASET_ROOT  = Path("processed_waldo")  # step1 결과 경로 (수정)
VIZ_OUTPUT    = Path("inspection_viz")  # 시각화 저장 폴더
SPLITS        = ["train", "val", "test"]
VIZ_SAMPLES   = 30                      # 시각화할 샘플 수
RANDOM_SEED   = 42


# ─────────────────────────────────────────────
# 1. bbox 클리핑 유틸
# ─────────────────────────────────────────────
def clip_bbox(cx, cy, w, h):
    """
    YOLO 포맷 (cx, cy, w, h) 모두 0~1 범위로 클리핑.
    cx, cy 는 중심점이므로 w/h 절반이 넘어가면 잘라냄.
    반환: (clipped_cx, clipped_cy, clipped_w, clipped_h), was_clipped
    """
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    x1c = max(0.0, min(1.0, x1))
    y1c = max(0.0, min(1.0, y1))
    x2c = max(0.0, min(1.0, x2))
    y2c = max(0.0, min(1.0, y2))

    was_clipped = (x1c != x1 or y1c != y1 or x2c != x2 or y2c != y2)

    new_w  = x2c - x1c
    new_h  = y2c - y1c
    new_cx = x1c + new_w / 2
    new_cy = y1c + new_h / 2

    return new_cx, new_cy, new_w, new_h, was_clipped


def is_valid_bbox(cx, cy, w, h):
    """bbox가 유효한지 확인 (크기 > 0, 범위 0~1)"""
    if not (0 <= cx <= 1 and 0 <= cy <= 1):
        return False
    if w <= 0 or h <= 0:
        return False
    if w > 1 or h > 1:
        return False
    return True


# ─────────────────────────────────────────────
# 2. 검수 + 클리핑 메인 함수
# ─────────────────────────────────────────────
def inspect_and_clip_split(split: str):
    img_dir   = DATASET_ROOT / split / "images"
    label_dir = DATASET_ROOT / split / "labels"

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    stats = {
        "total_labels"   : 0,
        "missing_image"  : [],   # 라벨은 있는데 이미지 없음
        "missing_label"  : [],   # 이미지는 있는데 라벨 없음
        "empty_label"    : [],   # 라벨 파일이 비어있음
        "clipped"        : [],   # bbox 클리핑된 파일
        "invalid_removed": [],   # 클리핑 후에도 크기 0이라 제거된 bbox
        "total_bboxes"   : 0,
        "clipped_bboxes" : 0,
    }

    label_files = sorted(label_dir.glob("*.txt"))
    stats["total_labels"] = len(label_files)

    for label_path in label_files:
        # 대응 이미지 확인
        img_path = None
        for ext in img_exts:
            candidate = img_dir / (label_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            stats["missing_image"].append(label_path.name)
            continue

        # 라벨 읽기
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            stats["empty_label"].append(label_path.name)
            continue

        new_lines     = []
        file_clipped  = False
        file_invalid  = False

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue  # 포맷 이상 → 스킵

            cls_id = parts[0]
            cx, cy, w, h = map(float, parts[1:])

            stats["total_bboxes"] += 1

            # 클리핑
            new_cx, new_cy, new_w, new_h, was_clipped = clip_bbox(cx, cy, w, h)

            if was_clipped:
                file_clipped = True
                stats["clipped_bboxes"] += 1

            # 클리핑 후 유효성 체크
            if not is_valid_bbox(new_cx, new_cy, new_w, new_h):
                file_invalid = True
                stats["invalid_removed"].append(f"{label_path.name} bbox({cx:.3f},{cy:.3f},{w:.3f},{h:.3f})")
                continue  # 이 bbox 제거

            new_lines.append(f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")

        if file_clipped:
            stats["clipped"].append(label_path.name)

        # 클리핑된 라벨 덮어쓰기
        if file_clipped or file_invalid:
            with open(label_path, "w") as f:
                f.write("\n".join(new_lines) + "\n" if new_lines else "")

    # 이미지는 있는데 라벨 없는 경우
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in img_exts:
            continue
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            stats["missing_label"].append(img_path.name)

    return stats


# ─────────────────────────────────────────────
# 3. 시각화
# ─────────────────────────────────────────────
def visualize_samples(split: str, n: int = VIZ_SAMPLES):
    img_dir   = DATASET_ROOT / split / "images"
    label_dir = DATASET_ROOT / split / "labels"
    out_dir   = VIZ_OUTPUT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    label_files = sorted(label_dir.glob("*.txt"))

    random.seed(RANDOM_SEED)
    samples = random.sample(label_files, min(n, len(label_files)))

    for label_path in samples:
        img_path = None
        for ext in img_exts:
            candidate = img_dir / (label_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img)

        with open(label_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = parts
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)

            # YOLO → pixel 좌표
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            bw = w * W
            bh = h * H

            rect = patches.Rectangle(
                (x1, y1), bw, bh,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, "waldo", color="red", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.5, pad=1))

        ax.set_title(f"{label_path.stem}", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{label_path.stem}.jpg", dpi=80, bbox_inches="tight")
        plt.close()

    print(f"  [{split}] 시각화 {len(samples)}장 저장 → {out_dir}")


# ─────────────────────────────────────────────
# 4. 리포트 출력
# ─────────────────────────────────────────────
def print_report(split: str, stats: dict):
    print(f"\n{'='*50}")
    print(f"  [{split}] 검수 결과")
    print(f"{'='*50}")
    print(f"  라벨 파일 수        : {stats['total_labels']}개")
    print(f"  전체 bbox 수        : {stats['total_bboxes']}개")
    print(f"  클리핑된 bbox       : {stats['clipped_bboxes']}개")
    print(f"  클리핑된 파일       : {len(stats['clipped'])}개")
    print(f"  이미지 없는 라벨    : {len(stats['missing_image'])}개")
    print(f"  라벨 없는 이미지    : {len(stats['missing_label'])}개")
    print(f"  빈 라벨 파일        : {len(stats['empty_label'])}개")
    print(f"  클리핑 후 제거 bbox : {len(stats['invalid_removed'])}개")

    if stats["missing_image"]:
        print(f"\n  ⚠️  이미지 없는 라벨:")
        for f in stats["missing_image"]:
            print(f"      - {f}")

    if stats["missing_label"]:
        print(f"\n  ⚠️  라벨 없는 이미지:")
        for f in stats["missing_label"]:
            print(f"      - {f}")

    if stats["invalid_removed"]:
        print(f"\n  ⚠️  클리핑 후 제거된 bbox:")
        for f in stats["invalid_removed"]:
            print(f"      - {f}")

    if (len(stats["missing_image"]) == 0 and
        len(stats["missing_label"]) == 0 and
        len(stats["empty_label"]) == 0 and
        stats["clipped_bboxes"] == 0):
        print("\n  ✅ 이상 없음!")


# ─────────────────────────────────────────────
# 5. 메인
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Step 2. 데이터 검수 시작")
    print("=" * 50)

    VIZ_OUTPUT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        img_dir   = DATASET_ROOT / split / "images"
        label_dir = DATASET_ROOT / split / "labels"

        if not img_dir.exists() or not label_dir.exists():
            print(f"\n[{split}] 폴더 없음 → 스킵")
            continue

        print(f"\n[{split}] 검수 중...")
        stats = inspect_and_clip_split(split)
        print_report(split, stats)

        print(f"\n[{split}] 시각화 생성 중...")
        visualize_samples(split, n=VIZ_SAMPLES)

    print(f"\n{'='*50}")
    print(f"  검수 완료!")
    print(f"  시각화 결과: {VIZ_OUTPUT.resolve()}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
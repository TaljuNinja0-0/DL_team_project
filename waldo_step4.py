"""
Step 4. 데이터 증강
- 목표: train 124장 → 300장 이상
- 적용 증강: 상하반전, 밝기/대비/채도, Cutout, Crop
- val/test는 증강 안 함 (검증 데이터 오염 방지)

폴더 구조 (step1 결과):
processed_waldo/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

결과 폴더 (augmented_waldo):
augmented_waldo/
├── train/
│   ├── images/   ← 원본 124장 + 증강본
│   └── labels/
├── val/          ← 그대로 복사
└── test/         ← 그대로 복사
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATASET_ROOT = Path(r"C:\Users\bibin\Downloads\where's_waldo_archive\processed_waldo")   # step1 결과 경로 (수정)
OUTPUT_ROOT  = Path(r"C:\Users\bibin\Downloads\where's_waldo_archive\augmented_waldo")   # 증강 결과 저장 경로
TARGET_COUNT = 300                       # 목표 train 이미지 수
RANDOM_SEED  = 42
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# 1. 증강 함수들
# ─────────────────────────────────────────────

def aug_flipud(img, bboxes):
    """상하 반전"""
    img_out = cv2.flip(img, 0)
    bboxes_out = []
    for cls_id, cx, cy, w, h in bboxes:
        bboxes_out.append((cls_id, cx, 1.0 - cy, w, h))
    return img_out, bboxes_out


def aug_brightness(img, bboxes, factor_range=(0.6, 1.4)):
    """밝기 조정"""
    factor = random.uniform(*factor_range)
    img_out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return img_out, bboxes


def aug_contrast(img, bboxes, factor_range=(0.6, 1.4)):
    """대비 조정"""
    factor = random.uniform(*factor_range)
    mean = img.mean()
    img_out = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    return img_out, bboxes


def aug_saturation(img, bboxes, factor_range=(0.5, 1.5)):
    """채도 조정 (HSV 변환)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(*factor_range)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    img_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img_out, bboxes


def aug_cutout(img, bboxes, n_holes=3, hole_ratio=0.08):
    """
    Cutout: 랜덤 영역을 회색으로 가림
    월리가 일부 가려진 상황 학습용
    """
    img_out = img.copy()
    H, W = img_out.shape[:2]
    for _ in range(n_holes):
        hole_w = int(W * hole_ratio)
        hole_h = int(H * hole_ratio)
        x = random.randint(0, W - hole_w)
        y = random.randint(0, H - hole_h)
        img_out[y:y+hole_h, x:x+hole_w] = 128  # 회색으로 가림
    return img_out, bboxes


def aug_crop(img, bboxes, crop_ratio_range=(0.75, 0.95)):
    """
    Waldo 중심으로 crop 증강
    Waldo가 반드시 포함되도록 crop 영역 설정
    bbox는 crop 기준으로 재계산
    """
    H, W = img.shape[:2]

    if not bboxes:
        return img, bboxes

    # 랜덤 Waldo 하나 선택해서 중심으로 crop
    cls_id, cx, cy, bw, bh = random.choice(bboxes)
    cx_px = int(cx * W)
    cy_px = int(cy * H)

    crop_ratio = random.uniform(*crop_ratio_range)
    crop_w = int(W * crop_ratio)
    crop_h = int(H * crop_ratio)

    # crop 좌상단 계산 (Waldo가 crop 안에 들어오도록)
    x1 = cx_px - crop_w // 2
    y1 = cy_px - crop_h // 2
    x1 = max(0, min(W - crop_w, x1))
    y1 = max(0, min(H - crop_h, y1))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    img_crop = img[y1:y2, x1:x2]
    img_out  = cv2.resize(img_crop, (W, H))  # 원래 크기로 복원

    # bbox 재계산
    bboxes_out = []
    for cls_id, cx, cy, bw, bh in bboxes:
        # 픽셀 좌표로 변환
        abs_cx = cx * W
        abs_cy = cy * H
        abs_w  = bw * W
        abs_h  = bh * H

        # crop 기준 상대 좌표
        new_cx = (abs_cx - x1) / crop_w
        new_cy = (abs_cy - y1) / crop_h
        new_w  = abs_w / crop_w
        new_h  = abs_h / crop_h

        # crop 범위 벗어나는 bbox 제거
        if not (0 < new_cx < 1 and 0 < new_cy < 1):
            continue
        # 경계 클리핑
        new_cx = np.clip(new_cx, 0, 1)
        new_cy = np.clip(new_cy, 0, 1)
        new_w  = np.clip(new_w, 0, 1)
        new_h  = np.clip(new_h, 0, 1)

        if new_w > 0 and new_h > 0:
            bboxes_out.append((cls_id, new_cx, new_cy, new_w, new_h))

    # crop 후 Waldo 없으면 원본 반환
    if not bboxes_out:
        return img, bboxes

    return img_out, bboxes_out


# ─────────────────────────────────────────────
# 2. 증강 파이프라인
# ─────────────────────────────────────────────

# 사용할 증강 조합 목록 (각 조합이 1장씩 생성)
AUG_PIPELINE = [
    ("flipud",              lambda img, bb: aug_flipud(img, bb)),
    ("brightness_up",       lambda img, bb: aug_brightness(img, bb, (1.2, 1.5))),
    ("brightness_down",     lambda img, bb: aug_brightness(img, bb, (0.5, 0.8))),
    ("contrast",            lambda img, bb: aug_contrast(img, bb)),
    ("saturation",          lambda img, bb: aug_saturation(img, bb)),
    ("cutout",              lambda img, bb: aug_cutout(img, bb)),
    ("crop",                lambda img, bb: aug_crop(img, bb)),
    ("flipud_brightness",   lambda img, bb: aug_brightness(*aug_flipud(img, bb))),
    ("flipud_cutout",       lambda img, bb: aug_cutout(*aug_flipud(img, bb))),
    ("flipud_crop",         lambda img, bb: aug_crop(*aug_flipud(img, bb))),
]


def read_label(label_path):
    """YOLO 라벨 읽기 → [(cls_id, cx, cy, w, h), ...]"""
    bboxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                bboxes.append((int(parts[0]), float(parts[1]),
                               float(parts[2]), float(parts[3]), float(parts[4])))
    return bboxes


def write_label(label_path, bboxes):
    """YOLO 라벨 저장"""
    with open(label_path, "w") as f:
        for cls_id, cx, cy, w, h in bboxes:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ─────────────────────────────────────────────
# 3. 메인 증강 처리
# ─────────────────────────────────────────────

def augment_train():
    img_src_dir   = DATASET_ROOT / "train" / "images"
    label_src_dir = DATASET_ROOT / "train" / "labels"
    img_dst_dir   = OUTPUT_ROOT  / "train" / "images"
    label_dst_dir = OUTPUT_ROOT  / "train" / "labels"

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)

    # 원본 파일 목록
    label_files = sorted(label_src_dir.glob("*.txt"))
    original_count = len(label_files)

    print(f"  원본 train 이미지: {original_count}장")
    print(f"  목표: {TARGET_COUNT}장")
    print(f"  필요 증강본: {TARGET_COUNT - original_count}장\n")

    # ── 원본 복사
    copied = 0
    src_pairs = []  # (img_path, label_path) 원본 쌍
    for label_path in label_files:
        img_path = None
        for ext in IMG_EXTS:
            candidate = img_src_dir / (label_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        shutil.copy2(img_path,   img_dst_dir   / img_path.name)
        shutil.copy2(label_path, label_dst_dir / label_path.name)
        src_pairs.append((img_path, label_path))
        copied += 1

    print(f"  원본 복사 완료: {copied}장")

    # ── 증강 생성
    aug_count = 0
    aug_idx   = 0
    needed    = TARGET_COUNT - copied

    # 원본을 랜덤 순서로 반복하면서 증강
    random.shuffle(src_pairs)
    pair_cycle = src_pairs * (needed // len(src_pairs) + 2)  # 충분히 반복

    for (img_path, label_path), (aug_name, aug_fn) in zip(
        pair_cycle, AUG_PIPELINE * (needed // len(AUG_PIPELINE) + 2)
    ):
        if aug_count >= needed:
            break

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        bboxes = read_label(label_path)

        try:
            img_aug, bboxes_aug = aug_fn(img, bboxes)
        except Exception as e:
            print(f"  [경고] 증강 실패 ({aug_name}): {e}")
            continue

        if not bboxes_aug:
            continue

        # 저장
        stem     = f"{label_path.stem}_aug{aug_idx:04d}_{aug_name}"
        out_img  = img_dst_dir   / f"{stem}.jpg"
        out_lbl  = label_dst_dir / f"{stem}.txt"

        cv2.imwrite(str(out_img), img_aug)
        write_label(out_lbl, bboxes_aug)

        aug_count += 1
        aug_idx   += 1

        if aug_count % 50 == 0:
            print(f"  증강 진행: {aug_count}/{needed}장")

    total = copied + aug_count
    print(f"\n  증강 완료!")
    print(f"  원본 {copied}장 + 증강 {aug_count}장 = 총 {total}장")
    return total


def copy_split(split: str):
    """val / test는 증강 없이 그대로 복사"""
    for sub in ["images", "labels"]:
        src = DATASET_ROOT / split / sub
        dst = OUTPUT_ROOT  / split / sub
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    count = len(list((DATASET_ROOT / split / "labels").glob("*.txt")))
    print(f"  [{split}] {count}장 복사 완료")


# ─────────────────────────────────────────────
# 4. data.yaml 생성
# ─────────────────────────────────────────────

def write_yaml():
    yaml_path = OUTPUT_ROOT / "data.yaml"
    content = f"""\
path: {OUTPUT_ROOT.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: 1
names: ['waldo']
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"\n  data.yaml 저장 → {yaml_path}")


# ─────────────────────────────────────────────
# 5. 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Step 4. 데이터 증강")
    print("=" * 55)

    print("\n[train] 증강 처리 중...")
    total_train = augment_train()

    print("\n[val / test] 복사 중...")
    copy_split("val")
    copy_split("test")

    write_yaml()

    print("\n" + "=" * 55)
    print(f"  최종 train: {total_train}장")
    print(f"  결과 위치 : {OUTPUT_ROOT.resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    main()
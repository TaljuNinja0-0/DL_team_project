"""
Step 3. EDA 분석
- Waldo bbox 크기 분포 (얼마나 작은지)
- 이미지 해상도 분포
- bbox 위치 히트맵 (어디에 주로 있는지)
- 분석 결과 → 증강/SAHI 파라미터 결정에 활용

dataset_waldo/ (step1 결과) 기준으로 실행
"""

from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm


# ─────────────────────────────────────────────
# 한글 폰트 설정 (Windows 기준 맑은 고딕)
# ─────────────────────────────────────────────
def set_korean_font():
    import platform
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":  # macOS
        plt.rcParams["font.family"] = "AppleGothic"
    else:  # Linux
        # 나눔고딕 설치 필요: apt-get install fonts-nanum
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

set_korean_font()


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATASET_ROOT = Path("processed_waldo")  # step1 결과 경로 (수정)
EDA_OUTPUT   = Path("eda_output")      # 분석 결과 저장 폴더
SPLITS       = ["train", "val", "test"]
HEATMAP_SIZE = 100                     # 히트맵 해상도 (100×100 grid)


# ─────────────────────────────────────────────
# 1. 데이터 수집
# ─────────────────────────────────────────────
def collect_data(splits):
    """
    모든 split에서 bbox 정보와 이미지 해상도 수집
    반환:
        bbox_w_list   : bbox 너비 (정규화, 0~1)
        bbox_h_list   : bbox 높이 (정규화, 0~1)
        bbox_cx_list  : bbox 중심 x (정규화)
        bbox_cy_list  : bbox 중심 y (정규화)
        bbox_area_list: bbox 면적 (정규화, w*h)
        img_w_list    : 이미지 픽셀 너비
        img_h_list    : 이미지 픽셀 높이
        split_counts  : split별 이미지 수
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    bbox_w_list    = []
    bbox_h_list    = []
    bbox_cx_list   = []
    bbox_cy_list   = []
    bbox_area_list = []
    img_w_list     = []
    img_h_list     = []
    split_counts   = defaultdict(int)

    for split in splits:
        img_dir   = DATASET_ROOT / split / "images"
        label_dir = DATASET_ROOT / split / "labels"

        if not img_dir.exists() or not label_dir.exists():
            continue

        label_files = sorted(label_dir.glob("*.txt"))

        for label_path in label_files:
            # 대응 이미지 찾기
            img_path = None
            for ext in img_exts:
                candidate = img_dir / (label_path.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            # 이미지 해상도
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            img_w_list.append(W)
            img_h_list.append(H)
            split_counts[split] += 1

            # bbox 읽기
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, cx, cy, w, h = parts
                    cx, cy, w, h = float(cx), float(cy), float(w), float(h)

                    bbox_cx_list.append(cx)
                    bbox_cy_list.append(cy)
                    bbox_w_list.append(w)
                    bbox_h_list.append(h)
                    bbox_area_list.append(w * h)

    return (bbox_w_list, bbox_h_list, bbox_cx_list, bbox_cy_list,
            bbox_area_list, img_w_list, img_h_list, split_counts)


# ─────────────────────────────────────────────
# 2. 통계 출력
# ─────────────────────────────────────────────
def print_stats(bbox_w, bbox_h, bbox_area, img_w, img_h, split_counts):
    print("\n" + "=" * 55)
    print("  📊 EDA 통계 요약")
    print("=" * 55)

    print("\n  [Split별 이미지 수]")
    for split, cnt in split_counts.items():
        print(f"    {split:5s}: {cnt}장")

    print(f"\n  [Waldo bbox 크기 - 정규화 기준 (0~1)]")
    print(f"    너비  평균: {np.mean(bbox_w):.4f}  "
          f"중앙: {np.median(bbox_w):.4f}  "
          f"최소: {np.min(bbox_w):.4f}  최대: {np.max(bbox_w):.4f}")
    print(f"    높이  평균: {np.mean(bbox_h):.4f}  "
          f"중앙: {np.median(bbox_h):.4f}  "
          f"최소: {np.min(bbox_h):.4f}  최대: {np.max(bbox_h):.4f}")
    print(f"    면적  평균: {np.mean(bbox_area):.6f}  "
          f"중앙: {np.median(bbox_area):.6f}  "
          f"최소: {np.min(bbox_area):.6f}  최대: {np.max(bbox_area):.6f}")

    # 픽셀 기준 (평균 이미지 크기 기준)
    avg_W = np.mean(img_w)
    avg_H = np.mean(img_h)
    print(f"\n  [Waldo bbox 크기 - 픽셀 기준 (평균 이미지 {avg_W:.0f}×{avg_H:.0f})]")
    print(f"    너비  평균: {np.mean(bbox_w)*avg_W:.1f}px  "
          f"최소: {np.min(bbox_w)*avg_W:.1f}px")
    print(f"    높이  평균: {np.mean(bbox_h)*avg_H:.1f}px  "
          f"최소: {np.min(bbox_h)*avg_H:.1f}px")

    # 작은 객체 비율 (COCO 기준: 32×32px 이하)
    small_thresh = 32
    small_count = sum(
        1 for w, h in zip(bbox_w, bbox_h)
        if w * avg_W < small_thresh or h * avg_H < small_thresh
    )
    print(f"\n  [소형 객체 비율]")
    print(f"    {small_thresh}px 미만 bbox: {small_count}/{len(bbox_w)}개 "
          f"({small_count/len(bbox_w)*100:.1f}%)")

    sahi_thresh = 64
    sahi_count = sum(
        1 for w, h in zip(bbox_w, bbox_h)
        if w * avg_W < sahi_thresh or h * avg_H < sahi_thresh
    )
    print(f"    {sahi_thresh}px 미만 bbox: {sahi_count}/{len(bbox_w)}개 "
          f"({sahi_count/len(bbox_w)*100:.1f}%)  ← SAHI 필요성 판단 기준")

    print(f"\n  [이미지 해상도]")
    print(f"    너비  평균: {avg_W:.0f}px  "
          f"최소: {np.min(img_w)}px  최대: {np.max(img_w)}px")
    print(f"    높이  평균: {avg_H:.0f}px  "
          f"최소: {np.min(img_h)}px  최대: {np.max(img_h)}px")
    print("=" * 55)


# ─────────────────────────────────────────────
# 3. 시각화
# ─────────────────────────────────────────────
def plot_eda(bbox_w, bbox_h, bbox_cx, bbox_cy, bbox_area, img_w, img_h):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Waldo Detection - EDA 분석", fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. bbox 너비 분포
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(bbox_w, bins=30, color="#4C72B0", edgecolor="white")
    ax1.axvline(np.mean(bbox_w), color="red", linestyle="--", label=f"평균 {np.mean(bbox_w):.3f}")
    ax1.axvline(np.median(bbox_w), color="orange", linestyle="--", label=f"중앙 {np.median(bbox_w):.3f}")
    ax1.set_title("bbox 너비 분포 (정규화)")
    ax1.set_xlabel("너비 (0~1)")
    ax1.set_ylabel("빈도")
    ax1.legend(fontsize=8)

    # ── 2. bbox 높이 분포
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(bbox_h, bins=30, color="#55A868", edgecolor="white")
    ax2.axvline(np.mean(bbox_h), color="red", linestyle="--", label=f"평균 {np.mean(bbox_h):.3f}")
    ax2.axvline(np.median(bbox_h), color="orange", linestyle="--", label=f"중앙 {np.median(bbox_h):.3f}")
    ax2.set_title("bbox 높이 분포 (정규화)")
    ax2.set_xlabel("높이 (0~1)")
    ax2.set_ylabel("빈도")
    ax2.legend(fontsize=8)

    # ── 3. bbox 면적 분포
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(bbox_area, bins=30, color="#C44E52", edgecolor="white")
    ax3.axvline(np.mean(bbox_area), color="blue", linestyle="--", label=f"평균 {np.mean(bbox_area):.5f}")
    ax3.set_title("bbox 면적 분포 (w×h, 정규화)")
    ax3.set_xlabel("면적")
    ax3.set_ylabel("빈도")
    ax3.legend(fontsize=8)

    # ── 4. bbox 위치 산점도
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(bbox_cx, bbox_cy, alpha=0.4, s=15, color="#8172B2")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.invert_yaxis()
    ax4.set_title("Waldo 중심 위치 산점도")
    ax4.set_xlabel("중심 x")
    ax4.set_ylabel("중심 y")

    # ── 5. bbox 위치 히트맵
    ax5 = fig.add_subplot(gs[1, 1])
    heatmap, xedges, yedges = np.histogram2d(
        bbox_cx, bbox_cy,
        bins=HEATMAP_SIZE,
        range=[[0, 1], [0, 1]]
    )
    # 가우시안 블러로 부드럽게
    from scipy.ndimage import gaussian_filter
    heatmap_smooth = gaussian_filter(heatmap.T, sigma=3)
    im = ax5.imshow(
        heatmap_smooth,
        origin="upper",
        extent=[0, 1, 1, 0],
        aspect="auto",
        cmap="hot"
    )
    plt.colorbar(im, ax=ax5, shrink=0.8)
    ax5.set_title("Waldo 위치 히트맵")
    ax5.set_xlabel("중심 x")
    ax5.set_ylabel("중심 y")

    # ── 6. bbox 너비 vs 높이 산점도
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(bbox_w, bbox_h, alpha=0.4, s=15, color="#CCB974")
    ax6.set_xlabel("bbox 너비")
    ax6.set_ylabel("bbox 높이")
    ax6.set_title("bbox 너비 vs 높이")

    # ── 7. 이미지 너비 분포
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(img_w, bins=20, color="#64B5CD", edgecolor="white")
    ax7.axvline(np.mean(img_w), color="red", linestyle="--", label=f"평균 {np.mean(img_w):.0f}px")
    ax7.set_title("이미지 너비 분포")
    ax7.set_xlabel("픽셀 너비")
    ax7.set_ylabel("빈도")
    ax7.legend(fontsize=8)

    # ── 8. 이미지 높이 분포
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(img_h, bins=20, color="#4C72B0", edgecolor="white")
    ax8.axvline(np.mean(img_h), color="red", linestyle="--", label=f"평균 {np.mean(img_h):.0f}px")
    ax8.set_title("이미지 높이 분포")
    ax8.set_xlabel("픽셀 높이")
    ax8.set_ylabel("빈도")
    ax8.legend(fontsize=8)

    # ── 9. bbox 픽셀 크기 분포 (평균 해상도 기준)
    ax9 = fig.add_subplot(gs[2, 2])
    avg_W = np.mean(img_w)
    avg_H = np.mean(img_h)
    bbox_px_w = [w * avg_W for w in bbox_w]
    bbox_px_h = [h * avg_H for h in bbox_h]
    ax9.scatter(bbox_px_w, bbox_px_h, alpha=0.4, s=15, color="#C44E52")
    ax9.axvline(32, color="orange", linestyle="--", label="32px (소형 기준)")
    ax9.axhline(32, color="orange", linestyle="--")
    ax9.axvline(64, color="red", linestyle="--", label="64px (SAHI 기준)")
    ax9.axhline(64, color="red", linestyle="--")
    ax9.set_xlabel("bbox 너비 (px)")
    ax9.set_ylabel("bbox 높이 (px)")
    ax9.set_title(f"bbox 픽셀 크기 (평균 {avg_W:.0f}×{avg_H:.0f} 기준)")
    ax9.legend(fontsize=8)

    EDA_OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = EDA_OUTPUT / "eda_analysis.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 EDA 차트 저장 → {out_path}")


# ─────────────────────────────────────────────
# 4. SAHI 파라미터 추천
# ─────────────────────────────────────────────
def recommend_sahi_params(bbox_w, bbox_h, img_w, img_h):
    avg_W = np.mean(img_w)
    avg_H = np.mean(img_h)
    avg_bbox_px = np.mean([w * avg_W for w in bbox_w])

    print("\n" + "=" * 55)
    print("  💡 SAHI 파라미터 추천 (학습 담당과 협의용)")
    print("=" * 55)

    # 슬라이스 크기 추천
    if avg_bbox_px < 20:
        slice_size = 320
        overlap    = 0.3
        note = "Waldo 매우 작음 → 작은 슬라이스 + 높은 overlap 권장"
    elif avg_bbox_px < 50:
        slice_size = 512
        overlap    = 0.25
        note = "Waldo 소형 → 중간 슬라이스 권장"
    else:
        slice_size = 640
        overlap    = 0.2
        note = "표준 슬라이스 적용 가능"

    print(f"\n  Waldo 평균 픽셀 크기: {avg_bbox_px:.1f}px")
    print(f"  → {note}")
    print(f"\n  추천 설정:")
    print(f"    slice_height     = {slice_size}")
    print(f"    slice_width      = {slice_size}")
    print(f"    overlap_height_ratio = {overlap}")
    print(f"    overlap_width_ratio  = {overlap}")
    print("=" * 55)


# ─────────────────────────────────────────────
# 5. 메인
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Step 3. EDA 분석 시작")
    print("=" * 55)

    (bbox_w, bbox_h, bbox_cx, bbox_cy,
     bbox_area, img_w, img_h, split_counts) = collect_data(SPLITS)

    if not bbox_w:
        print("  ❌ bbox 데이터 없음. DATASET_ROOT 경로 확인하세요.")
        return

    print(f"\n  수집 완료: bbox {len(bbox_w)}개 / 이미지 {len(img_w)}장")

    # 통계 출력
    print_stats(bbox_w, bbox_h, bbox_area, img_w, img_h, split_counts)

    # 시각화
    print("\n  차트 생성 중...")
    try:
        from scipy.ndimage import gaussian_filter
        plot_eda(bbox_w, bbox_h, bbox_cx, bbox_cy, bbox_area, img_w, img_h)
    except ImportError:
        print("  ⚠️  scipy 없음 → pip install scipy 후 재실행")

    # SAHI 파라미터 추천
    recommend_sahi_params(bbox_w, bbox_h, img_w, img_h)


if __name__ == "__main__":
    main()
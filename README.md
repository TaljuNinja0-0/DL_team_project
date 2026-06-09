# 딥러닝 실습 2조 - 월리를 찾아라 데이터 전처리

---

## 폴더 구조

```
📦 프로젝트 루트
├── 📂 dataset/              # 원본 데이터셋 (5개 클래스)
├── 📂 processed/            # 원본 증강 데이터셋 (train/val/test 분할 완료)
│
├── 📂 processed_waldo/      # Waldo 단일 클래스 필터링 결과 (Step 1 출력)
├── 📂 inspection_viz/       # 데이터 검수용 bbox 시각화 결과 (Step 2 출력)
├── 📂 eda_output/           # EDA 분석 차트 결과 (Step 3 출력)
└── 📂 augmented_waldo/      # 최종 증강 적용 데이터셋 (Step 4 출력)
```

---

## 코드 설명

| 파일 | 설명 |
|------|------|
| `waldo_step1.py` | Waldo 클래스 필터링 및 재라벨링 |
| `waldo_step2.py` | bbox 검수, 클리핑, 시각화 |
| `waldo_step3.py` | EDA 분석 |
| `waldo_step4.py` | 데이터 증강 |

---


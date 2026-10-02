# Data

MIT-Stanford Battery Dataset (Severson et al., *Nature Energy* 2019) — Kaggle errorcorrect 버전 (MATLAB v7.3 = HDF5)

| 원본 파일 | Batch | 용도 | 원본 셀 | 사용 셀 |
|---|---|---|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 | 학습 (정책 단위 CV + hold-out) | 46 | 36 |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 | 테스트 (필수) | 47 | 39 |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 | 추가 검증 | 46 | 44 |

- 다운로드 : https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle
- 원본(약 8GB)은 git에 올리지 않는다. 원하는 폴더에 두고 `MAT_DIR=<폴더> python load_data.py` 로 캐시(`data/cells_raw.pkl`, git 제외)를 만든다 (기본 `~/Downloads/archive`)

## 이 폴더의 파일
| 파일 | 만드는 곳 | 내용 |
|---|---|---|
| `cells_raw.pkl` (git 제외) | `load_data.py` | 셀별 summary 전체 + cycle 1~120 Qdlin |
| `data_cleaning_log.csv` | 01_EDA | 셀별 제거 여부와 사유 |
| `features_cycle100.csv` | 01_EDA, `src/features.py` | 셀당 1행, cycle 2~100 피처 + 수명 |
| `features_sensitivity_b1c0-4.csv` | 01_EDA | b1 c0~c4 (원논문 보정 수명) 민감도 분석용 |
| `eda_stats.json` | 01_EDA | 장표에 쓴 EDA 수치 |
| `model_input.csv`, `feature_sets.json`, `splits.json` | 02_feature_engineering | 모델 입력, 피처셋 A/B/C, 정책 단위 분할 20개 |
| `model_runs_20seeds.csv`, `model_report.json`, `test_predictions.csv` | 03_modeling | 420회 학습 결과, 최종 리포트, 테스트 예측 |

## 정제 규칙
- 인덱스 : summary·Qdlin 배열 j = cycle − 1 (cycle 10 = j9, cycle 100 = j99)
- 제거 : 다른 실험 프로토콜(b2 VarCharge 4, SLOWCYCLE 4), EOL(0.88Ah) 미도달 = 수명 미관측(b1 10, b3 2)
- b1 c0~c4 : barcode 대조 결과 b2에 이어진 셀이 없어 원논문 보정 수명을 검증할 수 없음 → 제외, 민감도 분석에만 사용
- b3 원논문 노이즈 채널(c2, c37, c42, c43) : 스파이크 점검(30mAh 초과)에 걸리지 않아 유지, 제거한 경우는 민감도 분석
- 단발성 QD 스파이크는 median filter(k=5), IR = 0은 결측

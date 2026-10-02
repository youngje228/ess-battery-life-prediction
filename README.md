# ESS 배터리 수명 예측 (초기 100 사이클 → cycle life)

MIT-Stanford 배터리 데이터(Severson et al., *Nature Energy* 2019)로 **초기 100 사이클 데이터만 보고 셀의 총 수명(EOL까지 사이클 수)을 예측**한다.
DS Mini Project · 울산 2반 · 김영제, 김지훈

| 항목 | 내용 |
|---|---|
| Task | Regression |
| Target | `log10(cycle_life)` · cycle_life = 방전 용량 0.88Ah(SOH 80%) 도달 사이클, 예측 후 역변환해 MAPE 계산 |
| 데이터 분할 | Train / Valid = Batch 1 (정책 단위 분할) · Test = Batch 2 (필수), Batch 3 (추가) |
| 지표 | MAPE (%) |
| 원논문 성능 (Target) | Regression MAPE 9.1% |

## 1. 성능 요약

**최종 모델** : 피처셋 A (`dQ_logvar` 1개) · Linear Regression (StandardScaler → LinearRegression)

| Index | MAPE (%) | 비고 |
|---|---|---|
| Train (Batch 1 CV) | **8.83** ± 0.81 | hold-out을 뺀 Batch 1에서 정책 단위 GroupKFold(5), 20 seed 평균 ± 표준편차 (seed 0 : 9.31) |
| Valid (Batch 1 Hold-out) | **9.63** ± 2.55 | 정책 단위 hold-out 20% (6~8셀), 20 seed 평균 ± 표준편차 (seed 0 : 7.80) |
| Test (Batch 2) | **28.56** | Batch 1 전체(36셀)로 재학습한 최종 모델 |
| Gap (Train − Valid) | +0.80 | 과적합 거의 없음 |
| Gap (Valid − Test) | +18.93 | 배치 간 일반화 실패 (아래 3장) |
| Gap (Target − Test) | +19.46 | 원논문 9.1% 대비 (아래 3장) |

Gap 부호 : 양수 = 뒤 단계에서 오차 증가. Valid는 모델 선택에 쓰였으므로 일반화 성능은 Test로 판단한다.

**Batch 3 (추가 검증)**

| Index | MAPE (%) |
|---|---|
| Test (Batch 3) | **12.81** |
| Gap (Batch 2 − Batch 3) | −15.75 (Batch 3가 Batch 2보다 오차가 작다) |

## 2. 무엇을 했나

### 데이터 정제 (`notebooks/01_EDA.ipynb`)
- 셀 수 : 46 / 47 / 46 → **36 / 39 / 44** (수명이 관측된 셀만 사용)
- 제거 : b1 EOL 미도달 10셀, b2 다른 실험 프로토콜(VarCharge, SLOWCYCLE) 8셀, b3 EOL 미도달 2셀
- b1 c0~c4 : 원논문은 b2로 실험이 이어졌다고 보고 수명을 보정했지만, 이번 데이터 버전의 b2에는 이어진 셀이 없다 (barcode 대조). 검증할 수 없어 학습에서 제외하고 민감도 분석으로만 사용
- QD 단발 스파이크는 셀 제거 대신 median filter(k=5), IR = 0은 결측 처리

### 피처 (`notebooks/02_feature_engineering.ipynb`)
cycle 2~100 측정값만 사용한다. EOL까지의 곡선으로 계산한 값(knee 등)은 미래 정보라 제외했다.

| 피처셋 | 피처 | 역할 |
|---|---|---|
| A | `dQ_logvar` = log10 Var(Q100(V) − Q10(V)) | 원논문 variance 모델 |
| B | A + `QD_slope_91_100` | DAY 1 기준(배치 간 부호 일관, 최소 \|r\| ≥ 0.2, 공선성 그룹 대표)을 통과한 피처 |
| C | B + dQ_logmin, dQ_mean, QD_2, QD_slope_2_100, QD_icpt_2_100 | 배치마다 부호가 뒤집히는 피처 포함 · 가설 검증용, 선택 대상에서 사전 제외 |

### 모델 학습과 선택 (`notebooks/03_modeling.ipynb`)
- 후보 7종 : Linear, Ridge, Lasso, ElasticNet, Gaussian Process, SVR(RBF), RandomForest(depth 2~3)
- 모든 모델은 `StandardScaler → 모델` Pipeline이라 CV fold마다 학습 fold로만 표준화한다
- 피처셋 3 × 모델 7 × 분할 20회 = 420회 학습, 하이퍼파라미터는 Train 안에서 정책 단위 GroupKFold(5)로 탐색
- **선택 규칙** (Test는 쓰지 않는다)
  1. 기준 = (Train CV MAPE + Valid MAPE) / 2의 20 seed 평균
  2. 1위와의 차이가 1위 기준값의 표준편차(0.93)보다 작은 후보 중 피처 수가 적고 단순한 모델
  3. 결과 : 1위가 셋 A · Linear(9.23 ± 0.93)이고 오차 범위 안 후보 중에서도 가장 단순하므로 그대로 선택. 셋 B 최고(Lasso 9.58)는 차이가 범위 안이지만 피처가 더 많다

## 3. 결과 해석

### 3-1. Batch 2 오차는 일반 구조 셀의 일괄 과대예측이다

| 그룹 | 셀 | MAPE (%) | 평균 부호 오차 (%) |
|---|---|---|---|
| b2 일반 구조 | 30 | 31.9 | **+31.9** |
| b2 newstructure | 9 | 17.5 | +15.7 |
| b3 (전부 newstructure) | 44 | 12.8 | +1.6 |

- b2 일반 셀은 **30셀 전부**를 실제보다 길게 예측한다 (최소 +8.6%). 무작위 오차가 아니라 **수준(절편) 차이**다
- DAY 1 EDA에서 이미 본 현상이다 : 배치별로 따로 맞춘 dQ_logvar–수명 직선은 기울기가 같지만(−0.30, −0.33, −0.31) b2만 아래로 이동해 같은 dQ_logvar에서 수명이 약 22% 짧다
- 반대로 b3는 bias가 +1.6%로 거의 없다. **ΔQ 신호의 기울기는 배치를 넘어 유지되고, 일반화를 깨는 것은 배치 수준 차이**라는 뜻이다

### 3-2. 원논문 9.1%와의 Gap (+19.46)

| 원인 | 근거 |
|---|---|
| 분할 방식 | 원논문은 b1과 b2를 섞어 번갈아 학습 / 테스트로 나눠 b2의 수준 차이가 학습에 들어갔다. 이번 과제는 배치 단위로 나눈다 |
| 외삽 | b2 수명의 76.9%가 학습 최소(534) 미만, b2 dQ_logvar의 44%가 학습 범위 밖 |
| 셀 구성 | 원논문 학습 41 / 1차 테스트 43셀, 이번 36 / 39셀 (b1 c0~c4 제외, b2 데이터 버전 차이) |

### 3-3. 셋 C 가설 : 배치마다 뒤집히는 피처는 테스트에서 무너진다

| 피처셋 | 최적 모델 | Batch 1 선택 기준 | Test b2 | Test b3 |
|---|---|---|---|---|
| A | Linear | 9.23 | 28.56 | 12.81 |
| B | Lasso | 9.58 | 32.07 | 12.40 |
| C | Linear | **6.40** | **101.96** | 58.40 |

셋 C는 Batch 1 안에서 가장 좋지만 Batch 2에서는 오차가 100%를 넘는다. DAY 1에서 "b1에서 |r| > 0.5인 피처 8개 중 4개가 b2에서 부호 반전"이라고 본 위험이 그대로 나타났고, 셋 C를 선택 대상에서 미리 뺀 규칙이 옳았다.

### 3-4. 민감도와 불확실성

| 학습 데이터 | 셀 | 수명 범위 | Test b2 | Test b3 |
|---|---|---|---|---|
| 기본 (c0~c4 제외) | 36 | 534~1,074 | 28.56 | 12.81 |
| c0~c4 원논문 보정 수명 포함 | 41 | 534~2,237 | 26.17 | 15.26 |

- 보정 수명을 넣어도 b2는 2.4%p만 좋아진다. b2 오차의 주원인은 학습 수명 범위가 아니라 배치 수준 차이다
- GPR 95% 예측 구간은 b2 셀의 41%, b3 셀의 84%만 덮는다. 배치가 바뀌면 모델이 자기 불확실성을 과소평가한다

### 3-5. ESS 관점
- 같은 셀 제품이라도 **생산 로트와 운전 구조가 바뀌면 같은 초기 신호에서 수명이 약 20~30% 달라진다.** 한 사이트에서 학습한 모델을 다른 사이트에 그대로 쓰면 교체 시점을 늦게 잡는 쪽(과대예측)으로 틀릴 수 있다
- 과대예측은 교체 지연 → 돌발 정지와 피크 대응 실패로 이어지는 위험한 방향이다. 운영에서는 새 로트·사이트마다 **소수 셀의 초기 데이터로 절편을 재보정**하는 절차가 필요하다
- 그래도 ΔQ 신호의 방향과 기울기는 세 배치 모두에서 유지되므로, 같은 로트 안에서의 **상대적 순위(어느 셀이 먼저 교체 대상인가)** 는 신뢰할 수 있다

### 3-6. 한계
- 학습 셀이 36개, 정책 20개뿐이라 Valid(6~8셀)의 분산이 크다 (Valid MAPE 표준편차 2.55)
- Valid MAPE는 모델 선택에 쓰였으므로 약간 낙관적이다
- 학습 배치(b1)에 newstructure 셀과 b2·b3 로트가 없어 배치 효과를 학습할 수 없다. 구조·로트 정보는 피처로 쓸 수 없는 조건이다
- b1 c0~c4의 실제 수명은 이 데이터로 검증할 수 없다
- b3는 ΔQ 곡선 시작점 왜곡이 있고, 가장 오래 산 2셀은 EOL 미도달로 제외되어 장수 쪽 평가가 불완전하다

## 4. 실행 방법

```bash
git clone https://github.com/youngje228/ess-battery-life-prediction.git
cd ess-battery-life-prediction
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
MAT_DIR=<.mat 폴더> python load_data.py      # → data/cells_raw.pkl (기본 ~/Downloads/archive)
```
그다음 `notebooks/`의 노트북을 01 → 02 → 03 순서로 실행한다 (작업 폴더 = `notebooks/`).

원본 데이터 : `2017-05-12`, `2018-02-20`, `2018-04-12_batchdata_updated_struct_errorcorrect.mat`

## 5. 폴더 구조

```
├── load_data.py                    # .mat(HDF5) → data/cells_raw.pkl
├── notebooks/
│   ├── 01_EDA.ipynb                # 정제, EDA Q1~Q5, 피처 테이블
│   ├── 02_feature_engineering.ipynb  # 피처셋 A/B/C, 정책 단위 분할 20개
│   └── 03_modeling.ipynb           # 420회 학습, 선택, Test, 잔차·가설·민감도
├── data/                           # features_cycle100.csv, model_input.csv, splits.json,
│                                   # model_runs_20seeds.csv, model_report.json, test_predictions.csv
├── figures/eda/, figures/model/
└── reports/                        # DAY 1 모델 전략 PDF
```

## 참고문헌
Severson, K. A. et al. Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy* 4, 383–391 (2019)

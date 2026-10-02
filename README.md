# ESS 배터리 수명 예측

**목적**: 배터리를 **처음 100사이클만** 돌려 보고, 그 셀이 언제 수명(방전 용량 80%)을 다할지 미리 예측한다. 학습 배치(Batch1)에서 만든 모델이 실험 시기가 다른 배치(Batch2, Batch3)에서도 통하는지 평가하고, 통하지 않는 곳과 이유를 정리한다.

## 프로젝트 개요

- **데이터셋**: MIT-Stanford Battery Dataset (Severson et al., *Nature Energy* 2019)
- **학습 데이터**: Batch 1 (2017-05-12), 46셀 중 36셀 사용 (80% 미도달 10셀 제외)
- **평가 데이터**: Batch 2 (2018-02-20), 47셀 중 39셀 사용 (cycle_life NaN 8셀 제외)
- **추가 평가**: Batch 3 (2018-04-12), 46셀 중 44셀 사용 (cycle_life NaN 2셀 제외)
- **태스크**: Regression. y = log10(cycle_life), 예측값을 10^ŷ로 되돌린 뒤 MAPE(%)로 평가, 목표 = 원논문 9.1%

## 파일 구조

```
├── data/
│   └── README.md                     # 원본 .mat 받는 곳, 캐시 만드는 법 (.mat과 캐시는 저장소에 없음)
├── notebooks/
│   ├── 01_EDA.ipynb                  # 데이터 로드, 제외 규칙, Q1~Q5
│   ├── 02_feature_engineering.ipynb  # 피처 표, 제외 규칙, 누수 점검
│   └── 03_modeling.ipynb             # 분할, 모델 비교, 최종 평가, 오류 분석
├── src/
│   ├── preprocess.py                 # 셀 품질 플래그 → data/cache/cell_flags.csv
│   ├── features.py                   # 피처 표 + 제외 규칙
│   ├── train.py                      # 학습/평가 파이프라인
│   ├── load_data.py                  # (보조) .mat(HDF5) → data/cache/*.pkl
│   ├── eda_utils.py                  # (보조) EDA/피처 계산 함수 (ΔQ, 실측 충전 전류 등)
│   └── plots.py                      # (보조) 모델링 그림 M1~M3
├── results/
│   ├── model_performance.csv         # 과제 표준 성능 표 (.md 사본 포함)
│   ├── model_comparison.csv          # 모델 × 피처 단계 비교
│   ├── sensitivity_protrusion.csv    # ΔQ 양수 셀 민감도
│   ├── predictions_holdout.csv, predictions_batch2.csv, predictions_batch3.csv
│   ├── eda_insights.md               # EDA 요약
│   └── figures/                      # EDA 그림 Q1~Q5, 모델링 그림 M1~M3 (.png)
├── .gitignore
├── requirements.txt
└── README.md
```

## 환경 설정

```bash
git clone <저장소 주소>
cd <저장소 폴더>
pip install -r requirements.txt

# 1) data/README.md를 보고 원본 .mat 3개를 data/에 둔다 (data.matr.io)
# 2) 캐시 만들기 (프로젝트 루트에서)
python -m src.load_data      # data/cache/b1~b3.pkl
python -m src.preprocess     # data/cache/cell_flags.csv
# 3) 학습/평가 → results/ 에 표와 그림 저장
python -m src.train
```

## EDA

- **Cycle Life 분포**
  - 중앙값 Batch1 772.5 / Batch2 472 / Batch3 1,005.5. 단수명(<500) 0% / 71.8% / 0%. Batch2는 기존 구조(392–514)와 newstructure(777–1,186) 두 무리. 통합 왜도 1.04 → log10 후 0.03.
  - 핵심 발견 : 배치마다 수명 수준이 크게 달라 배치 간 일반화가 가장 큰 위험이다 → 타깃 log10(cycle_life).
- **열화 곡선 분석**
  - knee point는 수명의 75–80% 지점이고 그 뒤 감소가 약 10배 빨라진다. 초기 100사이클 QD 변화는 +1.4–2.4mAh로 거의 없고 수명과의 ρ는 +0.42 / −0.10 / −0.14.
  - 핵심 발견 : 초기 용량 추세로는 수명을 알기 어렵고 knee는 미래 정보다 → 둘 다 피처로 쓰지 않는다.
- **ΔQ(V) 곡선 분석** (ΔQ = Q₁₀₀(V) − Q₁₀(V))
  - `dQ_log_var`(ΔQ 분산의 log10)와 log 수명의 r은 −0.844 / −0.918 / −0.762. Batch2 값은 상당 부분 두 무리 간 차이에서 나온다(무리 안 r −0.38 / −0.27).
  - 핵심 발견 : 용량이 거의 안 변한 시점에도 곡선 모양에 수명 신호가 있고, 세 배치에서 방향이 같다.
- **충전 속도(C-rate)와 수명의 관계**
  - Batch1 프로토콜별 평균 수명은 1,074–546으로 벌어지지만, 실측 평균 C-rate vs 수명 ρ는 −0.42 / +0.54 / −0.21로 부호가 바뀐다. Batch3 프로토콜 중 Batch1에 있는 것은 0/44.
  - 핵심 발견 : "고속 충전 = 단수명"은 Batch1에서만 부분적으로 맞다 → 프로토콜 이름은 쓰지 않고, 분할은 프로토콜 단위로 한다.
- **추가 확인한 내용**
  - 중복: ΔQ 계열끼리 r 0.98–1.00, chargetime과 `avgC_0to80_meas`는 r −0.99.
  - 데이터 품질: Batch1 10셀은 80% 도달 전 시험 종료(검열값), Batch2 8셀과 Batch3 2셀은 cycle_life NaN, Batch2 c41–c46은 IR이 전 구간 0.
  - **ΔQ 양수 셀**: max ΔQ가 +2mAh를 넘는 셀(원인 미확인). 모델링 셀 기준 Batch2 4셀, Batch3 8셀.
  - 핵심 발견 : 정답을 믿을 수 없는 셀은 빼고(→ 36 / 39 / 44셀), IR과 중복 피처는 빼며, ΔQ 양수 셀은 평가에 포함하고 민감도만 따로 본다.

## Modeling

### 피처 엔지니어링 전략

모든 피처는 cycle 100 이하 데이터로만 만든다 (cycle_life, n_cycles, knee는 읽지 않는다).

| 구분 | 피처 | 뜻 | DAY 2 결과 |
|---|---|---|---|
| 핵심 | `dQ_log_var` | ΔQ(V) 분산의 log10 (세 배치 모두 강한 음의 상관) | **채택** |
| 보조 | `avgC_0to80_meas` | cycle 10 실측 평균 충전 속도 (SOC 0→80%) | Batch1 CV 악화 → 탈락 |
| 후보 | `Tavg_mean_2_100` | cycle 2–100 평균 온도 (수명 상관 r −0.16 / +0.39 / −0.01) | Batch1 CV 악화 → 탈락 |

- **제외한 피처**: 절대 용량(QD_c2, 원값 Qdlin: 배치를 섞으면 가짜 상관), IR(Batch2 일부 0), 배치마다 부호가 바뀌는 피처(QD 추세, dQ_skew/kurt, chargetime), `dQ_log_var`와 중복(dQ_log_abs_min/mean), 프로토콜 one-hot, n_cycles와 knee(정답 누수).
- 피처 구성은 DAY 1에서 세 배치의 상관을 함께 보고 정했으므로 Batch2 MAPE는 다소 낙관적일 수 있다.

### 모델 선택 및 근거

- **후보 모델**: Dummy(학습 셀 log 수명 평균, 기준선), ElasticNet, Ridge, RandomForest(비선형 비교군). 모두 `StandardScaler + 모델` Pipeline.
- **최종 모델**: **Ridge + `dQ_log_var` 단독** (alpha = 0.01)
- **선택 이유**:
  - Batch1 CV MAPE만으로 골랐다: Ridge 9.27%가 가장 낮았다 (ElasticNet 9.34%, RandomForest 10.06%, Dummy 14.72%).
  - `avgC_0to80_meas`, `Tavg_mean_2_100`를 더하면 세 모델 모두 CV MAPE가 올라갔다 (Ridge 9.27 → 9.85 / 9.79). 전체 비교는 `results/model_comparison.csv`.
  - 피처 1개의 정규화 선형모델이 가장 단순하고, 학습 범위 밖으로도 값을 낸다.
- **검증 설계 (누수 방지)**:
  - Valid: Batch1 36셀을 프로토콜 단위로 hold-out (7셀 / 4프로토콜, `GroupShuffleSplit`, seed 42). 나머지 29셀 / 16프로토콜이 학습용.
  - Train(CV): 29셀에서 중첩 GroupKFold (바깥 4-fold 점수, 안쪽 3-fold 하이퍼파라미터 격자 탐색). 하이퍼파라미터는 이 Batch1 학습 셀 안의 탐색으로만 정했고, 스케일러는 Pipeline 안에서 학습 부분에만 fit.
  - Batch2와 Batch3는 선택이 끝난 뒤 **한 번만** 평가했고 선택에 쓰지 않았다 (model_comparison의 Batch2 열은 기록용).

## 성능 결과

**Gap 규칙**: Gap = 뒤 항목 MAPE − 앞 항목 MAPE (행 이름의 앞−뒤 순서와 반대), (+) = 오차 증가. 단위 %p.

| 구분 | MAPE (%) | 비고 |
| :--- | :---: | :--- |
| **Train (Batch 1 CV)** | 9.27 | 프로토콜 GroupKFold 4-fold 평균 (중첩 CV) |
| **Valid (Batch 1 Hold-out)** | 8.47 | 프로토콜 단위 hold-out 7셀 / 4프로토콜 |
| **Test (Batch 2)** | 29.6 | 39셀 (ΔQ 양수 셀 포함) |
| **Gap (Train-Valid)** | −0.80 | Valid − Train. (+) : 과적합 의심 |
| **Gap (Valid-Test)** | +21.14 | Test − Valid. (+) : 배치 간 일반화 저하 의심 |
| **Gap (Target-Test)** | +20.50 | Test − Target. Target : 원논문 **9.1%** |

**Batch3 추가 검증**

| 과제 구분 | 평가 구분 | 수치 결과 | 비고 |
| :--- | :--- | :---: | :--- |
| **Regression (MAPE)** | **Test (Batch 3)** | 12.22 | 44셀 (ΔQ 양수 셀 포함) |
| | **Gap (Batch 2 - Batch 3)** | −17.39 | Batch3 − Batch2. Batch3 오차가 더 작음 |
| | **Gap (Target - Test Batch 3)** | +3.12 | Test(Batch3) − 9.1 |

- **목표 9.1%는 Batch2에서 달성하지 못했다 (29.6%).** Batch3(12.22%)도 목표보다 3.12%p 높다.
- Valid는 7셀이라 잡음이 크다 (학습 29셀에서 프로토콜 하나씩 빼고 잰 MAPE가 1.56%–26.35%). Gap(Train-Valid) −0.80으로 과적합 여부를 단정하지 않는다.
- ΔQ 양수 셀 민감도 (확인용): 빼면 Batch2 29.60 → 30.74% (4셀 제외), Batch3 12.22 → 10.77% (8셀 제외). Batch2의 큰 오차를 설명하지 못한다.

## 오류 분석

학습 29셀의 수명 범위는 **534–1,054**다. 그림: `results/figures/M1_pred_vs_actual.png`, `M2_error_by_cell.png`.

**Batch2 (39셀, MAPE 29.6%)**

| 셀 무리 | 셀 수 | 실제 수명 | MAPE (%) | 방향 |
|---|---|---|---|---|
| 기존 구조 (모두 학습 범위 아래) | 30 | 392–514 | 33.88 | 30셀 모두 과대예측 |
| newstructure, 범위 안 | 7 | 777–1,029 | 18.01 | 7셀 중 6셀 과대예측 |
| newstructure, 범위 위 | 2 | 1,140–1,186 | 6.07 | 2셀 모두 과소예측 |

**모델이 가장 크게 틀린 셀의 공통점**

- Batch2 기존 구조 30셀은 모두 학습 최솟값 534보다 짧고, 모두 과대예측이며, Batch2 전체 절대 오차의 **약 88%**가 여기서 나온다. 가장 크게 틀린 5셀(b2c6, b2c15, b2c18, b2c2, b2c19)도 모두 이 무리다 (+44% – +66% 과대예측).
- newstructure 9셀은 MAPE 15.36%. 범위 위 2셀은 모두 과소예측, 범위 안 7셀 중 6셀은 과대예측이다. 전체 39셀 중 36셀이 과대예측이다.
- `dQ_log_var`와 log 수명의 r이 학습 셀 −0.81에서 Batch2 기존 구조 30셀 안 −0.375로 약해진다.
- Batch3 (44셀, MAPE 12.22%): 모두 newstructure, 학습 범위 아래 셀 없음. 범위 안 27셀 MAPE 9.28%, 범위 위 17셀 16.88%(13셀 과소예측). 가장 크게 틀린 5셀 중 4셀이 수명 1,642–1,935의 범위 위 셀.
- 진단 전용: Batch2 정답을 보고 고른 상수 log 보정 하나를 빼면 MAPE가 약 10.2%까지 내려간다. 오차의 상당 부분이 한 방향 편향이라는 지표일 뿐, 정답을 써서 만든 값이라 **성능 결과가 아니다**.

**원인 가설 및 개선 방향**

- 원인 가설 (모두 확인되지 않음)
  1. 셀 구조 차이: 기존 구조 Batch2 셀은 같은 ΔQ 분산에서 학습 셀보다 수명이 짧다. 구조나 시험 시기 변경이 관계를 바꿨을 수 있다.
  2. 학습 범위 잘림: 80% 미도달 Batch1 10셀을 빼서 범위가 좁아졌고, 범위 밖 셀(Batch2 아래 30셀, Batch3 위 17셀)은 외삽이 된다.
  3. 평균으로의 회귀: 관계가 약해지면 예측이 학습 평균 쪽으로 몰린다 (Batch2 과대예측, Batch3 범위 위 과소예측과 같은 방향).
- 개선 방향 (테스트 결과를 보고 고치면 누수라 이번에는 적용하지 않음): 셀 구조/배치 정보 반영 또는 배치 보정, 검열 셀을 살리는 생존분석형 타깃, 더 넓은 수명 범위의 학습 셀.

## ESS 도메인 해석

**활용 가능한 의사결정**

- 초기 100사이클 데이터만으로 수명이 짧을 셀을 **선별**하고 **교체 시점을 사전에 계획**한다. 과제 문서 자료에 따르면 교체 비용은 ESS CAPEX의 30–40% (1MWh 기준 셀 교체만 $30,000–$80,000)라 교체 시점 정확도가 곧 비용이다.

**한계와 실배포에 필요한 것**

- 한계: 학습 셀 36개(최종 fit 29셀), 학습 수명 범위 534–1,054 (Batch1 36셀 전체는 534–1,074)로 좁고 그 밖에서 일반화 실패(Batch2 기존 구조 30셀 MAPE 33.88%). 단일 셀 종류, 실험실 고정 급속충전 프로토콜이라 현장 조건과 다를 수 있고, 예측에 초기 100사이클이 필요하다.
- 실배포에 필요한 것: 입력 분포가 학습 범위를 벗어나면 경고하는 장치, 점 예측 대신 불확실성 구간, 현장 운영 데이터로 재학습, 더 넓은 수명 범위의 학습 셀.

## 참고문헌

- Severson, K. A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.

## 팀 구성

- 구태우 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch2, Batch3)

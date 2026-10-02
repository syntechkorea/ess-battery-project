최종 모델: Ridge + ['dQ_log_var'] (seed=42)

Gap 부호 규칙: Gap = 뒤 항목 MAPE − 앞 항목 MAPE (행 이름의 앞−뒤 순서와 반대), (+) = 오차 증가. 단위는 %p.

| 구분 | MAPE (%) | 비고 |
|---|---|---|
| Train (Batch1 CV) | 9.27 | 프로토콜 GroupKFold 4-fold 평균 (중첩 CV) |
| Valid (Batch1 Hold-out) | 8.47 | 프로토콜 단위 hold-out |
| Test (Batch2) | 29.6 | 1차 테스트셋 최종 성능 (39셀, 돌출 셀 포함) |
| Gap (Train-Valid) | -0.8 | Valid − Train. (+) : 과적합 의심 [Gap = 뒤 항목 MAPE − 앞 항목 MAPE (행 이름의 앞−뒤 순서와 반대), (+) = 오차 증가] |
| Gap (Valid-Test) | 21.14 | Test − Valid. (+) : 배치 간 일반화 저하 의심 [Gap = 뒤 항목 MAPE − 앞 항목 MAPE (행 이름의 앞−뒤 순서와 반대), (+) = 오차 증가] |
| Gap (Target-Test) | 20.5 | Test − Target. (+): 목표 미달; Target : 원논문 9.1% [Gap = 뒤 항목 MAPE − 앞 항목 MAPE (행 이름의 앞−뒤 순서와 반대), (+) = 오차 증가] |
| Test (Batch3) | 12.22 | 선택: 2차 테스트셋 (44셀, 돌출 셀 포함) |
| Gap (Batch2-Batch3) | -17.39 | 선택: Batch3 − Batch2. (+) : Batch3 오차가 Batch2보다 큼, (−) : Batch3 오차가 더 작음 [Gap = 뒤 항목 MAPE − 앞 항목 MAPE (행 이름의 앞−뒤 순서와 반대), (+) = 오차 증가] |
| Gap (Target-Test Batch3) | 3.12 | 선택: Test(Batch3) − Target. (+): 목표 미달; Target : 원논문 9.1% [Gap = 뒤 항목 MAPE − 앞 항목 MAPE (행 이름의 앞−뒤 순서와 반대), (+) = 오차 증가] |

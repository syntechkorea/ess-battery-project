# data/

원본 데이터와 캐시는 용량이 커서 **저장소에 포함되지 않는다** (`.gitignore`의 `*.mat`, `data/cache/`).

## 출처

- MIT–Stanford Battery Dataset (Severson et al., *Nature Energy* 2019)
- 원본 `.mat` 파일은 data.matr.io에서 받는다.

## 필요한 파일 (이 폴더 `data/`에 그대로 둔다)

| 배치 | 파일 이름 | 크기 (약) | 역할 |
|---|---|---|---|
| Batch1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 3.0 GB | 학습 |
| Batch2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 2.0 GB | 테스트 (필수) |
| Batch3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 3.2 GB | 추가 검증 (선택) |

## 캐시 만들기 (프로젝트 루트에서)

```bash
python -m src.load_data     # .mat → data/cache/b1.pkl, b2.pkl, b3.pkl (합계 약 175MB)
python -m src.preprocess    # 셀 품질 플래그 → data/cache/cell_flags.csv
```

- 캐시에는 셀마다 요약 시계열과 처음 100사이클의 `Qdlin`/`Tdlin`/`dQdV`만 들어 있다.
- cycle 10 실측 충전 전류 표(`data/cache/charge_current_c10.csv`)는 피처를 처음 만들 때(`python -m src.train` 등) `.mat`에서 읽어 자동으로 저장된다.
- 캐시가 만들어진 뒤의 분석·학습은 캐시를 읽는다. 단, 위 충전 전류 표를 처음 만들 때는 `.mat`이 필요하다.

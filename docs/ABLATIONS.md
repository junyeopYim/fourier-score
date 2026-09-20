# Gaussian output parameterization ablations

이 실험은 데이터의 주파수별 2차 통계를 제공하는 것이 같은 NCSN++와 DSM의
생성 품질 또는 학습 예산 효율을 개선하는지 검증합니다. 새 목적함수나 생성
품질 향상이 증명됐다는 주장은 하지 않습니다.

## 비교군

| `loss.type` | 기준 score | 신경망 잔차의 scaled-score 계수 | 역할 |
|---|---|---|---|
| `score` | 없음 | 1 | 기존 DSM 기준 |
| `scalar_gaussian` | 채널별 상수 분산 Gaussian | 채널별 b | 주파수 정보를 제거한 대조군 |
| `fourier_gaussian_unscaled` | 원래 Fourier Gaussian | 1 | 기준항의 효과 분리 |
| `fourier_gaussian` | 원래 Fourier Gaussian | 주파수별 b | 전체 방법 |

`scalar_gaussian`은 동일한 training-only cache에서 각 채널의 floored P를
주파수 평균한 값으로 대체합니다. 전체 평균 이미지 mu는 그대로 유지하고,
기준항과 잔차 스케일 모두에 평균 분산을 사용합니다. 상수 spectral multiplier는
pixel별 연산과 같으므로 불필요한 FFT를 수행하지 않습니다. EDM 전체를 재현한
대조군은 아닙니다. `diffusion`은 기존 score와의 부호 규약 확인용으로 남아 있지만
기본 비교 목록에서는 제외합니다.

전체 방법이 scalar 대조군보다 좋으면 주파수별 통계의 유용성을 지지합니다.
unscaled Fourier 대조군과 차이가 없다면 잔차 스케일링의 추가 효과는 입증되지
않은 것입니다. 어느 비교도 성능 차이가 나도록 설계된 정답 실험은 아닙니다.

## 실행

프로젝트 root에서 설치된 환경의 Python으로 실행합니다.

```bash
# 모델·데이터·통계를 생성하지 않고 네 비교군의 명령을 확인
python scripts/run_comparison.py -c configs/cifar10_ablation.json \
  --device cuda --seeds 42 --dry-run

# 데이터 준비와 device 동작 검사가 끝난 뒤, 명시적으로 학습을 시작할 때
python scripts/run_comparison.py -c configs/cifar10_ablation.json \
  --device cuda --seeds 42

# 반복 seed는 별도 run에 저장; 실행 계획만 먼저 확인
python scripts/run_comparison.py -c configs/cifar10_ablation.json \
  --device cuda --seeds 43 44 --dry-run

# 세 핵심 비교군만 선택할 수도 있음
python scripts/run_comparison.py -c configs/cifar10_ablation.json \
  --objectives score scalar_gaussian fourier_gaussian --seeds 42 --dry-run
```

50k update 프리셋은 탐색용입니다. 초기 우열만으로 수렴 성능이나 억셉 가능성을
판정하지 않습니다. 비교군 사이에서는 데이터 split, augmentation, seed, 초기
backbone, microbatch, optimizer, EMA, noise sampling과 평가 설정을 유지합니다.
실행기는 모든 출력 경로를 시작 전에 검사합니다. 기존 결과가 있으면 새 `name`을
지정하거나 해당 run을 `train.py -r last.pt`로 명시적으로 재개하십시오.

## 기록과 해석

- `metrics.jsonl`: 기존 전체/노이즈별 DSM과 선택적인 주파수별 및
  노이즈×주파수 DSM. step 0은 분석적인 기준항의 초기 성능을 보여줍니다.
- `frequency_bands`: 방사형 주파수의 경계와 채널당 FFT mode 수. 각 대역은
  mode별 평균이므로 전체 DSM을 복원할 때 mode 수로 가중해야 합니다.
  빈 대역 또는 빈 노이즈 bin은 null입니다. 주파수 진단은 CPU FFT를 사용합니다.
- `training_wall_seconds`: 초기화/데이터/통계 준비, 학습, 검증과 이전 저장을
  포함한 누적 세션 시간. resume 사이 중단 시간과 현재 checkpoint 쓰기는
  제외합니다. GPU 작업은 측정 경계에서 동기화합니다. 이는 optimizer만의
  시간이나 GPU 대여 비용이 아닙니다.
- `last.pt`, `ema_*.pt`, 샘플 `settings.json`: 해당 checkpoint의 누적 시간과
  step을 연결해 FID–update 및 FID–시간 곡선을 만들 수 있습니다. 과거 기록에
  시간이 없으면 추론 출력은 null로 남기며 임의 추정하지 않습니다.

통계 cache가 첫 실험에서만 새로 만들어지면 준비 시간의 조건이 다릅니다.
비교 전에 공통 통계를 준비해 모든 실험을 같은 cache 조건에서 실행하고,
통계 추정 비용은 별도로 보고하십시오. 진단/저장 주기와 장치도 맞춰야 합니다.

FID는 기존 `sample.py` → `evaluate.py fid` 경로로 checkpoint별 평가합니다.
공통 real split, 샘플 수, sampler, sampling batch, 실제 NFE와 평가 구현을
맞추고, pilot의 적은 샘플 수와 최종 평가의 샘플 수를 구분해 기록하십시오.
낮은 초기 DSM만으로 FID 향상을 주장하지 않습니다. 같은 품질에 도달하는
시간이 줄어드는 결과도 최종 품질 우위와 구분해 보고할 수 있습니다.

`standard_error`는 평가 관측치에 대한 표준오차입니다. 학습 seed별 변동성은
독립 학습 결과를 모아서 따로 계산해야 합니다.

## 기존 실험과 호환성

기존 `score`, `diffusion`, `fourier_gaussian`의 수식과 기본 설정은 유지합니다.
체크포인트에 새로운 output type이 명시되므로 서로 바꿔 재개하지 못합니다.
코드 hash 검사도 유지하므로 패치 전 checkpoint의 학습은 원래 코드 checkout에서
재개해야 합니다. 이전 checkpoint의 EMA 추론은 기존처럼 source 차이 경고와
함께 가능하며, 새로운 주파수 진단도 선택할 수 있습니다. 비교 실험의 이름은
`cifar10_50k_holdout5000_{parameterization}_s{seed}`이므로 기존 기본 run과 구분됩니다.

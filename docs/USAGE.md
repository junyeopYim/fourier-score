# 한국어 사용 안내

이 저장소는 같은 NCSN++와 DSM 목적함수에서 출력 재매개화를 비교합니다.
처음 읽을 때는 [제안 수식](MATH.md)과
[핵심 구현](../fourier_score/method.py),
[통계 추정](../fourier_score/statistics.py)을 함께 보면 됩니다.

## 설치와 작은 실행

저장소 루트에서 실행합니다.

```bash
uv sync --locked --python 3.11
uv run --locked python train.py -c configs/smoke.json --device cpu
uv run --locked python sample.py \
  -r saved/synthetic_ve_fourier_gaussian_s0/last.pt -o saved/smoke_samples
uv run --locked python evaluate.py dsm \
  -r saved/synthetic_ve_fourier_gaussian_s0/last.pt -o saved/smoke_dsm.json
```

합성 데이터로 세 번 업데이트하는 동작 검사입니다. 생성 품질 실험이 아닙니다.
기존 디렉터리는 덮어쓰지 않으므로 재실행할 때 새 `name`과 출력 경로를 사용하십시오.

## CIFAR-10

- `configs/cifar10_ablation.json`: 50K 탐색 실험, 5,000장 validation 분리.
- `configs/cifar10_950k.json`: 950,000 update, 전체 train 사용.
- `configs/cifar10_1m3.json`: 1,300,000 update, 전체 train 사용.

뒤의 두 설정은 학습 길이와 이름만 다릅니다. test DSM은 최종 진단용이며 튜닝에
사용하지 않습니다. 저장 이름에 학습 길이·split·출력 규약·seed가 들어갑니다.
95,000 update 프리셋은 없으며, 기존 950K 설정과 구분하십시오.

```bash
# 명령 확인: 다운로드나 학습을 실행하지 않습니다.
bash scripts/reproduce_cifar10.sh 950k --device cuda --seeds 42 --dry-run

# 한 비교군 학습: 데이터 다운로드와 공통 통계 준비까지 수행합니다.
uv run --locked python train.py -c configs/cifar10_950k.json --download \
  --device cuda --parameterization fourier_gaussian --set trainer.microbatch_size=32
```

`--parameterization`으로 `score`, `scalar_gaussian`,
`fourier_gaussian_unscaled`, `fourier_gaussian`을 선택합니다.
같은 DSM 목적함수를 사용하며 backbone과 sampler는 함께 바뀌지 않습니다.
전체 비교 실행기는 통계를 한 번 준비한 뒤 seed별 비교군을 순서대로 실행합니다.

## 평가와 재개

`sample.py`는 EMA 이미지와 실행 조건을 저장합니다.
`evaluate.py dsm`은 held-out DSM, `evaluate.py fid`는 생성 이미지의 FID/IS를
계산합니다. FID에는 선택 의존성 `uv sync --locked --extra metrics`와 첫 실행 시
Inception 가중치 다운로드가 필요합니다. 정확한 실행 예시는
[실험 레시피](EXPERIMENTS.md)에 있습니다.

학습 재개는 `train.py -r <last.pt>`를 사용합니다. **구조 개편 이전 checkpoint의
추론은 지원하지만, 학습 재개는 기존 source hash 검사 때문에 해당 원래 revision에서
해야 합니다.** 상세 내용은 [개발·이전 안내](DEVELOPMENT.md)를 참고하십시오.

## Frozen LDM과 공식 pretrained 참조

LDM은 공식 KL/VQ first stage를 고정하고 denoiser를 처음부터 학습합니다.
가중치와 데이터 준비, latent 캐시·통계, 비교 학습, latent sampling과 RGB decode,
FID까지 [LDM 실험 안내](LDM.md)에 정리했습니다.

```bash
# pretrained/ldm/와 data/의 기존 경로에 가중치·데이터·공식 split을 준비합니다.
uv run --locked --extra datasets python scripts/download_ldm.py \
  --model lsun_churches --with-data
uv run --locked --extra ldm python ldm.py prepare \
  -c configs/ldm/lsun_churches_l2.json --device cuda
uv run --locked --extra ldm python ldm.py compare \
  -c configs/ldm/lsun_churches_l2.json --device cuda --seeds 42 43 44

# 공식 Score-SDE 참조: CIFAR-10 기본형·deep형과 FFHQ-256.
uv run --locked --extra datasets python scripts/download_score_sde.py --all
uv run --locked python scripts/import_score_sde.py --model cifar10_ncsnpp_continuous
uv run --locked python sample.py \
  -r pretrained/score_sde/cifar10_ncsnpp_continuous/ema.pt \
  -o saved/official_cifar10_reference --device cuda --num-samples 64 --batch-size 64
```

Score-SDE 원본 가중치·config·해시는 `pretrained/score_sde/<model>/`에 저장됩니다.
다운로드를 반복하면 파일 검증 후 재사용하며, EMA 변환 결과는 추론 전용입니다.
이 공개 pretrained 모델들은 학습 이력이 다르므로 처음부터 학습하는 paired
비교군과 구분합니다. [다운로드·변환 상세](SCORE_SDE.md)를 참고하십시오.
기존 장치·구조 검증 기록도 장기 학습이나 FID 성능 검증을 뜻하지 않습니다.

설치: [INSTALL.md](INSTALL.md) · 설정: [CONFIG.md](CONFIG.md) ·
실험 해석: [ABLATIONS.md](ABLATIONS.md) · MPS: [MPS.md](MPS.md)

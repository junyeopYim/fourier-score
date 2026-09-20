# Fourier Image Generation Template

같은 **NCSN++ backbone과 DSM 목적함수**에서 score 출력과 Gaussian 출력 재매개화를 비교하는 이미지 생성 연구 프로젝트입니다. 첫 비교는 `score`, `scalar_gaussian`, `fourier_gaussian`으로 진행하고, `fourier_gaussian_unscaled`로 잔차 스케일링의 효과를 추가 분석할 수 있습니다. `diffusion`도 부호 규약 비교용으로 유지합니다. 실험 설계와 실행 방법은 [docs/ABLATIONS.md](docs/ABLATIONS.md)에 있습니다. `victoresque/pytorch-template`의 역할별 폴더·JSON 설정·BaseTrainer 구조를 바탕으로 구성했습니다.

Fourier Gaussian은 고정 Gaussian score에 주파수별로 스케일된 신경망 잔차를 더합니다. **MMSE scalar gate / linear gate / 학습형 gate는 없습니다.**

`scalar_gaussian`은 Gaussian 기준항의 계수를 줄이는 gate가 아닙니다. 같은 평균 이미지를 유지하면서, 주파수별 분산만 채널별 상수로 바꾸는 대조군입니다.

## 검증 현황 — 2026-09-20

코드 `ea56e75`를 기존 프로젝트 `.venv`에서 다시 검사했습니다.

| 확인 항목 | 결과 |
|---|---|
| 전체 테스트 | **143 passed, 2 skipped, 0 failed** — MPS 하드웨어가 없어 2개 건너뜀 |
| 실행 환경 | Python **3.11.15**, PyTorch **2.14.0+cu130**, TorchVision **0.29.0+cu130** |
| 실제 장치 검사 | CPU 및 **NVIDIA GeForce RTX 5060 Ti / CUDA 13.0**에서 소형 NCSN++ 검사 통과 |
| CIFAR-10 구조 검사 | 지원하는 **5개 출력 규약 모두 구조와 초기 backbone 가중치 hash 일치** |
| lockfile | 추적 중인 `uv.lock`에 대해 `uv lock --check --offline` 통과 |

[검증 문서](docs/VALIDATION.md), [실행 명령과 종료 코드](verification/2026-09-20/commands.json),
[테스트 원문](verification/2026-09-20/pytest.txt), [환경 및 소스 해시 요약](verification/2026-09-20/summary.json)을 공개합니다.
소형 모델 동작 검사와 전체 크기 모델의 구조 검사이며, CIFAR-10 장기 학습이나 FID/IS 성능을 입증한 결과는 아닙니다.
기존 9월 19일 기록은 [기록 목록](verification/README.md)에서 구분해 확인할 수 있습니다.

## 1. 설치

기본 Python은 **3.11**, `pyproject.toml`의 고정 버전은 **PyTorch 2.14.0 + TorchVision 0.29.0**입니다. 저장소에 포함된 `uv.lock`을 사용합니다.

```bash
cd fourier-score
uv python install 3.11
uv sync --locked --python 3.11
uv run --locked python -m pytest -q
```

이번 검증은 기존 `.venv`와 lockfile 일관성을 확인했습니다. 새 환경에서의 설치를 다시 실행한 기록은 아닙니다. `verification/uv_lock_attempt.txt`는 9월 19일 제작 환경의 과거 실패 기록으로 보존합니다.

기본 설치는 PyPI를 사용합니다. Apple Silicon에서는 macOS wheel을 사용하며, CUDA 머신은 설치된 wheel과 드라이버가 호환되어야 합니다. Windows의 기본 PyPI wheel에서 CUDA가 보이지 않는 경우에는 `docs/INSTALL.md`의 공식 index 설정 방법을 확인하십시오. 가상환경은 `.venv/`이며 TensorFlow를 설치하지 않습니다.

```bash
# 자동 선택: CUDA -> MPS -> CPU
uv run --locked python doctor.py

# Apple MPS: 실제 convolution/attention/FIR/backward/Adam/sample/RNG 검사
uv run --locked python doctor.py --device mps

# NVIDIA
uv run --locked python doctor.py --device cuda
```

장기 학습 전에 사용할 장치에서 `doctor.py`를 실행하십시오. 이번 CPU/CUDA 결과는 [검증 기록](docs/VALIDATION.md)에 있으며, MPS는 실제 하드웨어에서 추가 확인해야 합니다.

## 2. 다운로드 없는 동작 검사

```bash
uv run --locked python train.py -c configs/smoke.json
uv run --locked python sample.py \
  -r saved/synthetic_ve_fourier_gaussian_s0/last.pt \
  -o saved/smoke_samples
uv run --locked python test.py \
  -r saved/synthetic_ve_fourier_gaussian_s0/last.pt \
  -o saved/smoke_dsm.json
```

Smoke는 실제 소형 NCSN++를 사용하지만 입력은 합성 noise입니다. 생성 품질이나 논문 성능을 평가하는 실험이 아닙니다. 같은 run 디렉터리를 덮어쓰지 않으므로 재실행 시 `--set name=smoke2`처럼 이름을 바꾸십시오.

## 3. MNIST: 출력 재매개화 비교

MNIST는 28×28을 32×32로 zero-padding하고 [-1,1] 좌표를 사용합니다. 학습 데이터 중 5,000장을 validation으로 분리합니다.

```bash
uv run --locked python prepare.py -c configs/mnist.json --download

uv run --locked python train.py -c configs/mnist.json --set loss.type=score
uv run --locked python train.py -c configs/mnist.json --set loss.type=scalar_gaussian
uv run --locked python train.py -c configs/mnist.json --set loss.type=fourier_gaussian
```

`name=auto`이면 loss와 seed에 따라 별도 디렉터리가 자동으로 생성됩니다.

```bash
# 동일한 실험을 MPS에서 실행
uv run --locked python train.py -c configs/mnist.json \
  --device mps --set loss.type=fourier_gaussian

# 전체 배치는 그대로, GPU에 올라가는 microbatch만 줄이기
uv run --locked python train.py -c configs/mnist.json \
  --device mps --set loss.type=fourier_gaussian \
  --set trainer.microbatch_size=32
```

기본 비교 실행기는 네 비교군을 순차 실행합니다. 아래처럼 `--objectives`를 지정하면 세 핵심 비교군부터 실행할 수 있습니다. `--seeds`로 동일 seed의 실험군을 묶어 반복하며, 실행 전 모든 저장 경로를 확인해 기존 실험을 덮어쓰지 않습니다.

```bash
uv run --locked python scripts/run_comparison.py -c configs/mnist.json --device mps \
  --objectives score scalar_gaussian fourier_gaussian
```

## 4. CIFAR-10: 실제 NCSN++ 구조 그대로

```bash
# 모든 비교군이 같은 통계 cache 상태에서 시작하도록 먼저 준비
uv run --locked python prepare.py -c configs/cifar10_ablation.json --download
uv run --locked python inspect_model.py -c configs/cifar10_ablation.json

# 50k update 탐색 실험의 명령만 확인; --dry-run을 빼면 학습 시작
uv run --locked python scripts/run_comparison.py -c configs/cifar10_ablation.json \
  --device cuda --objectives score scalar_gaussian fourier_gaussian \
  --seeds 42 --set trainer.microbatch_size=32 --dry-run

# 기존 단일 실험 프리셋으로 학습할 때
uv run --locked python train.py -c configs/cifar10.json \
  --device cuda --set loss.type=fourier_gaussian \
  --set trainer.microbatch_size=32
```

CIFAR-10 프리셋은 nf=128, ch_mult=[1,2,2,2], level당 residual block 4개, attention resolution 16, BigGAN++ residual block, FIR, progressive input residual 구조입니다. 9월 20일 전체 크기 모델을 CPU에서 생성한 검사에서 다섯 출력 규약 모두 다음과 같았습니다.

| 항목 | 값 |
|---|---:|
| 학습 가능한 파라미터 | **62,758,787** |
| 고정 Fourier time embedding을 포함한 전체 파라미터 | 62,758,915 |
| Attention block | 6 |
| BigGAN++ ResNet block | 44 |

[CIFAR-10 ablation 구조 검사](verification/2026-09-20/cifar10_ablation_architecture.json)에 출력 규약별 구조 hash와 **초기 backbone 가중치 hash의 일치**를 기록했습니다. 이 검사는 양수인 가상 통계를 사용하며 데이터를 읽거나 전체 모델을 학습하지 않습니다. Gaussian 통계와 DFT 행렬은 학습 파라미터가 아닙니다.

`configs/cifar10_ablation.json`은 50,000 update, 10,000 update마다 EMA snapshot, 주파수 대역 6개를 사용하는 탐색용 설정입니다. 동일 seed의 세 비교군에서 checkpoint별 FID와 학습 예산 곡선을 먼저 확인하고, 반복 seed 및 `fourier_gaussian_unscaled`를 추가할 수 있습니다. 통계 준비 비용은 별도로 기록하고, 비교군 간 microbatch·검증·저장·샘플링 조건을 맞추십시오. 50k 시점의 우열만으로 수렴 성능을 판단하지 않습니다.

`configs/cifar10.json`은 5,000장 holdout을 사용합니다. `configs/cifar10_full.json`은 50,000장 전체 train과 test 진단을 사용하고, `configs/cifar10_paper950k.json`은 이전 fork의 950,000 update 설정입니다. 이름에 paper가 들어가도 논문 FID를 재현했다는 뜻은 아닙니다. test split을 하이퍼파라미터 튜닝에 사용하지 마십시오.

## 5. 출력 재매개화의 정확한 의미

`h`는 동일 NCSN++의 **sigma division 전 출력**, forward process는 `Y = alpha X + sigma epsilon`입니다.

| `loss.type` | 최종 scaled score `sigma*s` |
|---|---|
| `score` | `h` |
| `diffusion` | `-h` (`h`를 epsilon prediction으로 해석) |
| `scalar_gaussian` | `sigma*s_scalar + b_scalar*h` (주파수 평균 분산, 같은 평균 이미지) |
| `fourier_gaussian_unscaled` | `sigma*s_G + h` (Fourier 기준항만 사용) |
| `fourier_gaussian` | `sigma*s_G + F^-1[b*F(h)]` |

VE에서는 `alpha=1`, `s_G=-F^-1[F(Y-mu)/(P+sigma²)]`, `b=sqrt(P/(P+sigma²))`입니다. **Gaussian 기준항의 계수는 1**입니다. 기준항을 MMSE scalar로 축소하지 않습니다. 입력 whitening이나 sampler 변경도 loss 선택에 따라 몰래 켜지지 않습니다.

모든 경우 최종 score의 `||sigma*s + epsilon||²`를 최소화합니다. `loss.type`은 기존 설정·체크포인트와의 호환성을 위해 유지한 출력 규약 선택 이름입니다. `loss.reduction`으로 pixel mean 또는 원본 score-SDE의 half pixel sum을 선택하며, 같은 비교에서는 값을 고정하십시오. 평가 지표 `dsm_pixel_mean`은 모든 실험에서 동일한 pixel mean입니다.

`scalar_gaussian`은 각 채널의 `P`를 주파수 평균한 분산으로 바꿉니다. 통계 cache와 전체 평균 이미지는 공유하고, 계산은 동등한 pixel별 연산으로 수행합니다. `fourier_gaussian_unscaled`는 현재 Fourier 기준항을 그대로 두고 잔차 스케일링만 제거합니다. 둘 다 입력·backbone·손실 가중치·sampler를 변경하지 않습니다.

**중요:** 같은 forward process와 가중치를 사용하면 score DSM과 epsilon diffusion은 부호 재파라미터화 관계입니다. 독립적인 두 생성 원리의 성능 차이로 해석하면 안 됩니다. 일반 DDPM의 VP forward까지 비교하려면 다음 **별도 process 프리셋**을 사용하십시오.

```bash
uv run --locked python train.py -c configs/mnist_ddpm.json
uv run --locked python train.py -c configs/cifar10_ddpm.json
```

이 프리셋은 `process.type=ddpm`, 1,000개 linear beta step, ancestral DDPM sampler를 사용합니다. backbone 구조는 각 데이터셋의 NCSN++와 같지만 forward가 달라지므로 **출력 재매개화 비교와 분리**해야 합니다. DDPM forward에서도 다섯 출력 규약을 선택할 수 있으며 Fourier Gaussian의 `P`는 `alpha² P`, 평균은 `alpha mu`로 일반화됩니다. 이는 원래 VE 제안의 명시적인 확장입니다. 자세한 수식은 `docs/MATH.md`에 있습니다.

## 6. EMA 생성·평가·재시작

```bash
uv run --locked python sample.py \
  -r saved/mnist_ve_fourier_gaussian_s0/last.pt \
  -o saved/mnist_fg_samples --num-samples 1000 --batch-size 64 --steps 1000

uv run --locked python test.py \
  -r saved/mnist_ve_fourier_gaussian_s0/last.pt \
  -o saved/mnist_fg_dsm.json

uv run --locked python train.py \
  -r saved/mnist_ve_fourier_gaussian_s0/last.pt \
  --set trainer.iterations=20000
```

생성 결과는 `png/`, `samples_00000.npz` 등의 uint8 NHWC 배열, `preview.png`, `settings.json`으로 저장됩니다. NPZ key는 `samples`입니다. 마지막 round도 먼저 유효 배치 전체를 생성한 뒤 필요한 개수만 저장합니다. PC의 Langevin norm이 batch mean이기 때문에 **sampling batch size를 바꾸면 다른 sampling protocol**입니다.

VE는 `sampling.method=pc` 또는 `heun`을 지원합니다. DDPM ancestral sampler는 훈련된 전체 grid를 사용하므로 임의 `--steps` 축소를 거부합니다. 샘플링은 always EMA이며, `last.pt`와 `ema_*.pt` 모두 추론에 사용할 수 있습니다.

`last.pt`는 model, Adam, EMA, noise generator, 소비 완료된 data cursor, CPU/CUDA/MPS RNG, train-only 통계, 최종 config와 코드 hash를 저장합니다. `ema_*.pt`는 추론용이며 학습 재시작용이 아닙니다. 실제 장기 실험 도중에는 코드·loss·구조·process·optimizer·microbatch·backend 설정을 바꾸지 마십시오. 재시작 검사는 이를 거부합니다. 기존 `fourier-score` 체크포인트를 새 형식으로 resume하는 자동 변환은 없습니다.

새 학습 기록·체크포인트·샘플 `settings.json`에는 재개 전후를 합친 `training_wall_seconds`가 들어갑니다. 설정/데이터/통계 준비와 검증, 이전 저장 시간을 포함하며 세션 사이 중단 시간과 해당 체크포인트 자체의 저장 시간은 제외합니다. 과거 체크포인트의 시간을 추정해서 채우지 않습니다. `evaluation.frequency_bins>0`이면 노이즈×주파수 대역 DSM을 추가 기록합니다. 이는 CPU FFT로 계산하는 평가용 진단이며, 동일 비교에서는 진단 설정도 맞추십시오.

학습 콘솔은 기본적으로 진행률, 구간 평균 loss와 pixel mean 환산값, 속도, 학습 ETA,
평가·저장 상태를 요약합니다. CUDA에서는 할당/예약 메모리도 표시합니다.
첫 update와 이후 약 5초마다 진행 상태를 보여주며, 노이즈·주파수별 상세 값은
`metrics.jsonl`에 그대로 남습니다. `--set trainer.progress_every_seconds=1`로
갱신 주기를, `--set trainer.console=json` 또는 `quiet`로 출력 방식을 바꿀 수 있습니다.
ETA 범위, TensorBoard, source hash에 따른 기존 checkpoint 재개 제한은
[학습 상태 표시](docs/CONFIG.md#학습-상태-표시)를 참고하십시오.

## 7. 선택 기능: FID/IS

```bash
uv sync --locked --extra metrics

# 학습과 동일한 resize/crop/좌표 변환을 거친 real 이미지
uv run --locked python export_real.py -c configs/cifar10_full.json \
  --split train -o saved/cifar_real

# 생성 이미지가 실제로 50,000장 있는지 settings.json으로 확인
uv run --locked python sample.py -r saved/cifar10_ve_fourier_gaussian_s42/last.pt \
  -o saved/cifar_fg_50k --num-samples 50000 --batch-size 64

uv run --locked python metrics.py --real saved/cifar_real/png \
  --generated saved/cifar_fg_50k/png --device cuda -o saved/cifar_fg_fid.json
```

FID는 선택 dependency인 torch-fidelity를 사용하며 최초 Inception weight 다운로드가 필요합니다. TensorFlow는 필요하지 않습니다. **원본 score-SDE의 TF-Hub/TF-GAN 프로토콜과 수치를 직접 동일시하지 마십시오.** real split, 샘플 수, resize, library version, sampler batch, 실제 NFE를 모든 비교군에서 통일하십시오. MPS에서 훈련한 결과의 Inception 평가는 `metrics.py --device cpu`로 할 수 있습니다. 이번 검증에는 Inception 다운로드와 FID/IS 실행이 포함되지 않았습니다.

## 8. 폴더 구조와 추가 프리셋

```text
train.py / prepare.py / sample.py / test.py
inspect_model.py / doctor.py / export_real.py / metrics.py
parse_config.py / config.json / pyproject.toml
base/           BaseModel, BaseTrainer, consumed-cursor DataLoader
model/          동일 NCSN++, score adapters, Fourier 연산, DSM, 평가
sde/            forward processes, 공통 PC/Heun/DDPM samplers
trainer/        공통 학습 loop
logger/         터미널 진행 상태, JSONL, 선택 TensorBoard
data_loader/    MNIST, CIFAR10, 일반 image_folder, synthetic
configs/        하나의 strict schema를 상속하는 JSON
utils/          RNG, EMA, checkpoint, inference, image 저장
scripts/        동일 조건 비교 실행
verification/   실제 테스트 결과·환경·architecture audit
```

`configs/celeba64_folder.json`, `configs/ffhq256_folder.json`은 일반 이미지 디렉터리를 읽습니다. root 경로를 바꾸십시오. 파일 경로와 metadata로 cache identity를 만들므로 데이터 이동/수정이 기존 통계 cache에 영향을 줄 수 있습니다. 임의 폴더는 train 내부 holdout을 요구하며 검증 데이터가 없다고 train 성능을 validation으로 표시하지 않습니다.

FFHQ는 source의 output_skip/input_skip 구조와 level당 2개 residual block을 반영했습니다. **CelebA 폴더 프리셋은 positional embedding 구조를 사용하지만 forward는 continuous VE이며 원본 discrete SMLD/TFDS 레시피 재현이 아닙니다.** 얼굴 프리셋의 일반 folder crop/resize·holdout도 원본 공식 데이터 프로토콜과 다를 수 있습니다. 서로 다른 folder 실험은 `--set name=ffhq256_fg`처럼 이름을 구분하십시오.

프로젝트는 단일 CPU, 단일 CUDA GPU 또는 단일 MPS device를 지원합니다. DDP/multi-GPU 묶음 학습, AMP, compilation, 학습형 gate, graph latent, automatic pretrained download는 포함하지 않았습니다. 여러 GPU에서 독립 비교군을 병렬로 실행할 때는 각 프로세스의 `CUDA_VISIBLE_DEVICES`와 run 이름을 분리하십시오.

추가 문서: `docs/CONFIG.md`, `docs/MATH.md`, `docs/MPS.md`, `docs/PROVENANCE.md`, `docs/VALIDATION.md`.

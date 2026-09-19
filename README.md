# Fourier Image Generation Template

같은 **NCSN++ backbone**에서 `fourier_gaussian`, `score`, `diffusion`을 비교하는 이미지 생성 연구 프로젝트입니다. `victoresque/pytorch-template`의 역할별 폴더·JSON 설정·BaseTrainer 구조를 바탕으로 새로 구성했습니다. classifier용 LeNet이나 단순 CNN으로 대체하지 않았습니다.

**MMSE scalar gate / linear gate / 학습형 gate는 없습니다.** 이전 대화에서 정의한 Fourier Gaussian은 고정 Gaussian score에 주파수별로 스케일된 신경망 잔차를 더합니다. 이름은 `fourier_gaussian`으로 통일했습니다.

## 1. 설치

기준 날짜: **2026-09-19**. 설치 목표는 **PyTorch 2.14.0 + TorchVision 0.29.0**, Python 3.11입니다. 실제 제작 환경의 실행 검증은 **Python 3.13.5 / PyTorch 2.10.0+cpu**에서 수행했습니다. 이 차이를 숨기지 않기 위해 상세 환경과 테스트 결과를 `verification/`에 포함했습니다.

```bash
cd fourier_image_template
uv python install 3.11
uv sync --python 3.11
uv run --locked python -m pytest -q
```

**이 ZIP에는 `uv.lock`이 없습니다.** 제작 환경에서 패키지 서버 접근이 실패하여 정상적인 dependency resolution을 수행하지 못했습니다. 첫 `uv sync`로 실제 lockfile을 만든 뒤 커밋하십시오. 처음부터 `--locked`를 붙이지 마십시오. `pyproject.toml`의 torch/torchvision은 정확한 버전으로 고정되어 있지만, 이것이 전체 dependency lock을 대신하지는 않습니다. 실패 원문은 `verification/uv_lock_attempt.txt`입니다.

기본 설치는 PyPI를 사용합니다. Apple Silicon에서는 macOS wheel을 사용하며, CUDA 머신은 설치된 wheel과 드라이버가 호환되어야 합니다. Windows의 기본 PyPI wheel에서 CUDA가 보이지 않는 경우에는 `docs/INSTALL.md`의 공식 index 설정 방법을 확인하십시오. 가상환경은 `.venv/`이며 TensorFlow를 설치하지 않습니다.

```bash
# 자동 선택: CUDA -> MPS -> CPU
uv run --locked python doctor.py

# Apple MPS: 실제 convolution/attention/FIR/backward/Adam/sample/RNG 검사
uv run --locked python doctor.py --device mps

# NVIDIA
uv run --locked python doctor.py --device cuda
```

`doctor.py` 실패를 무시한 채 장기 학습을 시작하지 마십시오. 여기서는 MPS/CUDA 하드웨어가 없어 해당 device 테스트를 실행하지 못했습니다.

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

## 3. MNIST: loss만 바꾸기

MNIST는 28×28을 32×32로 zero-padding하고 [-1,1] 좌표를 사용합니다. 학습 데이터 중 5,000장을 validation으로 분리합니다.

```bash
uv run --locked python prepare.py -c configs/mnist.json --download

uv run --locked python train.py -c configs/mnist.json --set loss.type=score
uv run --locked python train.py -c configs/mnist.json --set loss.type=diffusion
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

여러 실험군을 한 번에 순차 실행할 수도 있습니다. 이미 존재하는 실험은 덮어쓰지 않습니다.

```bash
uv run --locked python scripts/run_comparison.py -c configs/mnist.json --device mps
```

## 4. CIFAR-10: 실제 NCSN++ 구조 그대로

```bash
uv run --locked python prepare.py -c configs/cifar10.json --download
uv run --locked python inspect_model.py -c configs/cifar10.json

uv run --locked python train.py -c configs/cifar10.json \
  --device cuda --set loss.type=fourier_gaussian \
  --set trainer.microbatch_size=32
```

CIFAR-10 프리셋은 nf=128, ch_mult=[1,2,2,2], level당 residual block 4개, attention resolution 16, BigGAN++ residual block, FIR, progressive input residual 구조입니다. 실제 검사에서 세 loss 모두 다음과 같았습니다.

| 항목 | 값 |
|---|---:|
| 학습 가능한 파라미터 | **62,758,787** |
| 고정 Fourier time embedding을 포함한 전체 파라미터 | 62,758,915 |
| Attention block | 6 |
| BigGAN++ ResNet block | 44 |

`verification/cifar10_architecture.json`에는 loss별 구조 hash와 **초기 가중치 hash가 같은 사실**을 기록했습니다. Gaussian 통계와 DFT 행렬은 학습 파라미터가 아닙니다.

`configs/cifar10.json`은 5,000장 holdout을 사용합니다. `configs/cifar10_full.json`은 50,000장 전체 train과 test 진단을 사용하고, `configs/cifar10_paper950k.json`은 이전 fork의 950,000 update 설정입니다. 이름에 paper가 들어가도 논문 FID를 재현했다는 뜻은 아닙니다. test split을 하이퍼파라미터 튜닝에 사용하지 마십시오.

## 5. 세 objective의 정확한 의미

`h`는 동일 NCSN++의 **sigma division 전 출력**, forward process는 `Y = alpha X + sigma epsilon`입니다.

| `loss.type` | 최종 scaled score `sigma*s` |
|---|---|
| `score` | `h` |
| `diffusion` | `-h` (`h`를 epsilon prediction으로 해석) |
| `fourier_gaussian` | `sigma*s_G + F^-1[b*F(h)]` |

VE에서는 `alpha=1`, `s_G=-F^-1[F(Y-mu)/(P+sigma²)]`, `b=sqrt(P/(P+sigma²))`입니다. **Gaussian 기준항의 계수는 1**입니다. 기준항을 MMSE scalar로 축소하지 않습니다. 입력 whitening이나 sampler 변경도 loss 선택에 따라 몰래 켜지지 않습니다.

세 경우 모두 최종 score의 `||sigma*s + epsilon||²`를 최소화합니다. `loss.reduction`으로 pixel mean 또는 원본 score-SDE의 half pixel sum을 선택하며, 같은 비교에서는 값을 고정하십시오. 평가 지표 `dsm_pixel_mean`은 모든 실험에서 동일한 pixel mean입니다.

**중요:** 같은 forward process와 가중치를 사용하면 score DSM과 epsilon diffusion은 부호 재파라미터화 관계입니다. 독립적인 두 생성 원리의 성능 차이로 해석하면 안 됩니다. 일반 DDPM의 VP forward까지 비교하려면 다음 **별도 process 프리셋**을 사용하십시오.

```bash
uv run --locked python train.py -c configs/mnist_ddpm.json
uv run --locked python train.py -c configs/cifar10_ddpm.json
```

이 프리셋은 `process.type=ddpm`, 1,000개 linear beta step, ancestral DDPM sampler를 사용합니다. backbone 구조는 각 데이터셋의 NCSN++와 같지만 forward가 달라지므로 **loss-only 비교와 분리**해야 합니다. DDPM forward에서도 세 objective를 선택할 수 있으며 Fourier Gaussian의 `P`는 `alpha² P`, 평균은 `alpha mu`로 일반화됩니다. 이는 원래 VE 제안의 명시적인 확장입니다. 자세한 수식은 `docs/MATH.md`에 있습니다.

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

## 7. 선택 기능: FID/IS

```bash
uv sync --extra metrics

# 학습과 동일한 resize/crop/좌표 변환을 거친 real 이미지
uv run --locked python export_real.py -c configs/cifar10_full.json \
  --split train -o saved/cifar_real

# 생성 이미지가 실제로 50,000장 있는지 settings.json으로 확인
uv run --locked python sample.py -r saved/cifar10_ve_fourier_gaussian_s42/last.pt \
  -o saved/cifar_fg_50k --num-samples 50000 --batch-size 64

uv run --locked python metrics.py --real saved/cifar_real/png \
  --generated saved/cifar_fg_50k/png --device cuda -o saved/cifar_fg_fid.json
```

FID는 선택 dependency인 torch-fidelity를 사용하며 최초 Inception weight 다운로드가 필요합니다. TensorFlow는 필요하지 않습니다. **원본 score-SDE의 TF-Hub/TF-GAN 프로토콜과 수치를 직접 동일시하지 마십시오.** real split, 샘플 수, resize, library version, sampler batch, step 수를 모든 비교군에서 통일하십시오. MPS에서 훈련한 결과의 Inception 평가는 `metrics.py --device cpu`로 할 수 있습니다. 제작 환경에서는 Inception 다운로드와 FID 실행을 검증하지 못했습니다.

## 8. 폴더 구조와 추가 프리셋

```text
train.py / prepare.py / sample.py / test.py
inspect_model.py / doctor.py / export_real.py / metrics.py
parse_config.py / config.json / pyproject.toml
base/           BaseModel, BaseTrainer, consumed-cursor DataLoader
model/          동일 NCSN++, score adapters, Fourier 연산, DSM, 평가
sde/            forward processes, 공통 PC/Heun/DDPM samplers
trainer/        공통 학습 loop
logger/         JSONL, 선택 TensorBoard
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

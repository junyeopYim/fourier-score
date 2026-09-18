# Score SDE Research Template

`score_sde_pytorch`의 핵심 연구 코드를 `pytorch-template`과 비슷한 역할별 구조로 정리한 **독립 실행형 프로젝트**입니다. 원본 저장소를 다시 내려받는 부트스트랩이 아니라, 필요한 모델·수식·학습 코드가 이 폴더 안에 들어 있습니다.

원본 기준: `junyeopYim/score_sde_pytorch@534c75478bf30acbff83568a52d843d0ff99913f` (`Add mmse`). 원본 GitHub 저장소는 수정하지 않았습니다. 출처·변경 범위는 `NOTICE`, `docs/PROVENANCE.json`, `docs/MIGRATION.md`에 정리했습니다.

## 구조

```text
score_sde_template/
├── train.py                 # 학습 / last.pt 재시작
├── prepare.py               # 데이터 준비 및 train-only 가우시안 통계
├── sample.py                # EMA 이미지 생성, NPZ 분할 저장
├── test.py                  # EMA DSM 평가
├── metrics.py               # 선택 기능: TF-Hub / TF-GAN FID·IS
├── pyproject.toml           # uv 의존성·선택 기능·PyTorch 인덱스
├── .python-version          # Python 3.11
├── uv.lock                  # 첫 uv sync 성공 시 생성; Git에 커밋
├── config.json              # 기본값: MNIST spectral_mmse
├── parse_config.py          # JSON 상속, 설정 검증, --set
├── configs/                 # 데이터셋별 숫자 설정
├── base/                    # 체크포인트 생명주기, 재시작 가능한 배치 스트림
├── data_loader/             # MNIST/CIFAR, CelebA/FFHQ, 이미지 폴더
├── model/
│   ├── model.py             # 모델 생성, score convention, 이산/연속 wrapper
│   ├── gaussian.py          # baseline / spectral / isotropic / linear / MMSE
│   ├── loss.py              # continuous DSM, discrete SMLD/DDPM
│   ├── metric.py            # 난수 상태를 분리한 DSM 평가
│   └── backbones/           # NCSN++, 레이어, native PyTorch FIR 연산
├── sde/                     # VE / VP / sub-VP, PC / probability-flow ODE
├── trainer/trainer.py       # 하나의 step 기반 Trainer
├── logger/                  # JSONL, 선택적 TensorBoard
├── utils/                   # EMA, 체크포인트, RNG, inference
├── tests/                   # 수식·설정·모델·재시작 회귀 검사
├── docs/                    # 변경 내역, 검증 기록, 출처
├── data/                    # 데이터와 통계 캐시 (빈 디렉터리)
└── saved/                   # 실험 결과 (빈 디렉터리)
```

## 1. 설치와 동작 검사

프로젝트 루트에서 실행합니다. Python은 `.python-version`과 `requires-python`으로 **3.11**을 지정했습니다. 설치·실행은 `uv`로 관리합니다. 가상환경은 `.venv/`이며 별도로 activate할 필요가 없습니다.

**이 ZIP에는 `uv.lock`이 아직 없습니다.** 제작 환경에서 외부 패키지 서버의 DNS 조회가 실패하여 실제 의존성 해석·설치를 검증하지 못했습니다. 처음에는 `uv sync`를 실행해 lockfile을 생성하십시오. 첫 실행부터 `--locked`를 붙이면 lockfile이 없어 실패합니다. 생성 후에는 `uv.lock`을 Git에 커밋하고 `uv sync --locked` / `uv run --locked`를 사용합니다.

```bash
# uv가 없는 경우 한 번 설치 (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
# 셸을 다시 열거나: source "$HOME/.local/bin/env"

uv python install 3.11
uv sync                           # 첫 설치: .venv + uv.lock 생성
uv run --locked python -m pytest -q

# 다운로드 없이 합성 입력 + 실제 소형 NCSN++로 실행 경로 확인
uv run --locked python train.py -c configs/smoke.json
uv run --locked python sample.py -r saved/smoke/last.pt -o saved/smoke/samples \
  --num-samples 4 --batch-size 2 --steps 4
uv run --locked python test.py -r saved/smoke/last.pt -o saved/smoke/evaluation.json
```

Linux x86-64 / Windows AMD64는 **torch 2.10.0 + torchvision 0.25.0의 CUDA 12.8 빌드**를 명시적으로 선택합니다. Apple Silicon macOS는 PyPI 휠을 사용하며 학습 코드는 CPU로 실행합니다. Intel Mac / Linux ARM은 이 설정의 지원 대상이 아닙니다. NVIDIA 드라이버는 별도로 준비해야 하며, 이 ZIP에 CUDA 패키지나 드라이버가 들어 있는 것은 아닙니다. GPU 없이도 CUDA 휠의 CPU 연산을 사용할 수 있지만 다운로드가 큽니다. CPU 전용 인덱스로 바꾸는 방법은 `docs/UV.md`에 있습니다.

기본 `uv sync`는 테스트용 `dev` 그룹도 설치하며 TensorFlow는 설치하지 않습니다. 얼굴 입력은 `--extra faces`, FID/IS는 `--extra metrics`, TensorBoard는 `--extra tensorboard`로 선택합니다. NumPy는 선택적 TensorFlow 2.15와 함께 쓸 수 있도록 `>=1.26.4,<2`로 제한했습니다. **설치 방식만 바꿨으며 Python 소스와 실험 JSON 설정은 이전 ZIP과 바이트 단위로 같습니다.** 라이브러리 버전이 달라진 환경에서의 학습 재현성은 별개입니다.

`smoke`는 생성 품질 실험이 아닙니다. 출력은 무작위 입력으로 몇 step 학습한 결과이며 품질을 평가할 의미가 없습니다. 같은 실험 이름의 재실행은 덮어쓰지 않고 오류를 냅니다. 다른 이름은 `--set name=smoke2`로 지정하십시오.

## 2. MNIST

```bash
uv run --locked python prepare.py -c config.json --download
uv run --locked python train.py -c config.json

# baseline 비교: 이름과 reference.mode 두 곳을 변경
uv run --locked python train.py -c configs/mnist.json --set name=mnist_baseline
```

기본 `config.json`은 `configs/mnist.json`을 상속하고 `spectral_mmse`만 선택합니다. MNIST는 28×28을 32×32로 패딩하고 [-1,1] 좌표를 사용합니다. 5,000장 holdout을 유지합니다.

## 3. CIFAR-10 spectral_mmse

```bash
uv run --locked python prepare.py -c configs/cifar10_paper.json --download
uv run --locked python train.py -c configs/cifar10_paper.json \
  --set name=cifar10_spectral_mmse \
  --set reference.mode=spectral_mmse
```

학습 VRAM이 부족하면 **총 배치는 유지하고** `--set trainer.microbatch_size=32`를 추가하십시오. 한 총 배치를 모두 처리한 뒤 gradient clipping, Adam, EMA를 각각 한 번 적용합니다. 마이크로배치 분할은 난수 호출 순서를 바꾸므로 모든 비교군에서 같은 값을 사용해야 합니다. 재시작 중에는 변경을 허용하지 않습니다.

기본은 FP32, TF32 비활성화, AMP 미사용, native PyTorch FIR입니다. CUDA JIT 확장을 빌드하지 않습니다. **단일 CPU 또는 단일 CUDA GPU** 학습입니다. DDP나 여러 GPU를 묶는 기능은 넣지 않았습니다. 서로 다른 GPU에서 독립적인 비교군을 병렬 실행하는 것은 가능합니다.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked python train.py -c configs/cifar10_paper.json --set name=cifar_baseline
CUDA_VISIBLE_DEVICES=1 uv run --locked python train.py -c configs/cifar10_paper.json \
  --set name=cifar_mmse --set reference.mode=spectral_mmse
```

위 두 명령은 각각 별도의 터미널에서 실행합니다. 첫 통계 생성은 `prepare.py`를 한 번 완료한 뒤 공유하십시오. 동시에 같은 통계 파일을 생성하거나 같은 실험 디렉터리에 쓰는 사용법은 지원하지 않습니다.

## 4. 설정 프리셋

| 파일 | 학습 설정의 출처 / 주요 값 |
|---|---|
| `mnist.json` | fork의 MNIST-small: nf 32, 10,000 updates, batch 128, 5,000 holdout |
| `cifar10_upstream.json` | 원본 VE NCSN++ config: nf 128, 1,300,001 updates, batch 128, 전체 train |
| `cifar10_paper.json` | fork의 `--protocol paper`: 위 모델/optimizer, **950,000 updates**, 전체 train |
| `cifar10_vp.json` | 원본 continuous VP DDPM++ config: positional embedding, EMA 0.9999, FIR off |
| `celeba64.json` | 원본 64×64 **discrete VE/SMLD**: 1,300,001 updates, batch 128, sigma_max 90 |
| `ffhq256.json` | 원본 256×256 continuous VE: 2,400,001 updates, batch 64, sigma_max 348, 2,000 scales |
| `smoke.json` | 합성 데이터, 축소 NCSN++, 3 updates; 실행 검사 전용 |

학습 recipe의 핵심 숫자는 유지했지만, 전체 원본 CLI와 모든 평가 옵션을 1:1로 복제한 것은 아닙니다. 특히 **샘플 미리보기의 기본 배치는 64**이며, 원본 CIFAR FID의 유효 배치 1024를 자동으로 사용하지 않습니다. `cifar10_paper`라는 이름도 이미 논문 수치를 재현했다는 뜻이 아닙니다.

JSON은 `extends`로 공통 설정을 상속합니다. 최종 값을 확인하려면 다음을 실행하십시오.

```bash
uv run --locked python train.py -c configs/cifar10_paper.json --dry-run
uv run --locked python train.py -c configs/cifar10_paper.json --dry-run \
  --set training.batch_size=128 --set trainer.microbatch_size=32
```

지원 기준항은 `baseline`, `spectral`, `isotropic`, `spectral_linear`, `isotropic_linear`, `spectral_mmse`, `isotropic_mmse`입니다. 가우시안 기준항은 VE/SMLD에서만 허용합니다. VP/sub-VP는 baseline 경로입니다. MMSE는 full FFT power에 대한 **이미지별 스칼라 가중치**이며, 주파수별 gate로 변경하지 않았습니다.

## 5. 재시작 / 출력

```bash
uv run --locked python train.py -r saved/cifar10_spectral_mmse/last.pt

# 원래 종료 지점보다 더 학습: 총 update 수를 지정
uv run --locked python train.py -r saved/mnist_spectral_mmse/last.pt --set training.n_iters=20000
```

`last.pt`에는 모델, Adam, EMA, 완료 step, 가우시안 통계, train/validation index, 배치 permutation/cursor, Python/NumPy/Torch/CUDA RNG, 설정과 코드 해시가 들어갑니다. `ema_step_*.pt`는 추론용 경량 EMA 스냅숏이므로 학습 재시작에는 사용하지 않습니다.

재시작은 모델·손실·optimizer·배치·마이크로배치·통계·코드가 달라지면 실패하도록 했습니다. 같은 하드웨어·라이브러리·설정에서의 재현성을 목표로 하며, CPU↔GPU 또는 다른 CUDA 환경 사이의 bitwise 동일성을 보장하지 않습니다. 데이터가 이동하면 얼굴 데이터의 경로 기반 fingerprint가 달라져 재시작이 거부될 수 있습니다.

**원래 `mnist_compare`에서 만든 체크포인트를 이 템플릿으로 직접 resume하는 기능은 없습니다.** 기존 장기 실험은 원래 코드에서 마무리하십시오. 이것은 이후 실험용으로 정리한 새 프로젝트이며, 자동 변환기 없이 예전 파일을 새 형식으로 추측해 읽지 않습니다.

## 6. 샘플 생성 / DSM 평가

```bash
uv run --locked python sample.py -r saved/cifar10_spectral_mmse/last.pt \
  -o saved/cifar10_spectral_mmse/preview --num-samples 64 --batch-size 64

uv run --locked python test.py -r saved/cifar10_spectral_mmse/last.pt \
  -o saved/cifar10_spectral_mmse/dsm.json --max-images 10000 --batch-size 128
```

생성 파일은 `samples_00000.npz` 등의 uint8 NHWC 배열(`samples` key), `preview.png`, `settings.json`입니다. 큰 표본은 작은 파일로 나누며, 마지막 round도 먼저 유효 배치 전체를 생성하고 필요한 개수만 내보냅니다. 출력 디렉터리가 비어 있지 않으면 덮어쓰지 않습니다. 부분 샘플 생성 자동 resume는 포함하지 않았습니다.

연속 시간 모델은 `--steps`로 샘플러 grid만 바꿀 수 있습니다. 이산 CelebA는 훈련된 sigma/label grid를 유지해야 합니다. `--sampler ode`도 사용할 수 있습니다. 원본의 배치 평균 Langevin norm을 유지했으므로 **유효 샘플 배치 변경은 다른 샘플 프로토콜**입니다. `--model-batch-size`는 모델 forward만 분할하므로 이 norm을 보존합니다.

DSM 평가는 원본과 같은 손실식/축약을 쓰지만, 평가 난수 bank와 집계 구현은 이 템플릿 방식입니다. 예전 runner의 평가 숫자와 bitwise 일치를 주장하지 않습니다. `FFHQ`처럼 별도 holdout이 없는 경우 결과를 `train_diagnostic`으로 명시합니다. 이를 검증 성능으로 해석하지 마십시오.

## 7. 선택 기능: CelebA / FFHQ, FID / IS

얼굴 reader는 원본 TFDS/TFRecord 전처리를 유지하며 TensorFlow를 선택 의존성으로 남겼습니다. CelebA는 TFDS의 **준비된** `celeb_a` 데이터와 공식 split이 필요하고, FFHQ는 원본 `ffhq-r08.tfrecords` 70,000개 레코드가 필요합니다. 원본 JPEG 폴더를 이 파일로 오인해 넣으면 안 됩니다.

```bash
uv sync --locked --extra faces
uv run --locked --extra faces python prepare.py -c configs/celeba64.json --set data.tfds_dir=/workspace/datasets/tfds
uv run --locked --extra faces python train.py -c configs/celeba64.json \
  --set data.tfds_dir=/workspace/datasets/tfds \
  --set name=celeba_mmse --set reference.mode=spectral_mmse

uv run --locked --extra faces python prepare.py -c configs/ffhq256.json \
  --set data.tfrecords_path=/workspace/datasets/ffhq/ffhq-r08.tfrecords
```

이 경로들을 계속 쓰려면 JSON에 저장하십시오. `prepare.py`에서 준 override는 원래 JSON 파일을 바꾸지 않으므로 `train.py`에도 같은 값을 주어야 합니다. `.data_loader.data_dir`은 MNIST/CIFAR 경로이며 얼굴 입력 경로를 바꾸는 옵션이 아닙니다.

기본 학습은 TensorFlow를 import하지 않습니다. FID/IS는 다음 선택 의존성을 사용합니다. **이 선택 기능은 번들 CPU 테스트에서 실행 검증되지 않았습니다.** 첫 사용 시 TF-Hub 모델 다운로드가 필요합니다. CIFAR10/CELEBA는 원본 TF-GAN Hub 모델, `--inception-v3`는 원본 FFHQ용 feature-vector 모델입니다. 두 backend의 특징 통계를 섞으면 안 됩니다.

```bash
uv sync --locked --extra metrics

# 예: 원본 CIFAR 프로토콜처럼 유효 배치 1024, forward만 32로 분할
uv run --locked python sample.py -r saved/cifar10_spectral_mmse/last.pt \
  -o saved/cifar10_spectral_mmse/samples50k \
  --num-samples 50000 --batch-size 1024 --model-batch-size 32 --steps 1000

# 준비된 real feature 파일의 key는 pool_3, shape는 [N,2048]
uv run --locked --extra metrics python metrics.py score --samples saved/cifar10_spectral_mmse/samples50k \
  --reference data/cifar10_stats.npz -o saved/cifar10_spectral_mmse/fid.json

# 필요하면 같은 backend로 실제 train 이미지에서 특징을 추출
uv run --locked --extra metrics python metrics.py extract-real -c configs/cifar10_paper.json -o data/cifar10_real.npz
```

`extract-real`은 새로운 reference 통계를 만드는 기능이며 원본 배포 통계와 같다고 자동 인증하지 않습니다. 직접 만든 통계에는 backend sidecar를 기록합니다. sidecar가 없는 외부 통계는 backend 검증 불가를 결과에 표시합니다. 수치를 비교하려면 데이터 split·전처리·양자화·특징 모델·sample count·sampler grid·유효 배치를 모두 맞추십시오. IS는 TF-GAN 전체 logits 방식이며 별도의 10-split 평균±표준편차를 출력하지 않습니다. 특징 배열은 RAM에 모으므로 50,000장 평가에는 충분한 시스템 메모리가 필요합니다.

## 유지한 것과 제외한 것

유지: NCSN++/DDPM++ backbone, Gaussian 7개 모드, VE/VP/sub-VP, DSM/SMLD/DDPM 손실, PC/ODE, Adam warmup/clipping, EMA, train-only FFT 통계, native FIR, 재시작 검사, 핵심 데이터셋 recipe.

제외: 과거 NCSN/NCSNv2 및 별도 legacy DDPM 모델, 데모 notebook/그림/실험 산출물, 중복 shell runner, CUDA JIT 소스, `absl`·`ml_collections` 중심 CLI, 모든 원본 config 조합, LSUN 전용 TF 데이터 파이프라인, likelihood/BPD·inpainting·colorization runner, 다중 GPU DDP. 데이터나 모델 가중치는 포함하지 않습니다. `image_folder`는 미리 전처리한 정확한 크기의 RGB 이미지를 읽는 확장점이지, 공식 LSUN 전처리 재현 구현은 아닙니다.

원본 코드 전체의 완전한 호환 레이어가 아니라, **다음 실험에서 수정할 파일을 찾기 쉽도록 핵심 경로를 남긴 템플릿**입니다. 새 기준항은 `model/gaussian.py`, 손실은 `model/loss.py`, 모델은 `model/backbones/`, 학습 흐름은 `trainer/trainer.py`에서 수정하십시오.


## uv 의존성 관리

`pyproject.toml`이 의존성 선언의 기준입니다. `requirements*.txt`는 이전 소스의 설치 안내와 호환되는 참고용 direct-dependency 스냅숏이며 uv는 읽지 않습니다. 새 의존성은 `uv add 패키지명`, 개발 의존성은 `uv add --dev 패키지명`, 삭제는 `uv remove 패키지명`으로 관리합니다. lockfile 생성·GPU/CPU 선택·선택 기능 실행·검증 한계는 `docs/UV.md`와 `docs/UV_VERIFICATION.md`에 정리했습니다.

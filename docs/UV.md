# uv 관리 안내

## 변경 범위와 현재 상태

이전 ZIP의 Python 파일과 실험 JSON은 변경하지 않았습니다. 모델·loss·sampler·FP32/TF32 설정·체크포인트 포맷도 그대로입니다. 새로운 것은 uv 프로젝트 선언과 설치 안내입니다. 모든 Python 파일의 해시를 비교한 결과는 `UV_VERIFICATION.md`에 기록합니다.

**uv.lock은 이 ZIP에 없습니다.** 현재 제작 환경은 PyPI 등 외부 패키지 서버의 DNS 조회가 실패하며 Python 3.11도 설치되어 있지 않습니다. 따라서 실제 `uv lock`, 새 가상환경 설치, CUDA/선택적 TensorFlow 실행은 완료하지 못했습니다. 직접 쓴 가짜 lockfile이나 기존 저장소의 다른 프로젝트 lockfile을 넣지 않았습니다. 첫 `uv sync`가 성공하면 해당 장비에서 실제 해결된 전이 의존성과 해시를 담은 `uv.lock`이 생성됩니다. 최초 성공 전에는 새 선언의 전체 의존성 해결이 검증됐다고 간주하지 마십시오.

## 처음 설치

압축 해제 후 `score_sde_template/`에서:

```bash
# Linux/macOS에서 uv가 아직 없다면
curl -LsSf https://astral.sh/uv/install.sh | sh
# 셸을 다시 열거나 source "$HOME/.local/bin/env"

uv python install 3.11
uv sync
uv run --locked python -m pytest -q
uv run --locked python train.py -c configs/smoke.json
```

Windows PowerShell의 uv 설치 방법:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.11
uv sync
uv run --locked python -m pytest -q
```

프로젝트가 flat script 구조이므로 `tool.uv.package=false`로 설정했습니다. `pip install -e .`나 배포용 wheel 빌드는 필요하지 않습니다. 실행은 프로젝트 루트에서 `uv run ...`으로 합니다.

첫 sync는 lockfile 생성과 설치를 함께 합니다. 이후 Git 사용 시:

```bash
git add pyproject.toml .python-version uv.lock
# 위 파일들을 커밋한 후 다른 장비에서는
uv sync --locked
uv run --locked python train.py -c configs/cifar10_paper.json \
  --set name=cifar10_spectral_mmse --set reference.mode=spectral_mmse
```

`.venv/`와 데이터·가중치는 커밋하지 않습니다. 이 bundle은 Python 3.11 계열만 허용합니다. patch 버전까지 고정하려면 검증한 3.11.x를 `uv python pin`으로 지정하십시오. lockfile은 NVIDIA 드라이버나 OS 라이브러리까지 고정하는 파일은 아닙니다.

## 선택 기능

아래 명령은 기본 `uv sync`가 먼저 성공하여 lockfile이 있는 상태를 전제로 합니다. `metrics` extra는 얼굴 데이터용 의존성도 포함합니다.

```bash
# CelebA / FFHQ: 원본 TFDS/TFRecord 전처리
uv sync --locked --extra faces
uv run --locked --extra faces python prepare.py -c configs/celeba64.json \
  --set data.tfds_dir=/workspace/datasets/tfds
uv run --locked --extra faces python train.py -c configs/celeba64.json \
  --set data.tfds_dir=/workspace/datasets/tfds \
  --set name=celeba_mmse --set reference.mode=spectral_mmse

# FID / IS
uv sync --locked --extra metrics
uv run --locked --extra metrics python metrics.py --help

# TensorBoard 기록
uv sync --locked --extra tensorboard
uv run --locked --extra tensorboard python train.py -c config.json \
  --set trainer.tensorboard=true
uv run --locked --extra tensorboard tensorboard --logdir saved

# 선택 기능을 모두 설치
uv sync --locked --all-extras
# 같은 선택 상태로 실행
uv run --locked --all-extras python metrics.py --help
```

sync뿐 아니라 `uv run`에도 필요한 extra를 명시했습니다. `uv run` 자체가 환경을 확인하므로 실행에 필요한 선택 기능을 생략하지 마십시오. 선택 상태를 바꾸지 않고 이미 준비된 가상환경을 그대로 쓰는 의도라면 `uv run --no-sync ...`도 가능하지만, 이 명령은 pyproject/환경의 최신 일치 상태를 검증하지 않습니다.

기본 dev 그룹은 pytest입니다. 테스트 도구를 제외할 때는 sync와 run 모두 `--no-dev`를 사용하십시오.

## 의존성 선택

- `torch==2.10.0`, `torchvision==0.25.0`: 기존 CPU 검증에 쓰인 버전 쌍을 고정했습니다. Linux/Windows는 명시적인 CUDA 12.8 인덱스입니다.
- NumPy `>=1.26.4,<2`: 기존 선택적 TF 2.15 스택과 일관되게 쓰기 위한 범위입니다. 첫 lock 후 정확한 버전은 `uv.lock`을 확인하십시오.
- `faces`: TensorFlow 2.15.1, TFDS 4.9.3, TF Metadata 1.15.0.
- `metrics`: faces 의존성 + TF-GAN 2.1.0, TF-Hub 0.15.0, TF Probability 0.23.0.
- `tensorboard`: TensorFlow 2.15와 같이 쓸 수 있도록 2.15 계열로 제한했습니다. 레거시 패키지의 pkg_resources 사용을 위해 선택 기능에 setuptools <81을 둡니다.

TensorFlow 배포는 `tensorflow==2.15.1` 한 가지를 사용합니다. `tensorflow-cpu`를 별도 추가하지 마십시오. 기존 코드에서 TensorFlow의 GPU를 숨기므로 얼굴 전처리/평가는 CPU로 처리하고 CUDA는 PyTorch에 남깁니다. 기본 학습에는 TensorFlow를 import하거나 설치하지 않습니다.

## PyTorch backend 변경

기본은 Linux x86-64와 Windows AMD64의 CUDA 12.8입니다. Apple Silicon macOS는 공식 PyPI 휠이며 이 코드의 device는 CPU입니다. 코드에 MPS/ROCm/DDP 지원을 추가한 것이 아닙니다. Intel macOS와 Linux ARM은 선언된 해석 대상에서 제외했습니다.

CPU-only Linux/Windows 환경에서는 `pyproject.toml`의 다음 **url 한 줄**을 수정합니다. 인덱스 이름 `pytorch`와 `tool.uv.sources`는 그대로 둡니다.

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

CUDA 13.0을 의도적으로 선택하려면 위 URL의 마지막 `cpu`/`cu128`을 `cu130`으로 바꾸십시오. 선택한 CUDA runtime을 지원하는 드라이버가 필요합니다. 이러한 변경 후에는:

```bash
uv lock
uv sync --locked
uv run --locked python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

이 방식은 backend 선택을 pyproject와 lockfile에 남깁니다. `uv pip install`로만 torch를 덮어쓰면 다음 sync에서 프로젝트 선언대로 돌아갈 수 있으므로 프로젝트의 인덱스를 수정하십시오. CPU와 GPU용 설정을 다르게 관리한 경우 서로 다른 lockfile을 같은 환경의 재현 근거로 취급하지 마십시오.

`CUDA_VISIBLE_DEVICES=0 uv run --locked ...`로 사용할 장치를 지정할 수 있습니다. CUDA 버전 선택은 TF32/AMP 설정을 바꾸지 않습니다. 기존 strict FP32 설정을 그대로 유지합니다.

## 패키지 추가·삭제·갱신

```bash
uv add 패키지명
uv add --dev 개발도구명
uv remove 패키지명
uv lock --check
uv tree
```

의도적인 전체 갱신은 `uv lock --upgrade` 이후 테스트하여 진행하십시오. `torch==...` 같은 exact pin은 이 명령만으로 바뀌지 않습니다. 직접 의존성을 변경해야 합니다. 장기 실험 도중 임의 업그레이드는 피하십시오.

`requirements*.txt`는 기존 설치 오류 메시지와 호환을 위해 유지한 pip용 직접 의존성 스냅숏입니다. uv의 입력은 아니며 CUDA 인덱스도 적용하지 않습니다. 필요할 때 lockfile에서 배포용 requirements를 **다른 경로**로 내보내려면:

```bash
uv export --locked --no-dev --format requirements-txt -o requirements-export.txt
uv export --locked --no-dev --extra metrics --format requirements-txt -o requirements-metrics-export.txt
```

## 공식 참고자료

- https://docs.astral.sh/uv/guides/projects/
- https://docs.astral.sh/uv/concepts/projects/dependencies/
- https://docs.astral.sh/uv/concepts/projects/sync/
- https://docs.astral.sh/uv/concepts/projects/config/
- https://docs.astral.sh/uv/guides/integration/pytorch/
- https://pytorch.org/get-started/previous-versions/

설치·lock 검증의 실제 성공/실패 기록은 `UV_VERIFICATION.md`를 우선 보십시오. 기존 `VERIFICATION.md`는 이전 ZIP 제작 시의 기록으로, uv로 만든 새 환경을 검증했다는 뜻이 아닙니다.

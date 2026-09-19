# MPS execution

설정은 `--device mps` 또는 `device=mps`입니다. `auto`는 CUDA, MPS, CPU 순으로
선택하고, 명시한 device가 없으면 CPU로 몰래 전환하지 않고 오류를 냅니다.

MPS 기본 `backend.spectral_transform=auto`는 **real separable DFT matmul**입니다.
FFT와 동일한 full orthonormal Fourier multiplier를 적용하지만, complex dtype을
생성하지 않으며 forward/backward의 신경망 및 Fourier 필터를 MPS에서 수행합니다.
Fourier 구조를 다른 기저로 바꾸는 것이 아닙니다. 통계 추정만 CPU float64로
진행하고 모델에 저장하는 통계·행렬·parameter는 FP32입니다.

CPU/CUDA의 auto는 `torch.fft.fft2/ifft2`입니다. 실수 DFT는 FFT보다 점근적인
계산량이 크며 고해상도에서 느릴 수 있습니다. 실제 speedup은 주장하지 않습니다.

명시적인 대안:

```bash
# 설치된 Mac에서 실제 전 경로 검사
uv run --locked python doctor.py --device mps

# Fourier 연산만 CPU에서 수행하고 differentiable copy로 연결
uv run --locked python doctor.py --device mps --spectral cpu
uv run --locked python train.py -c configs/mnist.json --device mps \
  --set backend.spectral_transform=cpu
```

CPU 경로에도 `detach()`를 넣지 않았습니다. 잔차 출력에 대한 gradient가 backbone까지
이어집니다. MPS에서 native FFT를 시험하려면 `--spectral fft`를 사용할 수 있지만
그 경로는 backend 지원을 사용자가 실제로 검사해야 합니다. 실패 시 자동 fallback은
하지 않습니다. 모든 run에 선택한 backend를 기록하고 동일 비교군에서 통일하십시오.

MPS 호환을 위해 source FIR의 6D zero-insertion padding을 4D 연산으로 바꾸었습니다.
filter·conv·parameter 구조는 같습니다. CPU 독립 oracle과의 forward/backward
동등성 테스트를 포함했습니다. 모델 생성 시 source의 float64 sigma buffer는 CPU에서
먼저 FP32로 변환한 뒤 device로 옮깁니다. MPS RNG도 checkpoint에 저장·복원합니다.

MPS availability와 OS 요구사항은 현재 설치한 torch의 공식 문서를 확인하십시오.
2026-09-19 확인한 PyTorch 2.14 문서는 macOS 14.0+와 MPS-enabled device 여부를
확인합니다: https://docs.pytorch.org/docs/2.14/notes/mps.html

**제작 환경에는 MPS 하드웨어가 없습니다.** real DFT의 FFT 일치, gradient,
신경망·학습·샘플링은 CPU에서 검사했지만 실제 Mac에서 성공했다는 뜻은 아닙니다.
`tests/test_devices.py`의 MPS 테스트와 `doctor.py --device mps`로 현장에서 검증하십시오.
최근 torch wheel이 제공되지 않는 Intel Mac에 대해 native wheel 설치를 보장하지 않습니다.

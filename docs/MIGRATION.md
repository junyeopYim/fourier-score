# 변경 범위와 호환성

## 원본 → 새 위치

| 원본 | 새 위치 |
|---|---|
| `mnist_compare/core.py` | `model/gaussian.py` |
| `models/ncsnpp.py` | `model/backbones/ncsnpp.py` |
| `models/layers.py`, `layerspp.py` | `model/backbones/layers.py`, `layerspp.py` |
| `models/up_or_down_sampling.py`, `op/upfirdn2d.py` native path | `model/backbones/up_or_down_sampling.py` |
| `models/utils.py` score adapters | `model/model.py` |
| `models/ema.py` | `utils/ema.py` |
| `losses.py` | `model/loss.py`, optimizer lifecycle는 `base/base_trainer.py` |
| `sde_lib.py`, `sampling.py` | `sde/sde_lib.py`, `sde/sampling.py` |
| `mnist_compare/run.py` | `prepare.py`, `train.py`, `test.py`, `sample.py`, `trainer/`, `data_loader/`, `utils/` |
| `mnist_compare/face_run.py` discrete wrapper | `model/model.py` |
| `mnist_compare/face_data.py` | `data_loader/face_data.py` |
| `evaluation.py`, CIFAR evaluator의 핵심 feature 경로 | 선택 기능 `metrics.py` |
| Python ConfigDict + argparse | JSON + `parse_config.ConfigParser` |

NCSN++의 모듈 순서와 parameter 이름은 가능한 그대로 유지했습니다. 하지만 전체 체크포인트 및 옵션 구조는 바뀌었으므로 예전 checkpoint 직접 resume는 지원하지 않습니다. 이 아카이브에는 자동 변환기를 넣지 않았습니다.

## 수학적 정의를 유지한 부분

- FFT는 full orthonormal `fft2(..., norm="ortho")`입니다. 실수 영상의 켤레 대칭을 유지하고, RGB 교차 채널 공분산을 새로 도입하지 않았습니다.
- 통계는 training split만 사용한 float64 sufficient statistics입니다. covariance floor=1e-4 기본값을 유지합니다.
- `s_G(x,sigma)=-(Sigma+sigma^2 I)^{-1}(x-m)`와 source residual score convention을 유지합니다.
- MMSE gate는 `sum(P*sigma^2/(P+sigma^2))/sum(P)`입니다. 정규화 분모, floored power, scalar weighting, chunked exact sum을 유지합니다.
- SMLD는 descending sigma에 대한 integer label과 실제 sigma를 구분합니다.
- MNIST의 pixel-mean과 CIFAR/face의 half-pixel-sum 손실 축약을 구분합니다.
- 원본 Adam beta2=0.999, step 증가 **전** warmup, gradient clipping, EMA update-count warmup을 유지합니다.
- PC는 corrector 후 predictor입니다. Langevin의 배치 평균 norm은 그대로이며, 모델 forward 분할은 유효 배치를 바꾸지 않습니다.

## 의도적으로 달라진 구현 / 제한

1. 모든 실행을 standalone 템플릿으로 옮겼습니다. `git rev-parse`, 원본 repo 폴더 검사, import-time CUDA 빌드를 제거했습니다.
2. 학습/평가/샘플링을 독립 entry point로 분리했습니다. 학습 중 자동 샘플 생성은 하지 않습니다. JSONL과 선택적 TensorBoard를 사용합니다.
3. RNG 저장은 Python/NumPy까지 확대했습니다. 평가 RNG는 격리합니다. 원본 MNIST/CIFAR의 CPU noise-bank별 10-bin 진단 대신 같은 DSM 식의 전체 평균을 제공합니다. 따라서 원본 runner와 평가 난수/집계의 완전 동일성은 보장하지 않습니다.
4. 전체 모델을 float32로 이동해 positional sigma buffer의 원본 float64 승격 문제를 방지합니다. 얼굴 runner에 있던 방식을 공통 모델 생성에 적용한 것입니다.
5. native resampling만 남겼습니다. 원본 native 경로의 연산식을 사용합니다. custom CUDA extension의 속도/bitwise 결과를 주장하지 않습니다.
6. 원본의 드물게 사용되는 `upsample_conv_2d` 분기에서 음수 tensor slicing과 잘못된 stride 차원을 수정했습니다. `torch.flip`, 2차원 stride와 grouped conv_transpose를 사용합니다. 원본의 잘못된 nearest interpolate 위치 인자도 키워드 인자로 수정했습니다. bias 없는 conv의 초기화도 조건부로 바꾸었습니다.
7. probability-flow Euler의 scalar zero diffusion을 처리했습니다. ancestral VE sigma grid의 device 이동을 보완했습니다. sub-VP에 없는 alpha grid를 요구하는 corrector 조합은 명시적으로 거부합니다.
8. EMA 로드에 길이/shape 검사를 추가하고 현재 device/dtype으로 이동합니다. gradient/샘플의 비유한 값은 중단합니다. ODE solver 실패도 보고합니다. 모델 생성·손실식의 다른 수학적 변경을 의도하지 않았습니다.
9. NFE는 실제 score 호출을 셉니다. 원본의 `N*(n_steps+1)`은 none predictor/corrector 등에서 실제 호출 수와 다를 수 있습니다.
10. CIFAR `upstream`과 `paper`를 별도 프리셋으로 구분했습니다. 평가/미리보기 배치는 CLI로 따로 관리합니다. 원본 eval 구조를 그대로 복제한 config는 아닙니다.
11. checkpoint/statistics 포맷은 template-v1입니다. 새 포맷 전용 제한적 `weights_only=True` 로드를 사용합니다. 레거시 NumPy pickle allowlist를 자동으로 추가하지 않습니다.
12. TensorFlow face/metric integration은 읽고 이식했으나 이 환경에서는 실행하지 못했습니다. GPU/전체 데이터 장기 학습/논문 FID 재현 역시 검증 범위 밖입니다.

## 데이터와 재시작 주의

MNIST/CIFAR는 raw uint8 데이터 내용의 SHA-256으로 확인합니다. 얼굴/이미지 폴더 fingerprint는 원본 방식처럼 경로·파일 크기·mtime 및 전처리 metadata에 기반합니다. TFRecord reader는 framing/길이/개수를 확인하지만 CRC 검증이나 레코드 전체 내용의 암호학적 검증을 수행하지 않습니다.

통계 캐시를 준비한 후 병렬 비교군을 시작하십시오. 동일 실험 디렉터리에 두 프로세스가 동시에 쓰는 것은 지원하지 않습니다. 정상 update 경계에서만 주기적인 last.pt를 저장합니다. 중단 순간의 부분 gradient를 checkpoint로 저장하지 않으며, 비정상 종료 시 마지막 완료 저장 지점으로 돌아갑니다.

서로 다른 장비·torch/CUDA/cuDNN 버전까지 학습 결과가 같다고 보장하지 않습니다. 이식 이후 실험은 별도 experiment series로 관리하고, 원본과의 성능 비교는 별도의 수치 실험으로 확인해야 합니다.


## uv 관리 전환 (2026-09-18)

Python 소스와 실험 JSON을 그대로 유지하고 pyproject.toml 프로젝트 메타데이터·선택 의존성·dev 그룹·명시적인 PyTorch 인덱스를 추가했습니다. 실행 안내를 uv로 변경했습니다. uv.lock은 제작 환경의 네트워크 제한으로 미포함이며 첫 uv sync 성공 시 생성합니다. docs/UV.md와 docs/UV_VERIFICATION.md를 참고하십시오.

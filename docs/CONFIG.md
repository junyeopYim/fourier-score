# 하나의 설정 체계

`configs/base.json`이 전체 schema와 기본값의 단일 출처입니다. 나머지 JSON은
`extends`로 필요한 부분만 변경합니다. `extends` 경로는 해당 파일 기준이며,
데이터와 저장 경로는 실행 working directory 기준입니다. 프로젝트 root에서
명령을 실행하십시오.

```bash
uv run --locked python train.py -c configs/cifar10.json --dry-run
uv run --locked python train.py -c configs/cifar10.json --dry-run \
  --set loss.type=score --set trainer.microbatch_size=32
```

미등록 key, 오탈자, 잘못된 type, 음수 learning rate, 비양수 sigma,
맞지 않는 sampler/process, image size와 architecture depth 불일치, 지원하지
않는 attention resolution을 거부합니다. JSON list override는 shell에서
따옴표로 묶으십시오: `--set 'arch.args.ch_mult=[1,2,2,2]'`.

| 범주 | 단일 설정 위치 |
|---|---|
| 데이터·image shape·전체 배치·worker | `data_loader.args.*` |
| NCSN++ 학습 레이어 구조 | `arch.args.*` |
| sigma/beta schedule·forward process | `process.*` |
| 출력 재매개화·공통 DSM reduction | `loss.*` |
| Gaussian 통계 floor/cache | `fourier.*` |
| Adam | `optimizer.args.*` |
| update 수·warmup·clip·EMA·microbatch | `trainer.*` |
| device | `device` 또는 동일 필드를 설정하는 `--device` |
| FP32/TF32·MPS Fourier implementation | `backend.*` |
| sample grid·batch·seed·sampler | `sampling.*` |
| 검증 batch·seed·noise bins·frequency bins | `evaluation.*` |

image_size, channels, sigma schedule을 model과 dataset 양쪽에 중복해서 지정하지
않습니다. legacy NCSN++가 요구하는 configuration object는 `backbone_config()`가
위의 단일 필드에서 생성합니다. sigma division은 objective adapter가 담당하며
사용자가 architecture option으로 다시 활성화하지 못하게 했습니다.

`loss.type` 변경은 arch / process / sampler를 변경하지 않습니다. DDPM으로
forward를 바꾸려면 process와 sampler가 일치하는 `_ddpm.json`을 선택합니다.
출력 재매개화 비교에서는 같은 데이터셋 JSON에서 `loss.type`만 바꾸십시오.

`loss.type`: `score`, `diffusion`, `scalar_gaussian`, `fourier_gaussian_unscaled`,
`fourier_gaussian`. 모두 같은 DSM 목적함수를 사용합니다. Gaussian 대조군은
`fourier`의 동일 통계 cache와 평균 이미지를 공유합니다.

`evaluation.frequency_bins`는 기본값 0으로 진단을 끕니다. 양수로 지정하면
노이즈×방사형 주파수 대역 DSM과 대역 경계를 기록합니다. 추론 시에도
`test.py --set evaluation.frequency_bins=6`으로 켤 수 있습니다.
`configs/cifar10_ablation.json`은 50,000 update와 6개 주파수 대역을 사용하는
탐색용 프리셋이며, 수렴 또는 논문 품질을 보장하는 설정이 아닙니다.

## 학습 상태 표시

`trainer.console=human`이 기본값입니다. stderr에 준비 단계, 평가 이미지 수,
학습 진행률·loss·속도·ETA, checkpoint 저장 시작/완료와 경로를 표시합니다.
터미널에서는 진행 행을 갱신하고, 리다이렉션/`tee`에서는 제어 문자 없는 줄을
출력합니다. 첫 update를 즉시 표시하며, 이후 `trainer.progress_every_seconds`
(기본 5초)마다 update/평가 batch가 끝나는 시점에 갱신합니다. 하나의 update가
오래 걸리면 그 update가 완료될 때까지 기다립니다.

`trainer.log_every`는 기존처럼 상세 JSONL/TensorBoard 기록 주기이며,
콘솔 갱신 주기와 독립적입니다. 노이즈·주파수별 배열을 포함한 평가 결과는
항상 run directory의 `metrics.jsonl`에 기록하고 콘솔에는 DSM 요약만 표시합니다.

- `loss`: 마지막 update의 loss. `avg` / `loss_avg`: 직전 JSONL train 기록 이후
  실제 이미지 수로 가중 평균한 loss (`window_steps`, `window_images`도 기록).
- `pixel(avg)` / `loss_pixel_mean_avg`: 같은 평균을 pixel mean으로 환산한 값.
  `half_sum`이면 `2 / (channels × height × width)`를 곱합니다. 평가와 단위는
  같지만 평가에는 EMA 가중치와 별도 이미지·noise를 사용합니다.
- `it/s`, `images_per_second`: 해당 기록 구간에서 데이터 대기와 학습에 걸린
  시간 기준입니다. `ETA(train)`은 이번 세션의 평균 update 시간으로 추정한
  **남은 학습 시간**이며 향후 평가·저장 시간은 제외합니다.
- CUDA에서는 PyTorch가 현재 할당/예약한 GiB도 표시합니다. GPU 전체 사용량은
  아닙니다. CPU/MPS에서는 이 필드를 생략합니다.

```bash
# 콘솔 갱신을 1초 간격으로; 상세 파일 기록 주기는 유지
uv run --locked python train.py -c configs/cifar10_ablation.json \
  --set trainer.progress_every_seconds=1

# 이전처럼 각 상세 JSON record를 stdout에도 출력
uv run --locked python train.py -c configs/cifar10_ablation.json \
  --set trainer.console=json
```

`trainer.console=quiet`는 콘솔만 끕니다. JSONL과 선택 TensorBoard 기록은
유지합니다. 사람이 읽는 로그를 파일에도 남기려면 `2>&1 | tee train.log`를
사용하십시오. TensorBoard는 선택 dependency와 기록 옵션을 함께 켭니다.

```bash
uv run --locked --extra tensorboard python train.py -c configs/cifar10_ablation.json \
  --set trainer.tensorboard=true
uv run --locked --extra tensorboard tensorboard --logdir saved
```

콘솔 설정은 같은 코드 버전의 checkpoint resume에서 변경할 수 있습니다.
단, 이 패치도 학습 소스 hash를 바꾸므로 **패치 이전 checkpoint의 학습 resume는
기존 source 검사에 의해 거부**됩니다. 진행 중인 실험을 재개하려면 해당 코드
버전을 유지하고, 새 표시 기능은 새 run에서 사용하십시오.

## Output naming

`name=auto`이면 `{dataset}_{process}_{loss}_s{seed}`를 사용합니다. 고정 이름을
설정했다면 loss를 바꿀 때 이름도 직접 구분해야 합니다. 동일 directory는
덮어쓰지 않고 실패합니다. image_folder를 여러 데이터셋에 쓸 때는 이름을
명시적으로 구분하십시오. Compare script는 custom name 뒤에 `{loss}_s{seed}`를 붙입니다.
기본 네 비교군은 `--objectives`로 선택할 수 있으며 `--seeds 42 43 44`는
각 seed에 대해 동일 설정의 모든 비교군을 실행합니다. `--dry-run`은 명령만
출력하며 데이터 로딩·통계 추정·학습·디렉터리 생성을 하지 않습니다.

## Resume

`train.py -r last.pt`는 저장된 완전한 config를 사용합니다. `-c` 동시 사용을
거부합니다. update 수와 logging 주기, data root와 worker 수 등 허용 항목만
변경할 수 있습니다. 데이터 fingerprint, split, source hash, objective,
architecture, optimizer, microbatch, backend는 확인합니다. device type/index와
PyTorch 버전이 다르면 엄격한 학습 resume를 거부합니다. 추론은 device 변경을
허용하고 새 환경을 기록합니다.

주기적 평가와 checkpoint 저장은 RNG를 바꾸지 않도록 설계되어 있습니다.
체크포인트는 worker가 prefetch한 위치가 아니라 **학습에 실제 소비한 배치**를
기록합니다. preprocessing은 worker에서 결정적이고 random flip은 trainer가
독립 generator로 처리합니다. worker는 spawn으로 실행합니다.

중간 KeyboardInterrupt 직전의 미완료 optimizer step은 저장하지 않습니다.
최근 정상 저장 checkpoint에서 다시 시작하십시오. 다른 hardware의 bitwise
동일 결과, 훈련 중 코드 변경, 기존 레포 checkpoint의 자동 변환은 보장하지 않습니다.

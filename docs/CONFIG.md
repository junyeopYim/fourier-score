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
| objective·loss reduction | `loss.*` |
| Gaussian 통계 floor/cache | `fourier.*` |
| Adam | `optimizer.args.*` |
| update 수·warmup·clip·EMA·microbatch | `trainer.*` |
| device | `device` 또는 동일 필드를 설정하는 `--device` |
| FP32/TF32·MPS Fourier implementation | `backend.*` |
| sample grid·batch·seed·sampler | `sampling.*` |
| 검증 batch·seed·noise bins | `evaluation.*` |

image_size, channels, sigma schedule을 model과 dataset 양쪽에 중복해서 지정하지
않습니다. legacy NCSN++가 요구하는 configuration object는 `backbone_config()`가
위의 단일 필드에서 생성합니다. sigma division은 objective adapter가 담당하며
사용자가 architecture option으로 다시 활성화하지 못하게 했습니다.

`loss.type` 변경은 arch / process / sampler를 변경하지 않습니다. DDPM으로
forward를 바꾸려면 process와 sampler가 일치하는 `_ddpm.json`을 선택합니다.
진짜 loss-only 비교에서는 같은 데이터셋 JSON에서 `loss.type`만 바꾸십시오.

## Output naming

`name=auto`이면 `{dataset}_{process}_{loss}_s{seed}`를 사용합니다. 고정 이름을
설정했다면 loss를 바꿀 때 이름도 직접 구분해야 합니다. 동일 directory는
덮어쓰지 않고 실패합니다. image_folder를 여러 데이터셋에 쓸 때는 이름을
명시적으로 구분하십시오. Compare script는 custom name 뒤에 loss 이름을 붙입니다.

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

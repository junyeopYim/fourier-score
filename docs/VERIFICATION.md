> 이 문서는 이전 ZIP의 CPU 검증 기록입니다. uv 전환본의 환경·설치 검증 상태는 `UV_VERIFICATION.md`를 참고하십시오.

# 검증 기록

## 실행 환경과 결과

- 실행일: 2026-09-18
- 환경: CPU-only Linux; 패키지의 실제 버전은 `TEST_ENVIRONMENT.json` 참고
- torch 2.10.0+cpu / torchvision 0.25.0+cpu
- NumPy 2.3.5 / SciPy 1.17.0 / Pillow 12.3.0 / pytest 9.0.2
- `python -m compileall -q ...`: 성공
- `python -m pytest -q`: **62 passed**, 실패/skip 없음
- `train.py` 합성 입력 + 실제 축소 NCSN++ 3 updates: 성공
- `train.py --resume`으로 4번째 update 실행: 성공
- `sample.py` EMA 로드, 유효 배치 2 / forward 배치 1로 정확히 3장 생성: 성공
- NPZ dtype=uint8, NHWC shape=[N,32,32,1], 완료 manifest: 확인
- `test.py` EMA DSM: 학습 종료 직후 기록과 동일한 0.42455875873565674
- `metrics.py --help`: TensorFlow 설치 없이 성공
- FFHQ 설정 `--dry-run`: TensorFlow 데이터 접근이나 큰 모델 생성 없이 성공

실제 명령 출력은 `verification_pytest.txt`, `verification_cli.txt`에 들어 있습니다. 출력 중 `/mnt/data/...`는 제작 환경의 임시 검사 경로이며 실행 설정의 기본 경로가 아닙니다.

## 자동 테스트의 범위

- MNIST/CIFAR 전처리 좌표와 shape
- float64 population moments, full FFT power와 Parseval 관계
- 가우시안 spectral score와 작은 dense covariance inverse의 일치
- 7개 reference 모드, x/sigma 미분, 학습 parameter 추가 없음
- MMSE closed form, 단조성, 끝값, chunked exact sum
- 선형 시간 gate
- continuous DSM의 두 reduction 및 likelihood weighting을 직접 계산과 비교
- discrete SMLD 손실의 직접 계산 비교
- VE/VP/sub-VP reverse drift 공식
- NCSN++의 3×3 progressive/input 조합 forward/backward
- 연속/이산 VE 학습 및 샘플링, VP/sub-VP 학습·샘플링
- native FIR up/down/conv의 shape·gradient, probability-flow Euler
- forward microbatch를 쓰면서 유효 Langevin 배치를 유지하는 검사
- 작은 문제에서 probability-flow ODE 유한성
- 모든 제공 설정 로드, 중요 recipe 값, 잘못된 설정 및 순환 상속 거부
- train-only split, exact flip mixture, stats identity 확인
- batch permutation의 정확한 재시작
- 중단 없이 4 updates vs 2 updates + resume 2 updates를 비교:
  모델, Adam state, EMA, 배치 스트림, RNG 상태가 `torch.equal` 기준 일치
- 위 재시작 검사를 microbatch 없음 / microbatch=1 두 조건에서 수행
- 덮어쓰기 방지, 다른 reference 모드의 resume 거부
- last.pt의 EMA 추론과 경량 EMA snapshot의 일치
- EMA context 예외 복구, 평가 RNG 격리
- 잘린 TFRecord header 거부 (TensorFlow decode 테스트는 아님)

## 검증하지 않은 것

**GPU 학습, CUDA 속도/메모리, 실제 전체 데이터셋 학습, 기존 runner와 전체 numerical parity, 논문 FID/IS 재현은 확인하지 않았습니다.**

TensorFlow, TFDS, TF-GAN 및 TF-Hub 모델은 이 실행 환경에 설치되지 않았습니다. 얼굴 전처리와 선택적 FID/IS 코드는 소스에서 이식했지만, 해당 통합 경로는 실제 데이터/모델로 실행 검증하지 않았습니다. 이 선택 기능이 학습용 기본 의존성 없이도 import되지 않는다는 점 및 CLI help 정도만 확인했습니다.

기존 원본 체크포인트의 자동 변환/재시작도 검증 대상이 아니며 지원하지 않습니다. 원본과 동일한 seed·설정만 주면 학습 결과가 그대로 재현된다는 의미가 아닙니다. 특히 평가 난수 생성과 실행 framework는 달라졌습니다.

이 검사는 템플릿의 실행 가능성과 명시한 핵심 수식/상태 관리에 대한 제한된 근거입니다. 새로운 GPU 환경에서 장기 실험을 시작하기 전 먼저 소규모 실행을 확인하십시오.

# uv 전환 검증 기록

날짜: 2026-09-18

## 완료한 검사

- 원본 ZIP과 비교하여 **Python 파일 37개, 실험 설정 JSON 8개가 바이트 단위로 동일**합니다.
- `pyproject.toml`은 Python의 TOML parser로 파싱했으며 각 의존성은 `packaging.Requirement`, 각 환경 조건은 `packaging.Marker`로 검사했습니다.
- 모든 Python 소스는 AST 구문 검사에 통과했습니다.
- 기존 실행 환경에서 `python -m pytest -q`: **62 passed in 16.95s**.
- 기존 실행 환경에서 `train.py --help`, CIFAR-10 설정 `--dry-run`, `metrics.py --help`: 종료 코드 0.
- Python 파일 전체가 동일하므로 기존 코드 해시 계산에 쓰이는 Python 소스 집합도 유지했습니다.

## 설치·lockfile 상태: 미완료

- 이 환경에는 Python 3.13.5만 있으며 프로젝트에 지정한 Python 3.11이 없습니다.
- `uv lock --offline --no-python-downloads`는 Python 3.11 interpreter 부재로 실패했습니다.
- 외부 패키지 서버 DNS 조회 결과: `gaierror: [Errno -3] Temporary failure in name resolution`.
- `uv pip compile pyproject.toml --python-version 3.11 --all-extras --offline`은 캐시에 필요한 torch 배포본이 없어 실패했습니다. 이 **offline 실패는 온라인 resolver에서 실제 의존성이 서로 충돌한다는 판정이 아닙니다**. 반대로 온라인 해석 성공을 확인한 것도 아닙니다.
- 실제 `uv sync`로 새 가상환경에 설치한 검사는 수행하지 못했습니다.
- 검증된 `uv.lock`을 생성하지 못했으므로 ZIP에는 넣지 않았습니다. 첫 `uv sync`에서 생성하십시오.

## 테스트 환경 구분

이번 62개 테스트는 **기존 CPU Python 3.13.5 / torch 2.10.0+cpu / torchvision 0.25.0+cpu / NumPy 2.3.5** 환경에서 실행했습니다. 새 프로젝트 선언의 Python 3.11 / NumPy <2 / CUDA 12.8 환경에서 실행한 결과가 아닙니다. 정확한 현재 패키지 버전은 `UV_TEST_ENVIRONMENT.json`에 있습니다.

GPU, 새 dependency graph, 얼굴 TFRecord/TFDS 실제 입력, TF-GAN/TF-Hub FID/IS 실행은 검증하지 못했습니다. 이번 변경이 수치 결과의 동일성이나 CUDA 성능을 검증한다는 뜻이 아닙니다.

## 원본 기록

- `uv_verification_pytest.txt`: 이번 테스트 출력
- `uv_verification_commands.json`: uv 명령 실패와 CLI 검사 성공의 실제 출력
- `uv_code_unchanged.json`: 동일한 Python/설정 파일의 SHA-256 목록
- `UV_TEST_ENVIRONMENT.json`: 이번 실행 환경
- 이전 `VERIFICATION.md`와 `verification_*.txt`: 이전 ZIP의 기록이며 새로운 uv 설치 결과가 아님

## 사용자 환경에서 첫 검사

```bash
uv python install 3.11
uv sync
uv lock --check
uv run --locked python -m pytest -q
uv run --locked python train.py -c configs/smoke.json
```

모두 성공한 뒤 생성된 `uv.lock`을 Git에 커밋하고 동일 실험군에서 공유하십시오. 최초 설치부터 `--locked`를 주면 lockfile이 아직 없으므로 실패합니다.

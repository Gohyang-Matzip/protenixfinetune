# 검증 기록

검증일: 2026-09-17. 프로그램 버전: 0.2.0. 로컬 macOS에서 실행했습니다. 사용자 요청대로 Protenix·의존성·모델 weight를 설치하거나 내려받지 않았습니다. 공개 데이터의 명시적 API 수집만 수행했습니다.

## 실행한 검사

| 실행 | 결과 |
|---|---|
| `python3 -m unittest discover -s tests -v` | 22개 중 13개 통과, Torch 의존 검사 9개 skip |
| `/Users/donghanlee/.venv/bin/python -m unittest discover -s tests -v` | **22개 모두 통과**, CPU |
| CLI help → CSV import → 세 가지 split → audit → inputs | 모두 exit 0 |
| 두 부분 manifest merge | 원래 관측과 동일한 round-trip 확인 |
| 같은 import 출력 경로 재사용 | exit 2, 기존 파일 보존 |
| 실제 PubChem AID 2244 + CID→SMILES 수집 | description/CSV/property JSON 수집 성공, 영수증 3개의 SHA256 재확인 |
| Native `prepare`를 Protenix 없는 Python에서 호출 | 명시적 의존성 오류, exit 2, 출력 디렉터리 생성 없음 |
| Python 구문·예제 JSON·현재 문서 내부 링크 | 확인 완료 |

환경은 표준 Python 3.14.6(Torch 없음)과 기존 가상환경 Python 3.13.12/Torch 2.11.0입니다. 두 환경 모두 Protenix가 없으며 CUDA는 사용할 수 없습니다. 기존 Torch 환경에는 NumPy가 없어 초기화 경고가 발생하지만 테스트는 통과했습니다. 이 검사를 위해 패키지를 설치하지 않았습니다.

## 학습 테스트가 확인한 것

- 화학/표적/both 분할의 결정성과 누출 차단, off-diagonal 제외.
- QC fail, uncertain, phenotypic 및 test 관측이 학습/validation에 섞이지 않음.
- Task별 출력 gradient와 서로 다른 task 집합 사이의 공유 hidden 전이.
- Joint에서 학습한 backbone을 다음 frozen 단계로 전달한 뒤 저장·복원해도 값 보존.
- 실제 작은 CPU 학습 루프와 checkpoint 재개의 최종 parameter 동등성.
- 양성 구조와 synthetic 분기, apo 매핑의 일대일 정수 제약, 인공 ligand 강체 배치의 내부 거리 보존.
- Head/eval의 stochastic encoder 재현성, assay별 지표 분리, 동일 평가 관측만 비교.
- 검토한 LIT outcome 유지, CSV 중복 header 거부, mapping별 관측 ID 구분.
- 다운로드의 SHA256·Content-Length 오류 및 기존 파일 덮어쓰기 차단.

검토 중 발견한 task 불일치 전이 차단, LIT label 강제변환, 불완전 다운로드의 성공 처리, CSV 중복 header 덮어쓰기, 서로 다른 assay import ID 충돌을 수정하고 회귀 검사를 추가했습니다. 독립 코드 검토에서도 해당 다섯 수정의 동작을 재확인했습니다.

## CLI 예제 결과

[가상 원본](../examples/general_screen.csv) 147개 관측은 6개 표적 × 12개 화합물 × 2개 task의 144개 binary 관측에 실패·불확실·phenotypic 각 1개를 더한 것입니다. 물리적/실험적 benchmark가 아닙니다.

`--strategy both --seed 42` 결과:

| split | 전체 관측 | 학습/지표에 적격인 관측 |
|---|---:|---:|
| train | 67 | 64 |
| val | 4 | 4 |
| test | 4 | 4 |
| excluded | 72 | 0 |

두 task 모두 train/val/test에 양성과 음성이 있습니다. Inputs JSON은 제외 교차쌍과 phenotype을 빼고 74개를 생성합니다. QC fail/uncertain도 예측 입력은 만들 수 있으므로 적격 관측 72개보다 2개 많습니다. 입력 JSON에는 label이 없습니다.

예제 출력은 로컬 `data/demo_import`, `data/demo_chemistry.csv`, `data/demo_target.csv`, `data/demo_both.csv`, `data/demo_*.audit.json`, `data/demo_inputs.json`에 보존했습니다. `data/`는 gitignore 대상입니다. 같은 명령을 다시 실행할 때는 새 출력 이름을 사용하세요.

## 공개 API smoke의 의미

실행한 명령:

```bash
python3 -m fragment_ft fetch-pubchem --aid 2244 --with-compounds \
  --output data/smoke/pubchem_2244_20260917
```

AID 2244는 **세포 기반 독성 assay**이며 API/파일 형식 검사에만 사용했습니다. Binding 학습 데이터로 편입하지 않았고 label을 curate하지 않았습니다. 실제 생성된 receipt의 파일 hash만 재검증했으며 이 작은 수집 성공으로 공개 corpus 전체의 품질·완전성·학습 개선을 주장하지 않습니다.

## 아직 검증하지 않은 것

실제 Protenix checkpoint 로딩과 feature API 실행, 실제 복합체/apo의 원자·화학적 대응, native 구조 loss·pose 품질, CUDA/BF16, NCCL·다중 GPU/다중 node, 실제 공개 corpus 성능은 미검증입니다. H100 사용 및 성능 개선 수치를 보고한 상태가 아닙니다. 실행 방법은 [학습 문서](TRAINING.md)에 있고, 실제 환경 검증 순서는 [연구계획](../FINETUNING_PLAN.md)에 있습니다.

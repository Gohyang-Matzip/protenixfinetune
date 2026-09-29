# 검증 기록

검증일: 2026-09-29(이전 기록 2026-09-28, 2026-09-17). 프로그램 버전: 0.2.0. 로컬 macOS에서 실행했습니다. Protenix와 모델 weight는 설치하거나 내려받지 않았습니다. 저장소 밖 임시 가상환경 `/tmp/protenix-p0-validation`에만 Torch, NumPy, RDKit를 설치해 테스트했습니다. 공개 데이터의 명시적 API 수집은 2026-09-17에만 수행했습니다. 이 문서가 테스트 수와 남은 검증 항목의 기준입니다.

## 실행한 검사

| 실행 | 결과 |
|---|---|
| `python3 -m unittest discover -s tests -v` (Torch·RDKit 없음) | 114개 중 84개 통과, Torch 의존 검사 27개와 RDKit 검사 3개 skip; 3.148초 |
| `/tmp/protenix-p0-validation/bin/python -m unittest discover -s tests -v` | **114개 모두 통과**, skip 0, CPU; 5.435초 |
| 메인 baseline CLI 연결 회귀 검사(`-p test_baselines.py`) | 15개 통과; 시스템 Python은 13개 통과 / RDKit 2개 skip |
| 별도 프로세스 `train --embeddings` → `predict --embeddings` → baseline 5개 → `compare` | exit 0; mock encoder의 두 디스크 shard로 CPU 학습·예측, prior·ligand·가상 ipTM/PAE/pLDDT와 6개 report 비교 |
| `compileall -q fragment_ft tests`, `git diff --check`, baseline/embed help | exit 0; 별도 빌드 단계 없음 |
| LSP diagnostics | basedpyright-langserver 미설치로 실행 불가; 구문 컴파일은 통과 |
| 새 가상환경에서 `pip install .`(기본 pip 21.2.4)과 pip 업그레이드 후 `pip install -e .` | 둘 다 RDKit 2025.09.2가 함께 설치되고 `fragment-ft audit`가 `molecule_identity.checked: true` 보고. 기본 pip 21.2.4는 `-e` 설치를 거부함 |
| RDKit 환경에서 세 demo split의 `audit` | 12개 분자 모두 InChIKey 검사 통과(`molecule_identity.checked: true`) |
| CLI help → CSV import → 세 가지 split → audit → inputs | 모두 exit 0, 아래 결과가 저장소의 demo 출력과 일치 |
| 두 부분 manifest merge | 원래 manifest와 byte 단위로 동일 |
| `audit --against` 참조 manifest | exit 0, `cross_corpus` 보고 |
| 같은 import 출력 경로 재사용 | exit 2, 기존 파일 보존 |
| Native `prepare`를 Protenix 없는 Python에서 호출 | 명시적 의존성 오류, exit 2, 출력 디렉터리 생성 없음 |
| 예제 JSON, 보관 문서를 포함한 모든 문서 내부 링크 | 확인 완료 |

두 환경 모두 Python 3.9.6입니다. 임시 가상환경은 Torch 2.8.0(CPU), NumPy 2.0.2, RDKit 2025.09.2이며 git이 있어 Protenix commit 기록 테스트도 실행했습니다. 두 환경 모두 Protenix가 없고 CUDA를 사용할 수 없습니다. RDKit 분자 동일성 검사는 stub 테스트와 함께 실제 RDKit로도 실행했습니다(Kekulé와 aromatic 표기의 aspirin이 같은 InChIKey로 묶임). Protenix 호출은 테스트에서 stub으로만 확인했습니다. 2026-09-17에는 Python 3.14.6(Torch 없음)과 Python 3.13.12/Torch 2.11.0에서 당시 22개 테스트를 실행했으며, 그때는 패키지를 설치하지 않았습니다.

## 테스트가 확인한 것

- 화학/표적/both 분할의 결정성(고정 golden 배정)과 누출 차단, off-diagonal 제외, 재분할 거부. 학습 불가 그룹의 train 고정, 할당량보다 큰 그룹의 train 유지, 여러 chem_group의 structure_id 공유와 빈 split 거부.
- Manifest의 알 수 없는·대소문자가 다른 열 거부, 대소문자만 다른 sample_id 거부, 같은 관측의 상충 label 거부와 반복 관측 집계, 같은 서열의 target_group 불일치 거부, InChIKey 분자 동일성(stub과 실제 RDKit).
- QC fail, uncertain, phenotypic, weight 0 및 test 관측이 학습/validation에 섞이지 않음.
- Import mapping의 key·값 검사, 중복 JSON key 거부, 상충 label의 오류/`uncertain` 처리, QC 없는 import의 `uncertain`, 행 필드 수, compound table 규칙, LIT SID 중복 거부, 내용 기반 자동 ID, `raw.jsonl`–manifest 연결.
- 다운로드의 SHA256·Content-Length·중단 오류, redirect 단계별 검사, 영수증의 검증 여부 기록, 기존 파일 덮어쓰기 차단, PubChem 화합물 표의 `.partial` 처리.
- 지표의 tie 처리, AUROC, enrichment@1%, n ≤ K의 @K null, split별 report, task macro 선택 점수.
- Task별 출력 gradient, 공유 hidden 전이, joint에서 학습한 backbone의 frozen 전달과 저장·복원, 구조 loss 경로와 관측 weight.
- 실제 작은 CPU 학습 루프: checkpoint 선택(`--select-metric`), 재개의 parameter 동등성, 이전 seed 방식 checkpoint와 이후 step 파일이 있는 출력으로의 재개 거부, hash 기반 seed의 독립성. CPU gloo 2-process에서 validation 분할·수집 순서.
- bf16 autocast에서 fp32 head logit, head mode의 pooled cache에서 packet 재로딩 생략, binding key 기반 packet 검증.
- 디스크 embedding: 고유 binding key당 1회 encode, shard 조합, 누락·중복·변조·설정/feature 불일치 거부. 로딩 시 검증 후 RAM 재사용, 동일 head의 native/mock 경로와 캐시 경로 예측 일치, Protenix·원본 checkpoint 없는 CPU train/predict/resume/init.
- Baseline: 표적/task train hit-rate와 cold-target task prior fallback, 실제 RDKit descriptor 모델의 train-only fitting, held-out 변경 불변성, zero-shot 방향·범위·sample 대응·provenance 검사, main CLI와 compare 호환. Zero-shot 숫자는 가상 fixture이며 실제 성능 측정이 아님.
- CLI를 통한 train/predict/export(작은 대체 backend), 잘못된 옵션의 사전 거부, 학습하지 않은 task 건너뛰기와 manifest 불일치 거부, compare 입력 검사.
- `prepare`의 사전 검사(출력 생성 전), binding key별 1회 featurize와 train 행에만 구조/synthetic 생성(Protenix stub), strict upstream 설정 key, Protenix source·commit 기록.
- 양성 구조와 synthetic 분기, apo 매핑의 일대일 정수 제약, 인공 ligand 강체 배치의 내부 거리 보존.
- README 예제 명령 연쇄의 결과 수(아래 표와 74개 입력).

CPU 대체 backend와 stub의 통과는 Protenix, CUDA 또는 NCCL 호환성을 입증하지 않습니다.

2026-09-29에 기존 demo를 새 임시 경로에서 재생성했습니다. chemistry/target/both split CSV와 audit JSON, inputs JSON은 committed 출력과 byte 단위로 동일하여 교체할 파일이 없었습니다. 과거 패키지 설치·공개 API 검사는 아래에 남긴 이전 실행 기록이며 이번 작업에서 반복하지 않았습니다.

## CLI 예제 결과

[가상 원본](../examples/general_screen.csv) 147개 관측은 6개 표적 × 12개 화합물 × 2개 task의 144개 binary 관측에 실패·불확실·phenotypic 각 1개를 더한 것입니다. 물리적/실험적 benchmark가 아닙니다.

`--strategy both --seed 42` 결과:

| split | 전체 관측 | 학습/지표에 적격인 관측 |
|---|---:|---:|
| train | 67 | 64 |
| val | 4 | 4 |
| test | 4 | 4 |
| excluded | 72 | 0 |

`chemistry`와 `target`은 각각 train 99/96, val 24/24, test 24/24입니다. 두 task 모두 train/val/test에 양성과 음성이 있습니다. Inputs JSON은 제외 교차쌍과 phenotype을 빼고 관측마다 하나씩 74개를 생성하며, 고유 표적–ligand 조합은 36개입니다. QC fail/uncertain도 예측 입력은 만들 수 있으므로 적격 관측 72개보다 2개 많습니다. 입력 JSON에는 label이 없습니다.

저장소의 `data/demo_*`는 공개 저장소에 포함된 예제 출력이며, 현재 코드로 프로젝트 루트에서 아래 명령을 실행해 다시 만들었습니다. 분할 CSV, inputs, merge 결과는 이전 커밋과 byte 단위로 같았습니다. 달라진 것은 `raw.jsonl`(행 번호·sample_id 연결), `provenance.json`(지정한 상대 경로, mapping hash, 상충·반복 집계), audit의 `molecule_identity`입니다. audit 3개는 RDKit가 설치된 Python으로 만들었으므로 `molecule_identity`가 `checked: true, molecules: 12`입니다. RDKit 없이 다시 만들면 이 두 값만 `false`/`0`으로 달라집니다. 모든 명령은 기존 출력을 덮어쓰지 않으므로 다시 만들 때는 새 위치에 만든 뒤 비교하세요.

```bash
python3 -m fragment_ft import-csv examples/general_screen.csv \
  --mapping examples/general_mapping.json --output data/demo_import
for s in chemistry target both; do
  python3 -m fragment_ft split data/demo_import/manifest.csv --strategy $s --seed 42 --output data/demo_$s.csv
  python3 -m fragment_ft audit data/demo_$s.csv --output data/demo_$s.audit.json
done
python3 -m fragment_ft inputs data/demo_both.csv \
  --targets examples/general_targets.json --output data/demo_inputs.json
head -73 data/demo_import/manifest.csv > data/demo_part1.csv
(head -1 data/demo_import/manifest.csv; tail -n +74 data/demo_import/manifest.csv) > data/demo_part2.csv
python3 -m fragment_ft merge data/demo_part1.csv data/demo_part2.csv --output data/demo_merged.csv
```

## 공개 API smoke의 의미

2026-09-17에 실행한 명령:

```bash
python3 -m fragment_ft fetch-pubchem --aid 2244 --with-compounds \
  --output data/smoke/pubchem_2244_20260917
```

AID 2244는 **세포 기반 독성 assay**이며 API/파일 형식 검사에만 사용했습니다. Binding 학습 데이터로 편입하지 않았고 label을 curate하지 않았습니다. 생성된 receipt 3개의 파일 hash를 2026-09-28에 다시 확인했습니다. 이 작은 수집 성공으로 공개 corpus 전체의 품질·완전성·학습 개선을 주장하지 않습니다. 이 자료는 저장소와 함께 공개되며 출처와 이용 정책은 [data/NOTICE](../data/NOTICE)에 있습니다. 당시 코드의 영수증에는 이후 추가된 `sha256_verified`, `content_length` 필드가 없습니다.

## 아직 검증하지 않은 것

- [ ] 실제 Protenix checkpoint 로딩과 feature API 실행, native `prepare`(RDKit 동일성 검사 자체는 확인됨), 실제 CIF/apo의 원자·화학적 대응.
- [ ] Native 구조 loss·pose 품질, CUDA/BF16 수치 동작, NCCL·다중 GPU/다중 node 처리량(16 H100 포함).
- [ ] 실제 공개 corpus 성능과 공개→개인 전이 효과.

H100 사용 및 성능 개선 수치를 보고한 상태가 아닙니다. 구현 테스트 통과로 위 항목을 대체하지 않습니다. 실행 방법은 [학습 문서](TRAINING.md)에 있고, 실제 환경 검증 순서는 [연구계획](../FINETUNING_PLAN.md)에 있습니다. 이전 구현 체크리스트는 [0.2.0 구현 상태](archive/implementation_0.2.md)에 보관했습니다.

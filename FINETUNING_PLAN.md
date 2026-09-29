# 범용 screening 모델 연구계획

업데이트: 2026-09-29. 사용 가능 자원: H100 16장, 4개월. 목표는 공개 fragment screening·repurposing 자료에서 학습하고 새로운 표적·화합물 및 자체 X-ray 자료로 전이할 수 있는 프로그램과 검증된 모델입니다. 프로그램 구현과 모델 성능 입증은 별개의 산출물입니다.

개인 자료는 두 표적, 공통 fragment 약 300종, X-ray 및 실제 apo 구조이며 양성 수는 아직 미정입니다. 이 자료는 개인 적용·전이 평가에 사용합니다. 두 표적만으로 범용 protein 일반화를 입증하지 않습니다. [기존 연구계획](docs/archive/two_target_research_plan_20260916.md)은 보관했습니다.

## 먼저 확인할 가설

공개 음성은 관련 task와 충분한 QC를 갖고 있을 때 도움이 될 수 있습니다. 다른 실험의 inactive를 비결합으로 합치면 잘못된 감독 신호와 source 편향이 늘어날 수 있습니다. FDA 승인약과 fragment의 크기·화학공간 차이 때문에 전체 성능과 fragment subset 성능을 따로 확인합니다. 음성 수 증가 자체를 개선의 증거로 보지 않습니다.

| 실험 | 목적 | 현재 실행 경로 |
|---|---|---|
| 개인 데이터만 frozen head | 공개 자료 없는 기준선 | `train --mode head` |
| 공개 QC 자료 head → 개인 head | 공개 공유 표현 전이 효과 | `--init-checkpoint` |
| 위 모델 + 양성 구조 joint | 구조 supervision 추가 효과 | `--mode joint --negative-mode classifier` |
| 위 모델 + 인공 음성 구조 | 인공 좌표가 도움이 되는지 검증 | `--negative-mode synthetic` |
| 동일 corpus uniform vs 균형 sampling | 대형 assay 지배 영향 | `--sampling uniform/assay` |
| task subset/출처별 ablation | biochemical 추가와 source shift 영향 | 검토한 manifest 부분집합으로 별도 run |

Protenix zero-shot(ipTM/PAE/pLDDT), ligand-only, 표적 hit-rate baseline을 1개월차에 먼저 평가합니다. 같은 관측 목록과 고정 split을 사용하고, 학습이 필요한 baseline은 train 관측만 사용합니다. Zero-shot confidence는 보정된 결합 확률이 아니므로 순위 지표와 calibration 지표의 의미를 구분합니다. 직접 affinity 회귀, PU learning, 임의 decoy 학습, 전 모델 학습은 현재 구현 범위가 아닙니다.

## 1개월차: corpus와 실패 없는 실행

1. PubChem·LIT-PCBA·NCATS와 공개 fragment screening에서 **실측 관측 목록**을 확보합니다. 구조가 없다는 이유로 음성을 만들지 않습니다. 원본·라이선스·hash·mapping·제외 이유를 남깁니다.
2. Assay 의미와 판정 규칙을 정하고 X-ray/직접 결합/biochemical을 분리합니다. Phenotypic 결과는 저장하되 protein 학습에서 제외합니다. 승인약 metadata는 화학공간 분석용으로 별도 보존합니다.
3. 표적 construct·화학 ID·유사성 그룹을 전역 정리합니다. 양성·음성·불확실 수와 독립 group 수, 반복 관측·상충 결과·출처 중복을 감사합니다. 같은 관측의 상충 label은 프로그램이 거부하므로 검토해 `uncertain`으로 표시하고, 출처 간 중복은 직접 정리합니다.
4. Cold chemistry, cold target, both 평가 세트를 고정하고 개인 test와 공개 pretraining train의 중복도 `audit --against`로 확인합니다. 사전학습 Protenix의 데이터 중복 가능성은 별도로 기록합니다.
5. 기존 서버의 Protenix를 연결해 양성/음성/apo 최소 사례로 feature·loss·gradient·save/resume·prediction을 실행합니다. GPU peak memory와 실제 step 시간을 측정합니다.
6. 동일 validation 관측에서 zero-shot 세 점수, ligand-only, hit-rate와 작은 frozen head를 비교합니다. 점수 추출 범위·seed·pose 선택 규칙은 label을 보기 전에 고정합니다. Zero-shot을 넘지 못하면 대규모 학습 전에 표현·데이터·task 가정을 다시 검토하며, 고정 test는 최종 평가까지 사용하지 않습니다.

종료 조건: versioned corpus/분할, assay label 정의서, native 1-GPU smoke 결과, baseline 대비 frozen head의 validation 결과. 이 조건 전에는 16장을 장기간 예약 학습에 쓰지 않습니다.

## 2개월차: frozen 기준선과 공개 전이

개인-only head와 public→personal head를 동일 held-out 관측에서 비교합니다. 공개 corpus는 task/target/assay 균형 sampling을 기본으로 사용하고 uniform과 비교합니다. 각 task train/val에 두 class가 있어야 하며 없는 task는 자료를 추가하거나 별도 실험으로 분리합니다.

평가 지표는 split/source/표적/task/assay별 average precision, AUROC, Brier, log loss, precision/recall/enrichment@K, enrichment@1%입니다. 현재 checkpoint 선택 기본값은 task macro validation average precision이며 `--select-metric log_loss`로 바꿀 수 있습니다. K와 subset은 실험 전에 정하고 test로 조정하지 않습니다. 신뢰구간은 독립 화학/표적 그룹 단위 bootstrap으로 후속 분석하며 현재 CLI에는 구현하지 않았습니다.

종료 조건: 개인-only 대비 공개 전이의 이득/손해를 target·fragment subset별로 제시. 큰 공개 corpus의 평균만 좋아졌다는 이유로 개인 성능 개선을 주장하지 않습니다.

## 3개월차: 제한적 joint와 인공 음성 비교

양성 복합체가 충분하고 native 대응이 검증되면 일부 backbone 모듈만 fine-tuning합니다. 음성은 기본적으로 binary loss만 적용합니다. 인공 음성 비교는 실제 apo에 ligand를 강체 이동한 구조 loss를 별도로 추가합니다. 이 좌표는 실험 정답이 아니므로 배치 거리·seed에 따라 결과가 달라지는지도 측정합니다.

Classifier와 synthetic 비교는 같은 초기값, split, seed, 양성 구조 학습 빈도를 유지합니다. Synthetic의 추가 forward 비용은 따로 보고하여 같은 step과 같은 GPU-hour 비교를 구분합니다. 양성 pose 품질은 native 추론 후 적절한 receptor alignment와 ligand 대칭을 고려한 외부 평가로 확인합니다. 현재 `predict`는 판별 점수만 제공하고 pose 생성/RMSD 계산은 제공하지 않습니다.

종료 조건: held-out hit 선별과 양성 pose 품질의 동시 평가. 성능·calibration·pose가 악화하면 joint/synthetic을 기본 모델로 채택하지 않습니다.

## 4개월차: 고정 test와 전달

설정을 validation으로 고정한 뒤 공통 test를 최종 평가합니다. 가능하면 다른 출처·캠페인·시간의 추가 holdout과 후속 실험으로 재현성을 확인합니다. 성공·실패 양쪽을 보존하고, 가설 변경 이후 test 재사용 여부를 명시합니다.

전달물은 데이터 snapshot/원본 hash/mapping/분할, 학습 설정·기반 checkpoint hash·delta, 검증 지표·선정 근거, native 환경 정보, 재현 명령입니다. 현재 구현된 기능과 실제 학습으로 얻은 성능을 구별해 보고합니다.

## H100 16장 사용 기준

H100 16장을 120일 연속 사용할 경우 계산상 46,080 GPU-hour입니다. 이는 상한 예산이며 실제 사용 가능 시간·메모리·네트워크는 서버 조건에 따릅니다. 처리량을 측정하기 전에 epoch 수·완료 시각·성능을 보장하지 않습니다.

Frozen head 실험은 고정 backbone의 pooled embedding을 GPU별로 나누어 한 번 추출하고, 저장된 벡터로 CPU에서 독립 seed/ablation 학습을 수행하는 경로를 우선 측정합니다. Backbone·입력·추출 설정이 바뀌면 embedding도 다시 생성합니다. 추출 시간과 CPU 학습 시간을 따로 기록하며 수 초 완료를 미리 보장하지 않습니다. Backbone joint에서 병렬 효율이 확인되면 2→8→16장으로 확장합니다. 16장 사용 자체를 목표로 삼지 않습니다.

실행 명령은 [학습 문서](docs/TRAINING.md), 데이터 정책은 [데이터 문서](docs/DATA.md)에 있습니다. 현재 실제 checkpoint 학습과 H100 실행은 수행하지 않았습니다.

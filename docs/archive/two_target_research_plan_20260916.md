# Fragment screening 데이터를 이용한 Protenix fine-tuning 계획

작성일: 2026-09-16. 자원: H100 16장, 약 4개월.

확인된 사용자 조건: **단백질 표적 2개, 두 표적에 동일한 fragment 약 300종을 사용한 X-ray screening, 실제 apo 구조 보유.** 표적별 결합 hit 수는 사용자가 추후 제공할 예정이다. 모든 조합에 유효한 측정이 있다면 고유 protein–fragment 조합은 약 600개이며, 구조 복제·회전·conformer 추가는 독립 실험 수를 늘리지 않는다. 측정 실패·미측정은 음성으로 처리하지 않는다.

이 문서는 연구계획이다. 학습 프로그램과 사용자 요청에 따른 인공 비결합 구조 비교 경로를 구현했으며 실행 방법은 [README.md](README.md)에 있다. 실제 구조 데이터, Protenix checkpoint 실행 및 GPU 처리량은 아직 검증하지 않았다. 아래 기간·학습 설정·자원 배분은 시작 가정이며, 첫 2주의 데이터 감사와 처리량 측정 후 조정한다.

## 1. 목표와 핵심 결정

최종 목표는 새 protein–fragment 조합에 대해 다음 두 출력을 제공하는 것이다.

1. 해당 screening 조건에서 결합 hit일 가능성 또는 우선순위.
2. 결합하는 경우 가능한 결합 자세와 그 구조의 신뢰도.

**이번 4개월의 범위는 보유한 두 표적에서 새로운 fragment를 선별하는 target-specific 모델이다.** 새로운 단백질 전반에 대한 일반화는 이번 데이터만으로 입증하지 않는다. X-ray label만으로 학습할 때 출력의 의미는 우선 해당 조건에서의 관측 hit 가능성이며, 용액에서의 결합 확률이나 Kd로 바로 해석하지 않는다.

**추천 접근은 사전학습 Protenix의 구조 예측 능력을 유지하면서, 양성 구조로 자세를 fine-tuning하고 양성·음성 실험 결과로 별도의 binding head를 학습하는 것이다.** 구조 confidence와 결합 확률을 서로 다른 출력으로 관리한다. 정량 Kd/Ki가 충분히 있을 때만 affinity 회귀를 추가한다.

확인한 공식 구현에는 diffusion, distogram, confidence head가 있으며, 본 계획에서 말하는 실험적 결합 여부 head는 추가 개발 사항이다. 공식 fine-tuning 예제 실행만으로 binary binding 학습이 완성되지는 않는다. [모델 코드](https://github.com/bytedance/Protenix/blob/4c355be4553512f72453ecbfb65e69f4c35d1413/protenix/model/protenix.py)

이 규모에서는 **단순 화합물 분류기 → frozen Protenix + 작은 head → 제한적인 구조 fine-tuning** 순서가 우선이다. 전 모델 fine-tuning은 기본 계획에서 제외하고, 충분한 positive 다양성과 검증 개선이 확인될 때만 추가한다. GPU보다 독립 positive 수, scaffold 다양성, 음성 판정의 신뢰성이 주요 제약이 될 가능성이 높다.

## 2. 비결합 fragment 구조를 어떻게 사용할 것인가

| 보유 자료 | 해석 | 학습에 사용하는 방법 |
|---|---|---|
| 결합이 관측된 protein–fragment 복합체 | 관측된 결합 자세 | 구조 정답 + 양성 binding label |
| 실험에서 비결합으로 판정된 fragment의 SMILES/SDF와 apo 단백질 | 해당 조건에서의 음성 관측; ligand의 상대 위치 정답은 없음 | binding label; ligand 복합체 좌표 loss는 적용하지 않음 |
| X-ray에서 ligand density가 검출되지 않은 구조 | 결정·농도·검출한계 조건에서 미검출 | QC를 통과한 screening-negative 또는 불확실 label |
| fragment를 단백질에서 임의로 멀리 배치한 구조 | 인위적 좌표이며 실험적 비결합 자세가 아님 | 기본 학습에서 제외; 사용자 요청에 따른 별도 synthetic 비교군에만 사용 |
| 실험 없이 docking score가 낮거나 임의 조합으로 만든 decoy | 합성 음성 후보 | 실험 음성과 분리; 초기 핵심 학습에서는 제외 |
| 같은 단백질 apo 구조를 여러 fragment에 복사 | protein 좌표 정보는 반복됨 | 각 protein–ligand assay는 구별하되 독립 구조 수로 중복 집계하지 않음 |

멀리 놓은 ligand를 정답으로 학습하면 배치 거리와 방향이라는 인공 규칙을 학습할 수 있다. 따라서 기본 모델은 음성에 구조 loss를 적용하지 않는다. 사용자가 승인한 비교군에서는 실제 apo에 ligand를 강체 회전·이동한 인공 좌표를 별도 supervision으로 추가한다. 양성 구조 학습 빈도와 분할을 유지하고, 배치 거리·seed에 따른 민감도를 평가한다. 이 비교군은 실험적으로 관측된 비결합 자세를 학습하는 것으로 해석하지 않는다.

사용자가 보유한 실제 apo 구조는 표적별 공통 reference/template 후보로 사용한다. 비결합 sample마다 측정한 apo가 따로 있어도 전체 ensemble을 사전에 정해 모든 fragment에 동일하게 적용한다. 특정 fragment를 측정한 뒤에야 얻는 apo/holo 변화는 새 fragment 예측의 입력으로 사용할 수 없으므로, 해당 sample에만 그 구조를 주는 경로는 만들지 않는다. fragment 자체의 SMILES/SDF 및 결합 topology는 음성에도 필요하다.

X-ray screening이라면 낮은 occupancy, disorder, 접근 불가능한 crystal pocket, 용해도와 soaking 조건을 확인한다. 미검출을 모든 조건에서의 비결합으로 확대 해석하지 않는다. PanDDA 연구는 기존 map에서 식별하기 어려운 부분 점유 결합 상태가 재분석에서 드러날 수 있음을 보여준다. [PanDDA 원논문](https://www.nature.com/articles/ncomms15123)

NMR/SPR 등이라면 실제 농도 범위, 응집·간섭, 재현성, 검출한계 및 대조군을 해당 방법에 맞게 기록한다. 수치 없이 얻은 음성을 임의의 Kd 값으로 변환하지 않는다. 실제로 `Kd > C` 같은 경계가 측정되었을 때만 검열된 수치로 보존한다.

## 3. 첫 번째 작업: 데이터 목록과 품질 감사

학습 단위는 파일 하나가 아니라 **protein construct × fragment × assay condition**이다. 구조 파일이 여러 개여도 동일 조합·반복 측정 관계를 보존한다.

최소 목록 필드:

| 구분 | 필드 |
|---|---|
| 식별 | sample_id, target_id, construct_id, ligand_id, assay_id, replicate_group |
| 단백질 | sequence, mutation, chain/assembly, apo_reference_path |
| 화합물 | isomeric SMILES, SDF, canonical identity, stereochemistry, charge, compound batch |
| 실험 | method, concentration, unit, pH, temperature, detection limit, date, batch |
| 라벨 | positive / negative / uncertain, label_source, label_confidence, binding_site |
| 수치 | Kd/Ki 등 값·단위·관계기호, 없으면 결측 |
| 구조 | bound_cif_path, map/MTZ 경로, resolution, occupancy, altloc, 구조 검토 결과 |
| 추적 | source, 공개/PDB 여부, checksum, split, exclusion_reason |

해야 할 일:

- [ ] 표적 수, 고유 positive/negative/uncertain 조합 수, 독립 scaffold 수를 집계한다.
- [ ] 표적별 class imbalance와 assay 간 label 충돌을 확인한다.
- [ ] 공통 ligand_id로 두 표적의 결과를 연결하고, 양쪽 hit / A만 hit / B만 hit / 양쪽 미검출을 집계한다. 불확실·측정 실패·미측정은 별도로 보존한다. 이 구분은 X-ray 관측 패턴이며, 조건 차이와 검출한계를 검토하기 전에는 정량적 선택성으로 해석하지 않는다.
- [ ] 원본 구조·실험 파일은 보존하고, 변환본과 배제 사유를 별도 기록한다.
- [ ] ligand 원자명, 원자 대응, 결합 차수, 방향족성, 입체화학, protonation을 검증한다.
- [ ] buffer·cryoprotectant·다른 ligand와 screening fragment를 구별한다.
- [ ] alternate conformation과 occupancy를 검토하고 bound-state 좌표를 일관되게 선택한다.
- [ ] biological assembly, crystal contact, symmetry mate가 결합 해석에 미치는 영향을 확인한다.
- [ ] 공유결합 fragment와 비공유결합 fragment를 구분한다.
- [ ] 꼭 필요한 금속·cofactor를 보존하는 정책을 정한다. 물 매개 상호작용은 초기 모델의 제한으로 기록한다.
- [ ] 미검출 중 품질이 낮은 사례는 uncertain으로 분리하고 초기 binary loss에서 제외한다.
- [ ] raw diffraction/NMR 데이터와 대형 cache/checkpoint는 Git에 넣지 않고 경로·해시만 관리한다.

산출물: versioned manifest, QC 보고서, 제외 목록, label 정의서.

## 4. 평가 문제와 데이터 분할을 먼저 고정

### 동일 표적에서 새로운 fragment를 찾는 경우

- ligand scaffold 또는 화학 유사도 그룹으로 train/validation/test를 분리한다.
- 같은 화합물의 반복 측정, protonation/conformer 변형, 유사한 유도체가 양쪽에 섞이지 않게 한다.
- 가능하면 마지막 screening batch/시점을 최종 test로 보관한다.
- scaffold 없는 작은 fragment도 있으므로 scaffold split만 믿지 않고 fingerprint 유사도 분포를 점검한다.
- 이번 두 표적은 공통 라이브러리를 사용했으므로 새 fragment 평가에서는 동일 화합물/화학 그룹을 두 표적에 걸쳐 같은 split에 둔다. 예를 들어 fragment F가 A의 test에 있으면 B에서도 test에 두며, B의 train에는 넣지 않는다. 화학 유사도 그룹은 공통 라이브러리에서 한 번 정한다.
- 두 표적의 PR-AUC/precision@K를 각각 보고한다. 표적별 기본 hit 비율 차이만으로 통합 지표가 높아지지 않는지 확인한다.

### 새로운 표적으로 일반화하려는 경우

- 추가로 protein sequence 및 pocket 유사성 기준으로 표적 그룹을 분리한다.
- ligand가 새롭다는 주장도 하려면 ligand 그룹까지 분리한 평가를 추가한다.
- 단일 표적 데이터로 얻은 결과를 새로운 단백질 전반에 대한 일반화로 해석하지 않는다.

### 누출 방지

- 사전학습 checkpoint의 데이터 cutoff, 공개 PDB와의 중복, 유사 복합체를 기록한다. 공개일 이후라는 조건만으로 완전한 독립성을 주장하지 않는다.
- test 복합체 또는 그 ligand 자세가 template/reference conformer로 입력되지 않게 한다.
- 입력 receptor는 양성·음성에 같은 규칙을 쓴다. 예: 표적별 공통 apo template 또는 사전에 고정한 receptor ensemble.
- 양성만 해당 holo receptor, 음성만 apo receptor를 쓰는 입력 구성을 피한다. 실제 새 fragment에는 알 수 없는 ligand별 구조 변화가 정답을 누설할 수 있다.
- pocket 위치를 입력한다면 학습 자료 또는 사전 지식으로 정하고, test ligand 좌표에서 추출하지 않는다. known-pocket과 blind 예측은 별도 평가한다.
- 각 표적의 MSA/template 정책과 seed·sampling 수를 비교 모델 간 고정한다.
- threshold와 calibration은 validation에서 결정하고 최종 test는 마지막에 사용한다.

권장 시작점은 그룹 기준 약 70/15/15 분할이다. 양성이 적으면 고정 비율보다 각 평가 집합의 독립 양성 수를 우선하고, 개발에는 grouped cross-validation을 사용한다. 최종 전향 검증 집합은 유지한다.

이번 300종 규모에서는 위 비율을 기계적으로 적용하지 않는다. 표적별 hit가 적으면 test에 양성이 몇 개만 남아 성능 추정이 매우 불안정해질 수 있다. 먼저 hit/scaffold 수를 세고, 가능한 fold 수의 grouped 교차검증과 새 라이브러리의 전향 검증을 중심으로 설계한다. 두 표적 중 하나를 통째로 제외한 평가는 탐색적으로만 보고, 일반적인 새 표적 성능의 추정치로 사용하지 않는다.

## 5. Protenix 버전과 데이터 파이프라인

확인 기준: 공식 저장소 main `4c355be4553512f72453ecbfb65e69f4c35d1413` (2026-08-01 commit), 확인일 2026-09-16.

- 공식 README의 최신 모델은 `protenix-v2`이며 공식 fine-tuning demo는 `protenix_base_default_v1.0.0`을 사용한다. 첫 pilot에서 둘의 구조 성능과 학습 가능성을 확인한 뒤 하나를 고정한다. 최신 모델이라는 이유만으로 사용자의 fragment 성능 향상을 가정하지 않는다. [공식 저장소](https://github.com/bytedance/Protenix), [fine-tuning 예제](https://github.com/bytedance/Protenix/blob/4c355be4553512f72453ecbfb65e69f4c35d1413/finetune_demo.sh)
- 코드 commit, checkpoint SHA256, 환경, config, 데이터 manifest 버전을 기록한다.
- 양성 구조는 공식 CIF 전처리 도구를 재사용한다. 공식 출력은 구조 `.pkl.gz`와 chain/interface index CSV다. 사설 ligand의 CCD/chemical-component 정보와 원자 대응을 별도로 점검한다. [자체 데이터 준비 문서](https://github.com/bytedance/Protenix/blob/4c355be4553512f72453ecbfb65e69f4c35d1413/docs/prepare_training_data.md)
- 전처리 과정에서 작은 fragment, 필요한 chain/cofactor, 관심 interface가 제거되지 않았는지 전후 개수와 실제 샘플을 비교한다.
- 음성은 존재하지 않는 복합체 좌표를 만들지 않고 protein–ligand 입력과 assay label을 읽는 별도 binding dataset으로 구성한다.
- `finetune_subset.txt`는 기존 구조 데이터의 subset 선택 예제다. 자체 negative manifest를 넣는 것만으로 binary task가 생기지는 않는다.
- MSA는 고유 sequence별로 생성·캐시한다. CPU, RAM, local NVMe, shared storage, multi-node network를 같이 확보한다.
- 전체 공식 학습 데이터를 내려받는 경로는 최소 1.5 TB 디스크가 필요하다고 안내한다. 자체 데이터 fine-tuning에 전체 다운로드가 반드시 필요한 것은 아니며, replay 범위와 MSA DB에 따라 저장 용량을 산정한다. [학습 문서](https://github.com/bytedance/Protenix/blob/4c355be4553512f72453ecbfb65e69f4c35d1413/docs/training_inference_instructions.md)

## 6. 비교 기준과 학습 순서

| 단계 | 실험 | 필요한 이유 |
|---|---|---|
| B0 | ligand fingerprint + 간단한 분류기; 표적별 또는 적절한 protein context 포함 | 화학적 유사성만으로 얻는 성능 기준 |
| B1 | 원본 Protenix 구조/신뢰도 | 구조 fine-tuning의 순수한 개선량 측정 |
| B2 | Boltz-2의 binary binding 출력; 가능하면 표준 docking 점수 | 기존 결합 판별 방법과 비교 |
| F1 | Protenix 고정 + 작은 binding head | 실험 음성의 정보가 얼마나 도움이 되는지 평가 |
| F2 | positive-only 구조 fine-tuning | 구조 데이터 자체의 효과 평가 |
| F3 | 검증된 일부 backbone과 binding head의 공동 학습 | 자세와 판별의 동시 개선 여부 평가 |
| F4 | 전체 파라미터 fine-tuning | F3보다 개선될 근거가 있을 때만 수행 |

Boltz-2 공식 문서는 binder/decoy 구별용 `affinity_probability_binary`를 affinity 회귀 출력과 구분한다. 이를 비교 기준으로 사용하며 Protenix 연구 목표를 대체하지 않는다. Boltz-2 연구의 별도 affinity 모듈과 frozen-trunk 학습은 F1의 관련 선행 사례지만, 사용자의 데이터에서 성능을 보장하는 근거는 아니다. [공식 출력 설명](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md), [Boltz-2 연구](https://pmc.ncbi.nlm.nih.gov/articles/PMC12262699/)

F1의 최소 구현 제안:

- Protenix trunk의 protein–ligand 표현을 pooling한 작은 분류 head에서 시작한다.
- 관측된 정답 ligand 좌표는 binding head의 입력으로 주지 않는다.
- predicted pose를 추가할 경우 양성·음성 모두 같은 생성 절차를 사용한다.
- frozen feature는 필요 부분만 캐시한다. 전체 pair tensor 저장 비용은 먼저 측정한다.
- 신뢰할 수 있는 positive/negative에 BCE를 적용한다. 불균형 보정은 sampling 또는 class weighting 중 단순한 방법 하나에서 시작한다.
- assay별 조건 차이가 크면 같은 assay 안에서 비교하고 표적·실험별 지표도 보고한다.
- 불확실 label을 위한 PU learning이나 복잡한 ranking loss는 단순 모델의 한계가 확인될 때 추가한다.

F3의 loss 구성 제안(positive 수와 다양성이 충분하고 F1/F2 결과가 진행을 뒷받침할 때):

```text
L = λstructure × mean(structure_loss on reliable positive complexes)
  + λbinding   × mean(binding_loss on reliable positive/negative assays)
```

두 항은 각자의 샘플 수로 정규화한다. 음성이 많다는 이유만으로 구조 학습 항이 묻히지 않게 한다. 구조 loss에는 기존 diffusion/distogram/confidence 등을 포함하되, **기본 모드는 음성 batch를 구조 loss 경로로 보내지 않는다.** 별도 `--negative-mode synthetic`에서만 인공 음성 구조 loss를 추가한다.

나중에 apo 보조 학습을 추가한다면 ligand의 관측 불가능한 위치뿐 아니라 관련 거리·confidence/resolved label 등 모든 supervision 경로를 점검한다. 좌표 mask 하나를 바꾸는 것만으로 음성 학습이 정확해진다고 가정하지 않는다. [기존 loss 구현](https://github.com/bytedance/Protenix/blob/4c355be4553512f72453ecbfb65e69f4c35d1413/protenix/model/loss.py)

부분 fine-tuning은 head → 일부 Pairformer block 또는 diffusion 모듈 순서로 범위를 늘린다. 어느 모듈이 적합한지는 F2/F3 비교로 결정한다. LoRA는 기본 요구사항으로 넣지 않고, 구현 지원이나 메모리상 이점이 확인될 때 검토한다.

초기 탐색용 가정: head LR `1e-4`–`1e-3`, backbone LR `1e-6`–`1e-5`. 이는 검증된 추천값이 아니라 작은 탐색 범위다. 공식 demo의 LR/step 수를 작은 fragment 데이터에 그대로 복사하지 않는다. validation 기반 early stopping을 사용한다.

필요하면 누출을 제거한 일반 protein–ligand 구조 일부를 replay해 기존 구조 성능 저하를 제어한다. 대형 replay 데이터 구축부터 시작하지 않는다.

## 7. 성공을 판단하는 지표

| 목적 | 우선 지표 | 주의점 |
|---|---|---|
| 결합 fragment 선별 | PR-AUC, precision/recall@K, EF@K | K는 실제 후속 실험 예산과 맞춤; 무작위 기준과 prevalence 표시 |
| 점수 신뢰성 | calibration curve/Brier score | 실제 양성 비율을 유지한 validation/test에서 측정 |
| 양성 자세 예측 | receptor 정렬 후 symmetry-corrected ligand heavy-atom RMSD | ligand 자체를 따로 정렬해 placement 오류를 없애지 않음 |
| 자세 성공률 | top-1 및 top-k RMSD < 2 Å, 보조적으로 더 엄격한 기준 | top-k는 동일 sampling budget에서 보고; 작은 fragment는 contact도 확인 |
| 구조 타당성 | clash, stereochemistry, pocket contact 복원 | RMSD 하나만으로 판단하지 않음 |
| 일반화 | 표적/화학 그룹/assay별 지표와 그룹 단위 불확실성 | 전체 평균이 한 표적에 지배되지 않게 함 |
| 실사용 가치 | 새 fragment의 실제 hit rate와 시간당 평가 수 | 후향 benchmark와 구분 |

음성에는 정답 결합 자세가 없으므로 ligand RMSD를 계산하지 않는다. AUROC는 보조 지표로 보고하고 class imbalance가 큰 경우 높은 accuracy를 성공 기준으로 쓰지 않는다.

최종 후보는 3개 seed 등 가능한 반복으로 재현성을 확인한다. bootstrap은 구조 복제 수가 아니라 독립 화합물/assay 또는 표적 그룹을 단위로 한다. 단일 표적에서는 표적 일반화 불확실성을 추정할 수 없다.

6–7주차의 첫 의사결정: F1이 B0/B1보다 grouped validation에서 개선되지 않으면 label/QC/split/입력 정보를 먼저 검토한다. GPU와 backbone 학습 범위를 바로 늘리지 않는다.

## 8. H100 16장·4개월 운영 계획

120일을 가정한 이론적 상한:

```text
16 GPUs × 24 hours × 120 days = 46,080 GPU-hours
```

실제 월 길이·할당 정책·장애·준비 시간을 반영해야 한다. 60–75%를 유효 실험 시간으로 잡으면 약 27,648–34,560 GPU-hours이며, 이는 측정값이 아닌 예산 가정이다.

첫 주에 H100의 GPU 메모리, SXM/PCIe, node 수, NVLink/InfiniBand, CPU/RAM/스토리지를 확인한다. 16장 사용 권한이 16장을 효율적으로 연결해 학습할 수 있음을 자동으로 의미하지는 않는다.

pilot에서 작은·중간·큰 실제 단백질 사례를 골라 다음을 측정한다.

- BF16 환경에서 forward/backward 성공, finite loss/gradient, checkpoint 저장·재개.
- crop/token 수, effective batch, GPU 메모리 peak, step time.
- ligand와 실제 pocket이 crop 안에 남는 비율.
- 1 → 8 → 16 GPU에서 처리량 및 통신 overhead.
- 전체 추론 시간과 MSA/전처리 시간.

실험 비용은 `steps × seconds_per_step × GPU_count / 3600`으로 산출한다. crop은 예를 들어 384에서 시작해 필요한 pocket context를 기준으로 확대하고, 실제 token화된 크기를 사용한다. mixed precision·gradient accumulation·기존 activation checkpointing을 우선 재사용하며 FSDP 등 추가 복잡성은 메모리 병목이 확인된 뒤 검토한다.

이번 규모에서는 초기 개발에 1–2장, 독립 설정·feature/pose 생성에 필요에 따라 추가 GPU를 사용한다. 반복되는 단백질 MSA는 두 표적의 construct별로 재사용하고, 작은 head 학습은 캐시된 feature로 수행한다. 규모를 확장할 근거가 생기면 본 학습 8장, 독립 설정/반복 4장, 검증·추론 2장, 여유 2장처럼 배분할 수 있다. 단일 작업에 16장을 묶는 것은 확장 효율이 확인된 경우에 한한다.

작은 데이터에 4개월 내내 같은 모델을 학습시키지 않는다. 데이터 정제, baseline, 독립 반복, ablation, 외부 검증에 자원을 배분한다.

## 9. 약 4개월의 일정과 단계별 완료 조건

| 기간 | 해야 할 일 | 산출물 / 다음 단계 조건 |
|---|---|---|
| 1–2주 | 데이터 감사, label 정의, split 고정, 환경·GPU pilot | manifest/QC/split, 구조·음성 사례의 전체 경로 점검, 비용 추정 |
| 3–4주 | 원본 Protenix, fingerprint 분류기, Boltz-2 평가 | 같은 test 정책의 baseline 표; 핵심 지표와 최소 개선 기준 사전 고정 |
| 5–7주 | frozen binding head, positive-only 구조 fine-tuning | 분류와 자세 개선을 각각 측정; 누출·암기 여부 점검 |
| 8–11주 | 근거가 있는 경우 작은 범위의 공동 fine-tuning; 추가 screening 후보 선정 | baseline 대비 개선 또는 한계 확인; 독립 실험 확대 준비 |
| 12–14주 | 제한된 hyperparameter 탐색, seed 반복, calibration, 외부 평가 | 최종 후보 checkpoint와 재현 가능한 평가; 새 실험용 후보 전달 |
| 15–17주 | 새 fragment 전향 검증 결과 분석, 오류 분석, 모델·추론·문서 정리 | 최종 결과표, 추론 pipeline, checkpoint, 데이터·모델 한계 보고서 |

17주는 근사치다. 정확한 GPU 사용 종료일에 맞춰 역산한다. 실험 확인은 GPU가 대신할 수 없으므로 측정·구매·구조결정 일정을 초기에 확보하고, 후보 확정은 가능한 한 12–14주 이전에 한다.

전향 검증에서는 상위 예측 후보뿐 아니라 화학 다양성을 맞춘 비교군/무작위군도 포함하고 같은 조건으로 측정한다. 샘플 수는 예상 hit rate와 실험 예산으로 정한다. 구조 확인이 가능한 일부 hit는 pose 정확도 검증에도 사용한다.

추가 실험 데이터를 모델 개선에 사용하는 active-learning round와, 최종 후보 모델을 평가하는 봉인된 test round는 구분한다. 새 데이터가 들어왔다고 이미 평가에 사용한 test를 반복적으로 학습에 섞지 않는다. 음성 label을 용액 결합 여부로 확장하려면 대표 양성·음성을 NMR/SPR 등 다른 방법으로 확인할 수 있는지 검토한다.

## 10. 데이터 규모에 따라 줄이거나 늘릴 범위

- 표적 하나 또는 소수, positive 수십 개 수준: target-specific ranking과 frozen head/단순 baseline을 우선한다. 전체 backbone 학습의 일반화 근거는 약하다.
- 독립 positive 수백 개와 신뢰할 수 있는 negative가 있는 경우: 일부 모듈 fine-tuning을 검증하고 grouped evaluation을 충분히 확보한다.
- 다양한 표적·scaffold의 많은 복합체가 있는 경우: 공동 학습과 표적 외부 평가의 범위를 늘린다.

이는 하드 cutoff가 아니라 실험 우선순위다. 단순 파일 수보다 독립 화학·단백질 다양성과 label 품질이 중요하다.

## 11. 최소 개발 범위와 최종 전달물

최소 개발 범위는 기존 Protenix에 다음만 추가하는 것이다.

1. 자체 manifest/chemical identity를 읽고 label과 split을 보존하는 데이터 경로.
2. 작은 binding head와 구조/분류 batch의 loss 분기.
3. 동일 입력 정책으로 baseline과 fine-tuned 모델을 평가하는 실행 경로.

핵심 확인은 음성 batch가 구조 loss에 들어가지 않는지, 정답 구조가 입력에 누출되지 않는지, ligand 원자 대응이 맞는지, checkpoint 재개 결과가 일관적인지다. 별도 학습 프레임워크나 dashboard를 먼저 만들 필요는 없다.

최종 전달물:

- 정제 데이터 manifest와 고정 split, QC 및 label 정의.
- 실행 가능한 학습/평가 config와 환경 기록.
- 원본 대비 비교표, seed별 결과, 실패 사례.
- 선택한 fine-tuned checkpoint와 checksum.
- 새 fragment 목록으로부터 binding 점수와 후보 pose를 생성하는 추론 경로.
- 전향 검증 결과 또는 아직 검증되지 않은 범위를 명확히 구분한 보고서.

## 12. 시작 전에 확보할 정보

우선순위가 높은 미확정 항목:

1. 두 표적 각각의 positive/negative/uncertain 수와 독립 positive scaffold 수.
2. 공통 라이브러리의 표적별 유효 측정 범위, X-ray hit 판정 기준과 screening 조건 차이.
3. apo 구조가 표적별 공통 reference인지, 각 soaking sample별 측정인지.
4. 현재 계획은 동일한 두 표적의 새 fragment 예측이며, 목적이 다르면 범위 조정.
5. 정량 affinity, 원본 map/MTZ, 공통 apo reference의 가용성.
6. H100 메모리·node 구성·스토리지, 정확한 할당 기간.
7. 전향 검증에 쓸 실험 시간과 예산.

가장 먼저 완료할 실무 작업은 대표 positive/negative/uncertain 사례와 전체 개수표를 확보하고 데이터 계약 및 평가 문제를 확정하는 것이다.

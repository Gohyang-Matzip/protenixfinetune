# 학습·전이·평가 실행

이 문서의 native 명령은 **기존 Protenix/PyTorch 실행 환경과 기반 checkpoint가 있는 서버**에서 실행합니다. 본 프로젝트는 설치·weight 다운로드를 수행하지 않았습니다. 로컬 검증은 대체 backend의 CPU 학습까지이며 [검증 기록](VALIDATION.md)에 실제 실행 범위를 구분했습니다.

학습 전 [데이터 문서](DATA.md)에 따라 corpus를 검토하고 분할합니다. 아래 `data/manifest.split.csv`는 그 결과 파일의 예시 경로입니다. 모든 명령은 프로젝트 루트에서 실행합니다. 저장소의 `data/`는 GitHub에 함께 공개되므로 비공개 manifest·targets·CIF/apo는 gitignore된 `private/`나 저장소 밖에 두고 예시 경로를 바꾸세요.

## Task와 sampling

공통 Protenix single/pair 표현을 pooling하고, 작은 head의 마지막 출력 행을 `assay_type:endpoint`별로 구분합니다. Target별 전용 head가 아니므로 새로운 target에서도 입력 단백질 표현을 사용할 수 있습니다. 이 구조만으로 새 표적 일반화가 입증되는 것은 아니며 cold 평가가 필요합니다.

- `--mode head`: Protenix를 고정하고 작은 head만 학습합니다. 공유 pooled 표현을 RAM에 캐시합니다. 대규모 corpus에서 메모리 사용량을 측정하세요.
- `--mode joint --negative-mode classifier`: 선택한 backbone 모듈도 업데이트하며 양성 관측 구조 loss를 추가합니다. 현재 joint 모드는 양성 구조를 반드시 필요로 합니다. 구조 없는 공개 corpus는 head 사전학습부터 시작합니다.
- `--mode joint --negative-mode synthetic`: 위 조건에 QC pass X-ray 음성의 인공 구조 loss를 추가합니다. 실제 apo가 필요합니다.
- 기본 `--sampling assay`는 task → target → source/assay → 관측을 각 단계에서 균등 추출합니다. `--sampling uniform`은 전체 행에서 균등 추출합니다. Class 균형을 인위적으로 맞추지는 않습니다.
- `weight`는 관측 BCE에만 적용합니다. 구조 loss는 별도로 선택한 양성 batch에 `--structure-weight`를 적용합니다. 구조/인공 음성 batch는 해당 eligible 행에서 균등 추출하므로 중복 구조를 corpus에서 먼저 정리하세요.

Train과 validation은 같은 task 집합을 가져야 하고, 각 task·split에 양성과 음성이 모두 있어야 합니다. 실패·불확실·weight 0·phenotypic·excluded는 사용하지 않습니다. 농도와 출처는 입력 feature가 아니며 score의 의미는 정한 endpoint에 한정됩니다.

## 표적별 공통 입력과 apo

[examples/targets.json](../examples/targets.json)의 서열을 실제 서열로 바꿉니다. 각 target에는 Protenix `proteinChain` 항목을 넣습니다. `pairedMsaPath`, `unpairedMsaPath`, `templatesPath`를 추가하면 표적의 모든 fragment에 같은 입력을 사용합니다. 파일 경로는 targets JSON에 상대적이며, 로딩할 때 절대 경로로 변환됩니다. 이전 upstream 형식의 `msa.precomputed_msa_dir`도 같은 방식으로 해석하며 디렉터리여야 합니다. Targets JSON은 target_id → 객체이고 경로·서열은 문자열, `msa`·`apo`는 객체(`apo`는 `path` 필수)여야 합니다. 로컬 MSA/template가 없으면 sequence-only 입력이며, 사용 여부는 준비하는 manifest 행이 참조하는 표적만으로 정합니다.

Synthetic 비교를 위해서는 target마다 `apo`를 추가합니다.

```json
{
  "A": {
    "sequence": "ACDEFGHIKLMNPQRSTVWY",
    "count": 1,
    "apo": {
      "path": "apo_A.cif",
      "chain_id": "A",
      "residue_offset": 0,
      "minimum_ca_fraction": 0.8
    }
  }
}
```

`residue_offset`은 입력 서열의 1-based 위치에 더할 apo residue 번호 차이입니다. 예: 서열 첫 잔기가 apo의 101번이면 100. 불연속 번호는 `"residue_map": {"1": 101, "2": 104}`처럼 명시할 수 있습니다. `residue_map`은 나열한 잔기만 덮어쓰고 나머지는 `residue_offset`(기본 0)을 따릅니다. 위 예에서 3번은 apo 105번이 아니라 3번에 대응하므로, 불연속 번호에는 `residue_offset`과 함께 쓰거나 모든 잔기를 나열하세요. Apo에 없는 번호로 매핑된 잔기는 mask에서 빠지며 Cα 비율 검사만 이를 잡습니다. 매핑은 정수이며 일대일이어야 합니다. 매핑된 원자 이름·잔기·원소를 검사하며, 충분한 Cα가 대응되지 않으면 중단합니다. Insertion code는 먼저 명시적으로 정리해야 합니다. Synthetic 생성은 현재 **한 개의 비수정 단백질 chain + 한 개의 비공유결합 ligand**를 지원합니다.

`apo` 항목은 **synthetic 정답 좌표 생성용**이며, 이것만으로 apo가 binding 모델의 template 입력이 되지는 않습니다. Apo를 공통 template로 조건화하려면 upstream 형식의 로컬 `templatesPath`도 제공해야 합니다. 개별 positive의 holo 구조를 그 fragment에만 template로 넣지 않습니다.

```bash
python3 -m fragment_ft inputs data/manifest.split.csv \
  --targets data/targets.json --output data/inputs.json
```

이 명령도 Protenix 없이 동작합니다. Fragment의 holo 좌표나 label은 native 입력 JSON에 넣지 않습니다. 입력은 **관측마다 한 항목**입니다. 예제 both split은 74개 항목이지만 고유 표적–ligand 조합은 36개입니다. 같은 서열이 서로 다른 `target_group`에 있으면 거부합니다.

## GPU 서버에서 데이터 준비

이후 명령은 PyTorch와 Protenix의 기존 실행 환경, CCD/cache 등 로컬 자원이 필요합니다. 기준으로 확인한 upstream 소스는 `4c355be4553512f72453ecbfb65e69f4c35d1413`입니다. 이 프로그램은 본체를 복사하거나 설치하지 않고 `--protenix-source`로 기존 소스를 연결합니다. 이 경로는 `protenix`가 실제로 import되는 디렉터리여야 하며, 다른 설치본이 먼저 잡히거나 `protenix/__init__.py`가 없으면 거부합니다. 실제 경로·commit·로컬 변경 여부(`actual_commit`, `dirty`)를 기록합니다. Commit은 소스 디렉터리 자체가 git checkout 루트일 때만 기록하며, commit을 알 수 없거나 기준과 다르거나 로컬 변경이 있으면 stderr에 경고합니다.

양성 복합체 CIF는 upstream 전처리 스크립트로 변환합니다. 아래 경로는 실제 서버 경로로 바꾸세요.

```bash
export PROTENIX_ROOT_DIR=/data/protenix
export PYTHONPATH=/opt/Protenix
python /opt/Protenix/scripts/prepare_training_data.py \
  -i data/positive_cif -o data/positive_indices.csv \
  -b data/positive_bioassembly -n 8
```

사설 ligand chemical-component 정보/CCD 캐시는 upstream 요구대로 준비해야 합니다. 생성된 index에서 `structure_id`, protein/ligand chain ID를 확인합니다. 같은 bioassembly의 다른 대칭 chain으로 자동 대체하지 않도록 연결부에서 요청한 interface를 고정합니다.

전처리 전 각 양성 chain의 서열·construct와 ligand의 화학적 동일성/입체화학이 targets 및 manifest와 일치하는지 확인하세요. 연결부는 구조 ID와 chain 종류/ID를 검사하지만, CIF ligand의 결합 그래프를 manifest SMILES와 자동 대조하지는 않습니다.

```bash
python -m fragment_ft prepare data/manifest.split.csv \
  --targets data/targets.json --output prepared/all \
  --protenix-source /opt/Protenix \
  --positive-indices data/positive_indices.csv \
  --bioassembly-dir data/positive_bioassembly --mmcif-dir data/positive_cif \
  --synthetic
```

출력 디렉터리를 만들기 전에 기존 출력, `--positive-indices`에 필요한 `--bioassembly-dir`·`--mmcif-dir`과 경로 존재, 모든 표적의 정의·서열, RDKit 분자 동일성, synthetic 대상 train 표적의 apo(`path`, `chain_id`) 파싱, interface index 대응을 먼저 검사합니다. 여기서 실패하면 디렉터리가 남지 않습니다. `prepare`는 RDKit가 필요합니다. `pip install -e '.[train]'`은 RDKit와 torch, numpy를 함께 설치합니다. 기존 Protenix 환경에서 설치 없이 실행한다면 `python3 -m pip install rdkit`로 따로 설치합니다.

- `.binding.pt`: 공통 target context와 fragment SMILES에서 만든 native 입력. Phenotypic/excluded를 제외한 모든 label에 생성합니다. Unknown·QC fail도 예측 입력은 만들 수 있으나 학습·지표에는 쓰지 않습니다. 같은 `(target_id, ligand_id, smiles)`는 한 번만 featurize하고 반복 관측의 파일은 hard link(지원하지 않는 파일시스템에서는 복사)입니다. Packet은 이 binding key로 검증하므로 label 수정이나 재분할로 무효화되지 않습니다.
- `.structure.pt`: 실험적 양성 복합체의 native feature/label. **train split**에서 `structure_id`가 있는 양성에만 생성합니다. Val/test 양성은 interface index 기록이 필요 없습니다. Ref_pos augmentation과 crop은 prepare 때 한 번 정해져 모든 학습 step에서 재사용됩니다.
- `.synthetic.pt`: 실제 apo 원자 좌표와 ligand reference conformer. **train split**의 QC를 통과한 X-ray 음성에만 생성하며, 거리·방향은 학습 시 정합니다. Val/test 표적에는 apo가 필요 없습니다.
- `preparation.json`: 모델명, 실제 소스, `upstream_overrides`, binding 관측 수와 실제 featurize 수(`binding_featurizations`), 입력과 tensor 파일의 SHA256. 학습 전에 cache 변경을 검출합니다.

각 native 호출 전에 행 내용으로 random/NumPy/Torch seed를 정하므로 같은 입력의 prepare 재실행은 같은 tensor를 만듭니다.

### Upstream 설정

`--upstream-config`는 upstream 설정을 덮어쓰는 JSON이며 upstream에 없는 key는 거부합니다. `train_confidence_only`와 원격 template 받기는 켤 수 없습니다. [examples/upstream_config.json](../examples/upstream_config.json)은 형식 예시일 뿐이며, 이 연결부가 이미 강제하는 기본값(`triangle_*: torch`, `model.N_cycle` 4, `diffusion_batch_size` 4)을 반복하므로 그대로 쓰면 아무것도 바뀌지 않습니다. 사용 환경에서 검증한 가속 kernel이 있으면 `triangle_attention`/`triangle_multiplicative` 값을 바꿉니다.

같은 JSON을 prepare, train, `--resume`, `--init-checkpoint`에 모두 사용하세요. Prepare 캐시와 `data.*` 값이 다르면 train과 predict가 거부합니다. Init은 전체 overrides가 같아야 하고, resume은 기록된 metadata 전체가 같아야 합니다. `upstream_overrides`를 기록하지 않은 이전 캐시는 `data.*` 비교를 생략한다는 경고와 함께 사용합니다. 캐시나 checkpoint의 Protenix commit이 현재 소스와 다르면 경고만 합니다. `--model-name`(기본 `protenix_base_default_v1.0.0`, 또는 `protenix-v2`)도 prepare와 train에서 같아야 합니다.

Head만 학습하려면 positive index 관련 옵션과 `--synthetic`을 생략할 수 있습니다. 구조 전처리는 기본적으로 full complex이며, 실제 크기가 크면 `--crop-size 384` 등을 별도로 측정하세요. Synthetic은 full-complex 정답이므로 비교 시에는 우선 양성도 full complex로 준비합니다.

현재 **실험적 양성 및 synthetic 구조 학습 branch는 dummy MSA/template**를 사용합니다. Binding branch만 targets에 지정한 로컬 MSA/template를 사용합니다. 구조 fine-tuning에서 전체 native MSA 학습 파이프라인까지 확장한 버전은 아닙니다. 물·금속·assembly 처리와 원자 대응은 실제 구조로 먼저 확인해야 합니다.

## 학습 모드와 인공 비결합 비교

먼저 작은 판별기만 학습합니다. Backbone은 고정하고 pooling된 표현을 메모리에 캐시합니다. Native MSA sampling은 eval에서도 난수를 쓰므로, head 학습과 평가에서는 label/split과 무관한 입력별 고정 seed를 사용합니다. 재시작 후에도 같은 표현을 계산하며 외부 학습 RNG를 바꾸지 않습니다. 캐시된 표현이 있으면 train과 predict 모두 binding packet을 다시 읽지 않습니다.

```bash
python -m fragment_ft train data/manifest.split.csv \
  --features prepared/all --base-checkpoint /data/protenix/checkpoint/protenix_base_default_v1.0.0.pt \
  --protenix-source /opt/Protenix --mode head --output runs/head \
  --steps 1000 --eval-every 50 --device cuda
```

- 한 step은 rank마다 `--accumulate`(기본 4)개의 microbatch를 처리하며 microbatch마다 complex 하나입니다. `--steps 1000`은 rank당 4,000번의 forward입니다.
- `--precision` 기본값은 train/predict 모두 `bf16`이지만 CUDA에서만 적용되며 CPU는 fp32입니다(metadata의 `effective_precision`). Binding head와 logit은 bf16에서도 fp32로 계산합니다. BF16 수치 동작은 미검증이므로 첫 native smoke는 `--precision fp32`로 실행하세요.
- 기존 `--output`(resume 제외), head mode의 `--trainable-prefix`, prefix 없는 joint mode는 feature hash 계산과 기반 checkpoint 로딩 전에 거부합니다. `--steps`, `--eval-every`, `--accumulate`, `--hidden`, `--top-k`, `--dist-timeout-minutes`는 양의 정수여야 합니다.

본체 일부도 fine-tuning하는 기본 비교군:

```bash
python -m fragment_ft train data/manifest.split.csv \
  --features prepared/all --base-checkpoint /data/protenix/checkpoint/protenix_base_default_v1.0.0.pt \
  --protenix-source /opt/Protenix --mode joint --output runs/joint \
  --trainable-prefix pairformer_stack --trainable-prefix diffusion_module \
  --negative-mode classifier --steps 1000 --seed 42
```

인공 비결합 구조 비교군은 같은 설정에서 다음 옵션을 사용합니다.

```bash
python -m fragment_ft train data/manifest.split.csv \
  --features prepared/all --base-checkpoint /data/protenix/checkpoint/protenix_base_default_v1.0.0.pt \
  --protenix-source /opt/Protenix --mode joint --output runs/synthetic40 \
  --trainable-prefix pairformer_stack --trainable-prefix diffusion_module \
  --negative-mode synthetic --negative-weight 1 \
  --synthetic-distance 40 --placement-seed 10000 --steps 1000 --seed 42
```

학습 loss는 다음과 같습니다.

```text
classifier: observation_weight × BCE_task(양성·음성) + structure_weight × L_structure(양성)
synthetic:  observation_weight × BCE_task(양성·음성) + structure_weight × [L_structure(양성)
                                             + negative_weight × L_structure(인공 음성)]
```

양성 structure sample은 binding sample과 독립적으로 뽑습니다. Synthetic에서는 음성 structure sample을 추가로 뽑으므로 기본 비교군의 양성 학습 빈도를 줄이지 않습니다. 추가 forward 때문에 계산 비용은 더 듭니다. Native 구조 loss 전체를 인공 좌표에 적용하는 ablation이며, 좌표 loss만 단독으로 바꾸는 실험은 아닙니다.

`synthetic-distance`는 단백질과 ligand의 bounding sphere 사이 최소 여유 거리(Å)입니다. 관측된 receptor 좌표는 유지하고 ligand를 강체 회전·이동하므로 내부 결합 길이는 보존됩니다. 배치는 `(--seed, --placement-seed, step, microbatch, rank)`의 hash로 정해지므로 재현 가능하며, 두 seed 중 하나만 바꿔도 달라집니다. 20/40/80 Å와 다른 placement seed로 비교하면 배치 방식에 대한 민감도를 확인할 수 있습니다. 이 거리들 자체에 생물물리학적 정답이라는 의미는 없습니다.

LR, step 수, loss weight는 출발값이며 데이터 규모에 맞춰 validation으로 정해야 합니다. 매 평가(`--eval-every`, 기본 50 step)마다 `step_XXXXXX.pt`와 `step_XXXXXX.json`을 저장합니다. JSON에는 `macro_validation_average_precision`, `macro_validation_log_loss`, 직전 평가 이후 모든 rank·microbatch의 평균 train loss(`mean_train_loss_since_last_eval`), validation 예측, `best_so_far`가 있습니다. 두 macro 값은 task별 평균을 다시 평균하며, task 안에서는 split/source/target/assay 그룹을 같은 가중치로 평균합니다. AP는 양성이 없는 그룹을 건너뜁니다. `best_so_far`는 `--select-metric`을 따릅니다. 기본 `average_precision`은 높을수록, `log_loss`는 낮을수록 좋습니다. Screening 유병률에서 log loss는 순위를 매기지 못하는 base-rate 예측을 선호할 수 있어 기본값을 AP로 정했습니다. Checkpoint에는 `best_selection_score`, `select_metric`, 지금까지 최저 macro log loss(`best_validation_loss`)가 저장됩니다. 표준 출력에도 평가마다 step, validation log loss·AP, `best_so_far`, checkpoint 경로를 JSON 한 줄로 씁니다. 사용자는 마지막으로 `best_so_far: true`였던 checkpoint를 선택합니다. 자동 early stopping이나 best.pt 복사는 없습니다. `test`는 학습·checkpoint 선택에 사용하지 않습니다.

단일 node 16 GPU 실행은 같은 명령 앞에 다음 launcher를 사용합니다.

```bash
torchrun --standalone --nproc_per_node=16 -m fragment_ft train ...
```

실제 cluster가 8장 × 2 node라면 torchrun의 `--nnodes`, `--node_rank`, `--master_addr`, `--master_port`를 cluster 구성대로 지정합니다. 먼저 1장으로 native smoke test를 통과한 뒤 확장하세요. Validation은 모든 rank가 나눠 예측하고 rank 0이 모아 채점·저장합니다. 분산 통신 timeout은 `--dist-timeout-minutes`(기본 60)입니다. 로컬에서는 CPU gloo 2-process 테스트만 실행했으며 NCCL·다중 GPU는 미검증입니다. 기본 kernel은 `torch`이며, 검증한 가속 kernel은 [Upstream 설정](#upstream-설정)처럼 `--upstream-config`로 지정합니다.

## 공개 사전학습 → 개인 데이터 fine-tuning

공개 corpus와 개인 데이터는 각자 manifest/targets/cache를 준비합니다. 새 task의 test 정보가 공개 pretraining train에 들어가지 않도록 **두 단계에 걸친** compound/target/구조 중복 감사가 필요합니다. 개인 manifest를 [`audit --against`](DATA.md#corpus-간-누출-감사)로 공개 manifest와 대조하세요. 각 corpus를 따로 split했다고 전이 실험이 독립적인 것은 아닙니다.

```bash
# 1. 공개 관측으로 head 사전학습. 공개 양성 구조가 있다면 joint 비교도 가능.
python -m fragment_ft train data/public.split.csv \
  --features prepared/public --base-checkpoint /data/protenix/checkpoint/protenix_base_default_v1.0.0.pt \
  --protenix-source /opt/Protenix --mode head --output runs/public \
  --steps 1000 --eval-every 50 --sampling assay

# 2. validation으로 선택한 공개 checkpoint에서 새 개인 데이터 학습 시작.
python -m fragment_ft train private/personal.split.csv \
  --features prepared/personal --base-checkpoint /data/protenix/checkpoint/protenix_base_default_v1.0.0.pt \
  --protenix-source /opt/Protenix --mode head --output runs/personal \
  --init-checkpoint runs/public/step_001000.pt --steps 1000 --eval-every 50
```

`step_001000.pt`는 경로 예시이며 마지막 checkpoint가 항상 최선은 아닙니다. `--init-checkpoint`는 optimizer/step을 초기화하고 공통 표현을 가져옵니다. 같은 task 이름의 출력은 대응시켜 복사하며 새 task 출력은 초기화합니다. Task가 전혀 겹치지 않아도 공유 hidden/backbone은 전달합니다. Base checkpoint hash, native model/config, head hidden 크기는 같아야 합니다. Public head 학습만 했다면 전달되는 개선은 작은 head의 표현이며 backbone은 원본입니다.

`--resume`는 **동일 실험 이어학습**입니다. Manifest/cache/기반 weight, task 순서, mode/prefix, seed, sampling, GPU 수 등 기록된 설정이 일치해야 하며 optimizer와 step을 복원합니다. `--init-checkpoint`와 함께 사용할 수 없습니다. Mode/task가 바뀌면 init을 사용하세요. `--select-metric`도 기록된 설정에 포함되며 `--dist-timeout-minutes`는 포함되지 않습니다. 학습 seed는 `(--seed, step, microbatch, rank)`의 SHA256으로 정하므로 서로 다른 `--seed`는 독립적인 표본 순서를 만듭니다. 이 방식(`seed_scheme` 2) 이전 checkpoint는 `--resume`할 수 없으므로 `--init-checkpoint`와 새 `--output`으로 이어갑니다. 이전 0.1 checkpoint는 예측/초기화 시 기본 task `xray:hit`로 읽지만, metadata가 확장되어 0.1 실험의 strict resume는 지원하지 않습니다.

## 저장·재개·평가

Checkpoint는 **학습한 parameter delta, head, optimizer, step, task 목록, 데이터/기반 weight hash**를 저장합니다. 앞 단계에서 학습한 뒤 현재는 동결한 backbone delta도 보존합니다. Frozen 기반 weight는 복제하지 않으므로 원본 checkpoint가 계속 필요합니다. 학습 mode, prefix, seed, GPU 수 등은 재개 시 같아야 합니다. 매 step의 난수 seed를 다시 정하므로 평가 호출이 이후 학습 RNG를 바꾸지 않습니다.

```bash
# 앞선 train 명령과 같은 설정에 추가; steps는 최종 총 step 수입니다.
--resume runs/joint/step_000500.pt --steps 1000
```

기존 결과를 덮어쓰지 않습니다. 같은 `--output`으로 resume할 때 resume step보다 뒤의 `step_*` 파일(`.pt`/`.json`/`.partial`)이 있으면 학습 전에 바로 거부하므로, 과거 checkpoint에서 이어가려면 새 `--output`을 사용하세요. 최신 checkpoint 파일은 저장 중 임시 `.partial`에서 완성 후 변경되며, 실패한 파일도 임의 삭제하지 않습니다.

선택한 checkpoint를 공통 test에 평가합니다.

```bash
python -m fragment_ft predict data/manifest.split.csv \
  --features prepared/all --checkpoint runs/joint/step_001000.pt \
  --base-checkpoint /data/protenix/checkpoint/protenix_base_default_v1.0.0.pt \
  --protenix-source /opt/Protenix --split test --output runs/joint_test.json
```

`predict`는 checkpoint의 학습 manifest와 SHA256이 다른 manifest를 `--split` 값과 무관하게 거부합니다. 다른 manifest의 held-out 행은 학습에 쓰였을 수 있기 때문입니다. 의도한 경우 `--allow-different-manifest`를 지정하며, 출력의 `same_manifest_as_training`에 기록됩니다. Checkpoint가 학습하지 않은 task의 행은 추론 전에 건너뛰고 stderr에 경고하며 `untrained_tasks`에 개수를 남깁니다. 남는 행이 없으면 실패합니다. 기존 `--output`은 모델을 읽기 전에 거부합니다(`export`도 같음).

Synthetic checkpoint도 같은 방법으로 `runs/synthetic40_test.json`에 평가한 다음:

```bash
python -m fragment_ft compare runs/joint_test.json runs/synthetic40_test.json \
  --output runs/comparison.json --top-k 20
```

`compare`는 평가 sample/metadata를 포함한 평가 관측이 다르면 거부하며 Protenix 없이 동작합니다. 문자열이 아닌 필드, 0/1이 아닌 label, 채점할 관측이 없는 report도 거부합니다. 비교한 checkpoint의 학습 manifest가 서로 다르거나 report가 `--allow-different-manifest`로 만들어졌으면 stderr에 경고하고, 출력의 각 항목에 `same_manifest_as_training`을 남깁니다.

지표는 split/source/표적/task/assay별 average precision(AP), AUROC, Brier, log loss, precision/recall/enrichment@K, enrichment@1%(상위 ceil(0.01·n)개)입니다. `metrics_by_target`의 key는 `[split, source, target_id, task, assay_id]` JSON 목록이고 각 항목에 `split`이 있으므로 `--split all`도 train과 test를 섞지 않습니다. AP와 AUROC는 score tie를 묶어 계산하며, AUROC는 한 class만 있으면 null입니다. Top-K와 상위 1% 경계의 tie는 동일 확률 선택의 기대 hit 수를 사용합니다. 그룹의 관측 수 n이 K 이하이면 @K 지표는 자명하게 1 또는 유병률이 되므로 null입니다(`k`는 기록). AP는 사다리꼴 PR-AUC와 구분합니다. 한 class가 없는 표적의 지표는 제한적으로 해석해야 합니다.

새 fragment는 label을 `unknown`으로 작성하고, 공통 target context로 다시 prepare한 뒤 `predict --split all`로 평가할 수 있습니다. 현재 prepare에는 split 열이 필요하므로 새 평가 목록은 모두 `test`로 지정합니다. 학습 manifest와 다른 파일이므로 `--allow-different-manifest`가 필요합니다. 출력 점수는 해당 `assay_type:endpoint`의 선별 점수이며 Kd 또는 보정된 결합 확률이 아닙니다. 학습한 task에 대해서만 예측하며, 다른 task의 행은 경고와 함께 건너뜁니다.

Pose 추론에는 fine-tuned backbone을 native 형식으로 export한 뒤 기존 Protenix 추론 도구에 연결합니다.

```bash
python -m fragment_ft export --checkpoint runs/joint/step_001000.pt \
  --base-checkpoint /data/protenix/checkpoint/protenix_base_default_v1.0.0.pt \
  --protenix-source /opt/Protenix --output checkpoints/fragment_backbone.pt
```

출력 상위 디렉터리는 미리 준비하세요. Binding head는 별도 fine-tuning checkpoint에 남습니다. `head` mode만 학습했다면 export되는 backbone은 원본과 같습니다. 이 프로그램의 `predict`는 binding 점수만 출력하며, native pose 생성과 RMSD 평가는 실제 Protenix 추론·구조 평가 도구에서 수행합니다.

## Native 통합의 현재 제한

확인한 upstream API commit은 `4c355be4553512f72453ecbfb65e69f4c35d1413`입니다. 다른 revision은 API 호환성을 먼저 확인합니다. `prepare`와 `train`은 같은 모델/feature 설정을 사용하세요([Upstream 설정](#upstream-설정)). Source 경로/commit과 checkpoint·manifest·tensor hash를 기록하며 tensor가 바뀌면 사용을 거부합니다.

실제 CIF 원자·ligand graph/입체화학 대응, structural branch의 native MSA 확장, CUDA/BF16 수치 동작, NCCL·다중 node 처리량은 사용 환경에서 검증해야 합니다. Synthetic은 한 비수정 protein chain + 한 비공유결합 ligand를 지원하고 identity atom permutation을 사용합니다. 금속/cofactor/변형 residue 등 복잡한 입력의 학습 지원을 일반화해서 주장하지 않습니다.

현재 학습은 한 microbatch에 한 complex를 처리하며 한 step은 rank당 `--accumulate`개 microbatch입니다. Rank별 seed로 표본을 추출하므로 rank 간 같은 관측을 뽑을 수 있습니다. 16 GPU는 gradient를 병렬 계산할 뿐 독립 표본 수를 늘리지 않습니다. Head만 학습하면 DDP 통신·중복 encoding 비용이 이득보다 클 수 있습니다. 1장 native smoke와 2장 비교를 통과한 뒤 16장으로 늘립니다.

공식 API 근거: [학습](https://github.com/bytedance/Protenix/blob/4c355be4553512f72453ecbfb65e69f4c35d1413/runner/train.py), [자체 구조 준비](https://github.com/bytedance/Protenix/blob/4c355be4553512f72453ecbfb65e69f4c35d1413/docs/prepare_training_data.md), [모델](https://github.com/bytedance/Protenix/blob/4c355be4553512f72453ecbfb65e69f4c35d1413/protenix/model/protenix.py).

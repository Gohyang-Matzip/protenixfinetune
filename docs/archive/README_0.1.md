# Fragment screening → Protenix fine-tuning

두 표적에 공통으로 screening한 fragment library를 위한 학습 프로그램입니다.
**Protenix를 설치하거나 checkpoint를 다운로드하지 않습니다.** 데이터 점검과 분할은 Python 표준 라이브러리만으로 실행됩니다. 학습·전처리는 이미 Protenix 실행 환경이 준비된 GPU 서버에서 수행합니다.

지원 학습:

| 설정 | 학습 내용 |
|---|---|
| `--mode head` | Protenix 고정, protein/ligand/pair 표현을 pooling한 작은 판별기 학습 |
| `--mode joint --negative-mode classifier` | 양성 구조 loss + 양성·음성 binding BCE |
| `--mode joint --negative-mode synthetic` | 기본 joint loss + 실제 apo에 ligand를 인위적으로 떨어뜨린 **음성 구조 loss** |

Synthetic 모드는 사용자 요청에 따른 비교 실험입니다. 인공 좌표는 실험 정답이나 물리적 평형 분포가 아닙니다. 기본값은 `classifier`이며, 불확실·미측정 결과는 어느 학습 모드에서도 음성으로 바꾸지 않습니다.

## 지금 실행할 수 있는 검사

프로젝트 디렉터리에서 실행합니다. 설치 명령은 필요하지 않습니다.

```bash
python3 -m fragment_ft --help
python3 -m fragment_ft validate examples/manifest.csv
python3 -m unittest discover -s tests -v
```

PyTorch가 있는 Python으로 테스트하면 미분, 실제 학습 루프, synthetic 배치, checkpoint 재개 검사도 실행합니다. 없는 환경에서는 해당 검사만 skip합니다. 예제의 단백질·label·그룹은 **형식 설명용 가상 데이터**이며 과학적 benchmark가 아닙니다.

## 1. 데이터 목록

[examples/manifest.csv](../../examples/manifest.csv)를 복사해 실제 데이터를 기록합니다.

| 열 | 의미 |
|---|---|
| `sample_id` | 각 표적–fragment 측정의 고유 ID. 영문/숫자/`_-.`만 사용 |
| `target_id` | 예: A 또는 B. construct가 다르면 별도 ID |
| `ligand_id` | 두 표적에서 같은 fragment는 같은 ID |
| `smiles` | canonical isomeric SMILES; 좌표 파일을 입력으로 넣지 않음 |
| `chem_group` | scaffold/화학 유사도로 미리 정한 그룹. 두 표적에 동일하게 사용 |
| `label` | `1`: 신뢰할 hit, `0`: 신뢰할 미검출, `uncertain`/`unknown`: 학습 제외 |
| `split` | `train`, `val`, `test`; 자동 분할 전에는 빈칸 |
| `structure_id` | 양성만: upstream 구조 index CSV의 `pdb_id`와 정확히 일치 |
| `protein_chain_id`, `ligand_chain_id` | 양성 복합체의 index CSV에 표시된 chain ID |

음성의 마지막 세 열은 비워 둡니다. 측정 실패·용해도 문제 등을 자동으로 label 0에 넣지 마세요. 반복 측정이 있으면 개별 `sample_id`를 주되 `ligand_id`와 `chem_group`은 유지합니다. 반복 수가 많은 화합물이 학습을 지배하지 않도록 manifest 작성 시 재현성 검토 후 대표 측정으로 정리하는 것을 권합니다.

```bash
python3 -m fragment_ft validate data/manifest.csv
python3 -m fragment_ft split data/manifest.csv \
  --output data/manifest.split.csv --seed 42 \
  --validation-fraction 0.15 --test-fraction 0.15
```

이미 split이 있으면 재분할을 거부합니다. 같은 fragment/chemistry group은 두 표적 모두에서 같은 split에 배치됩니다. 자동 분할은 label을 보면서 좋은 test를 고르지 않습니다. 양성이 적으면 고정 비율을 그대로 쓰지 말고 hit/group 수에 맞춰 사전에 분할을 설계하세요. 학습은 train과 validation 각각에 두 class가 없으면 중단합니다.

**화학 그룹 자체를 자동으로 계산하지는 않습니다.** 정확히 같은 SMILES/ID의 모순은 검출하지만, 다른 SMILES 표기로 표현한 동일 화합물이나 잘못 정한 유사도 그룹은 사용자가 표준화해야 합니다.

## 2. 표적별 공통 입력과 apo

[examples/targets.json](../../examples/targets.json)의 서열을 실제 서열로 바꿉니다. 각 target에는 Protenix `proteinChain` 항목을 넣습니다. `pairedMsaPath`, `unpairedMsaPath`, `templatesPath`를 추가하면 표적의 모든 fragment에 같은 입력을 사용합니다. 파일 경로는 targets JSON에 상대적이며, 로딩할 때 절대 경로로 변환됩니다. 로컬 MSA/template가 없으면 sequence-only 입력입니다.

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

`residue_offset`은 입력 서열의 1-based 위치에 더할 apo residue 번호 차이입니다. 예: 서열 첫 잔기가 apo의 101번이면 100. 불연속 번호는 `"residue_map": {"1": 101, "2": 104}`처럼 명시할 수 있습니다. 매핑은 정수이며 일대일이어야 합니다. 매핑된 원자 이름·잔기·원소를 검사하며, 충분한 Cα가 대응되지 않으면 중단합니다. Insertion code는 먼저 명시적으로 정리해야 합니다. Synthetic 생성은 현재 **한 개의 비수정 단백질 chain + 한 개의 비공유결합 ligand**를 지원합니다.

`apo` 항목은 **synthetic 정답 좌표 생성용**이며, 이것만으로 apo가 binding 모델의 template 입력이 되지는 않습니다. Apo를 공통 template로 조건화하려면 upstream 형식의 로컬 `templatesPath`도 제공해야 합니다. 개별 positive의 holo 구조를 그 fragment에만 template로 넣지 않습니다.

```bash
python3 -m fragment_ft inputs data/manifest.split.csv \
  --targets data/targets.json --output data/inputs.json
```

이 명령도 Protenix 없이 동작합니다. Fragment의 holo 좌표나 label은 native 입력 JSON에 넣지 않습니다.

## 3. GPU 서버에서 데이터 준비

이후 명령은 PyTorch와 Protenix의 기존 실행 환경, CCD/cache 등 로컬 자원이 필요합니다. 기준으로 확인한 upstream 소스는 `4c355be4553512f72453ecbfb65e69f4c35d1413`입니다. 이 프로그램은 본체를 복사하거나 설치하지 않고 `--protenix-source`로 기존 소스를 연결합니다. 실제 연결 경로·commit도 기록합니다.

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

- `.binding.pt`: 공통 target context와 fragment SMILES에서 만든 native 입력. 모든 label에 생성합니다.
- `.structure.pt`: 실험적 양성 복합체의 native feature/label. `structure_id`가 있는 양성에만 생성합니다.
- `.synthetic.pt`: 실제 apo 원자 좌표와 ligand reference conformer. 확실한 음성에만 생성하며, 거리·방향은 학습 시 정합니다.
- `preparation.json`: 모델명, 실제 소스, 설정, 입력과 tensor 파일의 SHA256. 학습 전에 cache 변경을 검출합니다.

Head만 학습하려면 positive index 관련 옵션과 `--synthetic`을 생략할 수 있습니다. 구조 전처리는 기본적으로 full complex이며, 실제 크기가 크면 `--crop-size 384` 등을 별도로 측정하세요. Synthetic은 full-complex 정답이므로 비교 시에는 우선 양성도 full complex로 준비합니다.

현재 **실험적 양성 및 synthetic 구조 학습 branch는 dummy MSA/template**를 사용합니다. Binding branch만 targets에 지정한 로컬 MSA/template를 사용합니다. 구조 fine-tuning에서 전체 native MSA 학습 파이프라인까지 확장한 버전은 아닙니다. 물·금속·assembly 처리와 원자 대응은 실제 구조로 먼저 확인해야 합니다.

## 4. 기본 모델과 인공 비결합 모델 비교

먼저 작은 판별기만 학습합니다. Backbone은 고정하고 pooling된 표현을 메모리에 캐시합니다. Native MSA sampling은 eval에서도 난수를 쓰므로, head 학습과 평가에서는 label/split과 무관한 입력별 고정 seed를 사용합니다. 재시작 후에도 같은 표현을 계산하며 외부 학습 RNG를 바꾸지 않습니다.

```bash
python -m fragment_ft train data/manifest.split.csv \
  --features prepared/all --base-checkpoint /data/protenix/checkpoint/protenix_base_default_v1.0.0.pt \
  --protenix-source /opt/Protenix --mode head --output runs/head \
  --steps 1000 --eval-every 50 --device cuda
```

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
classifier: BCE(양성·음성) + structure_weight × L_structure(양성)
synthetic:  BCE(양성·음성) + structure_weight × [L_structure(양성)
                                             + negative_weight × L_structure(인공 음성)]
```

양성 structure sample은 binding sample과 독립적으로 뽑습니다. Synthetic에서는 음성 structure sample을 추가로 뽑으므로 기본 비교군의 양성 학습 빈도를 줄이지 않습니다. 추가 forward 때문에 계산 비용은 더 듭니다. Native 구조 loss 전체를 인공 좌표에 적용하는 ablation이며, 좌표 loss만 단독으로 바꾸는 실험은 아닙니다.

`synthetic-distance`는 단백질과 ligand의 bounding sphere 사이 최소 여유 거리(Å)입니다. 관측된 receptor 좌표는 유지하고 ligand를 강체 회전·이동하므로 내부 결합 길이는 보존됩니다. 각 step/rank에 따라 배치를 재현 가능하게 바꿉니다. 20/40/80 Å와 다른 placement seed로 비교하면 배치 방식에 대한 민감도를 확인할 수 있습니다. 이 거리들 자체에 생물물리학적 정답이라는 의미는 없습니다.

LR, step 수, loss weight는 출발값이며 데이터 규모에 맞춰 validation으로 정해야 합니다. 자동 early stopping은 없고, 매 평가 시 저장된 `macro_validation_log_loss`로 checkpoint를 선택합니다. `test`는 학습·checkpoint 선택에 사용하지 않습니다.

단일 node 16 GPU 실행은 같은 명령 앞에 다음 launcher를 사용합니다.

```bash
torchrun --standalone --nproc_per_node=16 -m fragment_ft train ...
```

실제 cluster가 8장 × 2 node라면 torchrun의 `--nnodes`, `--node_rank`, `--master_addr`, `--master_port`를 cluster 구성대로 지정합니다. 먼저 1장으로 native smoke test를 통과한 뒤 확장하세요. 기본 kernel은 `torch`이며, 기존 환경에서 검증한 가속 kernel은 `--upstream-config examples/upstream_config.json`으로 설정합니다.

## 5. 저장·재개·평가

Checkpoint는 **학습한 parameter delta, head, optimizer, step, 데이터/기반 weight hash**를 저장합니다. Frozen 기반 weight는 복제하지 않으므로 원본 checkpoint가 계속 필요합니다. 학습 mode, prefix, seed, GPU 수 등은 재개 시 같아야 합니다. 매 step의 난수 seed를 다시 정하므로 평가 호출이 이후 학습 RNG를 바꾸지 않습니다.

```bash
# 앞선 train 명령과 같은 설정에 추가; steps는 최종 총 step 수입니다.
--resume runs/joint/step_000500.pt --steps 1000
```

기존 결과를 덮어쓰지 않습니다. 과거 checkpoint에서 다른 경로로 이어가려면 새 `--output`을 사용하세요. 최신 checkpoint 파일은 저장 중 임시 `.partial`에서 완성 후 변경되며, 실패한 파일도 임의 삭제하지 않습니다.

선택한 checkpoint를 공통 test에 평가합니다.

```bash
python -m fragment_ft predict data/manifest.split.csv \
  --features prepared/all --checkpoint runs/joint/step_001000.pt \
  --base-checkpoint /data/protenix/checkpoint/protenix_base_default_v1.0.0.pt \
  --protenix-source /opt/Protenix --split test --output runs/joint_test.json
```

Synthetic checkpoint도 같은 방법으로 `runs/synthetic40_test.json`에 평가한 다음:

```bash
python -m fragment_ft compare runs/joint_test.json runs/synthetic40_test.json \
  --output runs/comparison.json --top-k 20
```

`compare`는 평가 sample/label/split이 다르면 거부하며 Protenix 없이 동작합니다. 표적별 average precision(AP), Brier, log loss, precision/recall/enrichment@K를 제공합니다. AP는 score tie를 묶어 계산하고, top-K 경계의 tie는 동일 확률 선택의 기대 hit 수를 사용합니다. AP는 사다리꼴 PR-AUC와 구분합니다. 한 class가 없는 표적의 지표는 제한적으로 해석해야 합니다.

새 fragment는 label을 `unknown`으로 작성하고, 공통 target context로 다시 prepare한 뒤 `predict --split all`로 평가할 수 있습니다. 현재 prepare에는 split 열이 필요하므로 새 평가 목록은 모두 `test`로 지정합니다. 출력 점수는 X-ray hit 선별 점수이며 Kd 또는 검증된 확률이 아닙니다.

Pose 추론에는 fine-tuned backbone을 native 형식으로 export한 뒤 기존 Protenix 추론 도구에 연결합니다.

```bash
python -m fragment_ft export --checkpoint runs/joint/step_001000.pt \
  --base-checkpoint /data/protenix/checkpoint/protenix_base_default_v1.0.0.pt \
  --protenix-source /opt/Protenix --output checkpoints/fragment_backbone.pt
```

출력 상위 디렉터리는 미리 준비하세요. Binding head는 별도 fine-tuning checkpoint에 남습니다. `head` mode만 학습했다면 export되는 backbone은 원본과 같습니다. 이 프로그램의 `predict`는 binding 점수만 출력하며, native pose 생성과 RMSD 평가는 실제 Protenix 추론·구조 평가 도구에서 수행합니다.

## 검증 범위

- 로컬에서 Protenix 설치 없이 manifest/분할/CLI 검사를 실행했습니다.
- 기존 PyTorch 환경에서 작은 대체 backend로 gradient, positive/synthetic 분기, 실제 학습 루프, 저장·재개 동등성을 검사했습니다.
- 연결 코드는 공식 Protenix의 config, featurizer, trunk, structure loss API를 읽고 작성했습니다.
- **실제 Protenix checkpoint를 로드한 실행, 실제 apo/CIF 전처리, CUDA/BF16 및 다중 GPU 성능은 미검증입니다.** 사용자 제한에 따라 본체/의존성을 설치하지 않았고 실제 구조 데이터도 아직 없습니다.
- Synthetic의 atom permutation은 identity로 제한합니다. 실제 양성은 native symmetry 처리를 유지합니다. 인공 좌표의 atom naming에 대한 추가 민감도는 별도 확인 대상입니다.

관련 근거: [공식 학습](https://github.com/bytedance/Protenix/blob/4c355be4553512f72453ecbfb65e69f4c35d1413/runner/train.py), [자체 데이터 준비](https://github.com/bytedance/Protenix/blob/4c355be4553512f72453ecbfb65e69f4c35d1413/docs/prepare_training_data.md), [모델 API](https://github.com/bytedance/Protenix/blob/4c355be4553512f72453ecbfb65e69f4c35d1413/protenix/model/protenix.py).

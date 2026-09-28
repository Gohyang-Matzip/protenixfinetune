# Screening data → Protenix fine-tuning

GitHub: https://github.com/Gohyang-Matzip/protenixfinetune

공개 fragment screening·drug repurposing 결과와 자체 실험을 함께 사용하는 **범용 다중 표적 학습 프로그램**입니다. 양성 구조와 실험 양성·음성을 구분해 학습하고, 공개 데이터 사전학습에서 개인 데이터 fine-tuning으로 이어갈 수 있습니다.

**Protenix를 설치하지 않습니다.** 데이터 수집·검증·분할은 Python 3.9+ 표준 라이브러리로 실행됩니다(3.9.6에서 테스트). 학습은 기존 PyTorch/Protenix 환경을 연결합니다. 공개 데이터 다운로드는 명시적인 `fetch` 명령에서만 수행합니다.

| 기능 | 구현 |
|---|---|
| 공개 데이터 | PubChem AID 다운로드, CSV mapping, LIT-PCBA active/inactive import, 원본·SHA256 기록 |
| 라벨 | assay 종류·endpoint·농도·QC·출처 보존; 불확실·실패·미측정 제외; 같은 관측의 상충 label 거부 |
| 학습 | frozen head 또는 양성 구조 + 다중 task 판별; task/표적/assay 균형 sampling |
| 전이 | `--init-checkpoint`로 공유 표현 전달, 새 task 출력 초기화; 별도 `--resume` |
| 평가 | cold chemistry / cold target / both 분할, split·assay별 AP·AUROC·enrichment, 동일 평가목록 비교, corpus 간 누출 감사 |
| 인공 음성 | 실제 apo에 ligand를 멀리 배치하는 opt-in 구조 ablation |

`xray:hit`, `direct_binding:binding`, `biochemical:inhibition`은 각각 다른 출력입니다. Cell viability 같은 phenotypic 관측은 원본·manifest에 보존하지만 protein 학습에서 제외합니다. FDA 승인 여부나 assay의 `Inactive`만으로 비결합을 판정하지 않습니다.

## 바로 실행하는 예제

프로젝트 디렉터리에서 아래 명령을 실행합니다. 모델·패키지 설치가 필요 없습니다. 예제 147개 관측, 6개 표적, 12개 화합물은 **형식 검증용 가상 데이터**입니다. 출력은 새 임시 디렉터리 `$OUT`에 씁니다.

```bash
OUT=$(mktemp -d)
python3 -m fragment_ft import-csv examples/general_screen.csv \
  --mapping examples/general_mapping.json --output "$OUT/demo_import"
python3 -m fragment_ft split "$OUT/demo_import/manifest.csv" \
  --strategy both --seed 42 --output "$OUT/demo_both.csv"
python3 -m fragment_ft audit "$OUT/demo_both.csv" --output "$OUT/demo_both.audit.json"
python3 -m fragment_ft inputs "$OUT/demo_both.csv" \
  --targets examples/general_targets.json --output "$OUT/demo_inputs.json"
cmp "$OUT/demo_both.csv" data/demo_both.csv && echo "committed demo split reproduced"
python3 -m unittest discover -s tests -v
```

모든 출력 명령은 기존 파일을 덮어쓰지 않으므로 재실행할 때는 새 `OUT`을 만드세요. `both`는 두 축을 따로 나눈 뒤 train/train, val/val, test/test 조합만 사용하고 나머지를 `excluded`로 보존합니다. 각 task의 train/val 양성·음성 수는 `audit`에서 확인합니다.

테스트는 93개입니다. PyTorch·RDKit가 없는 Python은 Torch 의존 테스트 21개와 RDKit 테스트 1개를 skip하고, PyTorch·RDKit 환경(Python 3.9.6, Torch 2.8.0 CPU, RDKit 2025.09.2)에서는 모두 실행해 통과했습니다. 실제 Protenix checkpoint, CIF/apo 전처리, CUDA/BF16·NCCL 다중 GPU는 아직 실행 검증하지 않았습니다. [검증 기록](docs/VALIDATION.md)을 확인하세요.

설치하면 `fragment-ft` 명령과 RDKit가 함께 설치됩니다. RDKit로 `audit`와 `prepare`가 SMILES 표기가 달라도 같은 분자(InChIKey)를 찾아 chem_group 불일치를 검사합니다. 학습 환경에서는 `-e '.[train]'`으로 torch, numpy도 함께 설치합니다.

```bash
python3 -m pip install --upgrade pip   # 편집 설치(-e)에는 pip 21.3 이상 필요
python3 -m pip install -e .
```

설치 없이 `python3 -m fragment_ft`로 실행할 수도 있습니다. 이때 RDKit가 없으면 `audit`는 분자 동일성 검사만 건너뛰고 `molecule_identity.checked: false`로 기록하며, `prepare`는 거부합니다. Protenix와 모델 weight는 어느 경우에도 설치·다운로드하지 않습니다.

## 저장소의 `data/`

`data/`는 저장소와 함께 GitHub에 공개됩니다. 위 예제 명령으로 만든 가상 demo 출력과 PubChem API smoke 다운로드만 들어 있습니다([출처·조건](data/NOTICE)). 비공개·미발표 자료, 원본 CIF, 개인 manifest는 `data/`에 두지 말고 gitignore된 `private/`나 저장소 밖에 두세요. `prepared/`, `runs/`, `checkpoints/`와 `*.pt`도 gitignore 대상입니다.

## 문서

- [데이터 수집과 라벨 규칙](docs/DATA.md): PubChem·LIT-PCBA·NCATS·PRISM, CSV schema, 그룹 분할, 원본 추적.
- [학습·전이·평가 실행](docs/TRAINING.md): 서버 준비, task head, joint/synthetic 비교, checkpoint, torchrun.
- [4개월 연구·검증 계획](FINETUNING_PLAN.md): 공개 사전학습, cold 평가, 두 표적 적용, GPU 사용 기준.
- [검증 기록](docs/VALIDATION.md): 실제로 실행한 검사, 테스트 수, 남은 실제 환경 검증.
- [보관 문서](docs/archive/): 이전 README·구현 기록·연구계획. 예: [0.2.0 구현 상태](docs/archive/implementation_0.2.md), [이전 두 표적 연구계획](docs/archive/two_target_research_plan_20260916.md).

전체 명령은 `python3 -m fragment_ft --help`에서 볼 수 있습니다. 기존 [간단한 manifest](examples/manifest.csv)도 지원하며, assay 열이 없으면 기존 `xray:hit` 의미를 사용합니다. 새 공개 데이터에는 확장 schema를 사용하세요.

프로젝트 라이선스는 Apache-2.0입니다([LICENSE](LICENSE)). `data/smoke/`의 PubChem 자료는 이 라이선스로 재허락하지 않으며, 출처와 이용 정책은 [NOTICE](data/NOTICE)에 있습니다.

# Screening data → Protenix fine-tuning

공개 fragment screening·drug repurposing 결과와 자체 실험을 함께 사용하는 **범용 다중 표적 학습 프로그램**입니다. 양성 구조와 실험 양성·음성을 구분해 학습하고, 공개 데이터 사전학습에서 개인 데이터 fine-tuning으로 이어갈 수 있습니다.

**Protenix를 설치하지 않습니다.** 데이터 수집·검증·분할은 Python 3.11+ 표준 라이브러리로 실행됩니다. 학습은 기존 PyTorch/Protenix 환경을 연결합니다. 공개 데이터 다운로드는 명시적인 `fetch` 명령에서만 수행합니다.

| 기능 | 구현 |
|---|---|
| 공개 데이터 | PubChem AID 다운로드, CSV mapping, LIT-PCBA active/inactive import, 원본·SHA256 기록 |
| 라벨 | assay 종류·endpoint·농도·QC·출처 보존; 불확실·실패·미측정 제외 |
| 학습 | frozen head 또는 양성 구조 + 다중 task 판별; task/표적/assay 균형 sampling |
| 전이 | `--init-checkpoint`로 공유 표현 전달, 새 task 출력 초기화; 별도 `--resume` |
| 평가 | cold chemistry / cold target / both 분할, assay별 지표, 동일 평가목록 비교 |
| 인공 음성 | 실제 apo에 ligand를 멀리 배치하는 opt-in 구조 ablation |

`xray:hit`, `direct_binding:binding`, `biochemical:inhibition`은 각각 다른 출력입니다. Cell viability 같은 phenotypic 관측은 원본·manifest에 보존하지만 protein 학습에서 제외합니다. FDA 승인 여부나 assay의 `Inactive`만으로 비결합을 판정하지 않습니다.

## 바로 실행하는 예제

프로젝트 디렉터리에서 아래 명령을 실행합니다. 모델·패키지 설치가 필요 없습니다. 예제 147개 관측, 6개 표적, 12개 화합물은 **형식 검증용 가상 데이터**입니다.

```bash
mkdir -p data
python3 -m fragment_ft import-csv examples/general_screen.csv \
  --mapping examples/general_mapping.json --output data/demo_import
python3 -m fragment_ft split data/demo_import/manifest.csv \
  --strategy both --seed 42 --output data/demo_both.csv
python3 -m fragment_ft audit data/demo_both.csv --output data/demo_audit.json
python3 -m fragment_ft inputs data/demo_both.csv \
  --targets examples/general_targets.json --output data/demo_inputs.json
python3 -m unittest discover -s tests -v
```

같은 출력 경로로 재실행하면 덮어쓰기를 거부합니다. 새 출력 경로를 사용하세요. `both`는 두 축을 따로 나눈 뒤 train/train, val/val, test/test 조합만 사용하고 나머지를 `excluded`로 보존합니다. 각 task의 train/val 양성·음성 수는 `audit`에서 확인합니다.

PyTorch가 없는 Python은 학습 테스트만 skip합니다. 기존 PyTorch 환경에서는 22개 테스트를 모두 실행합니다. 실제 Protenix checkpoint, CIF/apo 전처리, CUDA/BF16·다중 GPU는 아직 실행 검증하지 않았습니다. [검증 기록](docs/VALIDATION.md)을 확인하세요.

## 문서

- [데이터 수집과 라벨 규칙](docs/DATA.md): PubChem·LIT-PCBA·NCATS·PRISM, CSV schema, 그룹 분할, 원본 추적.
- [학습·전이·평가 실행](docs/TRAINING.md): 서버 준비, task head, joint/synthetic 비교, checkpoint, torchrun.
- [4개월 연구·검증 계획](FINETUNING_PLAN.md): 공개 사전학습, cold 평가, 두 표적 적용, GPU 사용 기준.
- [구현 범위와 검증 상태](IMPLEMENTATION_PLAN.md): 구현 항목과 남은 실제 환경 검증.
- [이전 두 표적 연구계획](docs/archive/two_target_research_plan_20260916.md): 변경 전 문서 보관본.

전체 명령은 `python3 -m fragment_ft --help`에서 볼 수 있습니다. 기존 [간단한 manifest](examples/manifest.csv)도 지원하며, assay 열이 없으면 기존 `xray:hit` 의미를 사용합니다. 새 공개 데이터에는 확장 schema를 사용하세요.

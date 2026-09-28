# 데이터 수집과 관측 라벨

학습 단위는 **표적 construct × 화합물 × assay 관측**입니다. 공급자의 active/inactive를 원본과 함께 보존하고, 검토한 mapping으로 task별 label을 만듭니다. 본 프로그램은 논문에서 음성 목록을 자동 추론하거나, 측정되지 않은 protein–compound 조합을 음성으로 생성하지 않습니다.

> 저장소의 `data/`는 GitHub에 함께 공개됩니다. 아래 예시의 `data/...` 경로에는 공개해도 되는 자료만 두고, 비공개·미발표 자료는 gitignore된 `private/`나 저장소 밖 경로를 사용하세요.

## 어떤 음성을 사용하는가

| 관측 | `assay_type:endpoint` 예시 | 처리 |
|---|---|---|
| X-ray에서 신뢰할 ligand density 관측/미검출 | `xray:hit` | QC 통과 관측만 binary 학습 |
| 직접 결합 assay의 양성/음성 | `direct_binding:binding` | 검출한계·판정 규칙을 기록해 별도 task로 학습 |
| 효소 억제/활성 assay | `biochemical:inhibition` | 기능적 결과 task; 직접 결합 label과 분리 |
| 세포 생존·증식 변화 | `phenotypic:viability` | 보관·감사만 수행, protein 학습·예측 제외 |
| 실험 실패, 불확실, 미측정 | 원래 task 유지 | `quality=fail/uncertain` 또는 `label=uncertain/unknown` |

FDA 승인약도 해당 assay에서 유효하게 측정된 결과가 있어야 음성으로 사용할 수 있습니다. 약물 승인 상태 자체는 학습 label이 아닙니다. X-ray 미검출도 모든 조건에서의 비결합을 뜻하지 않습니다. 음성에서 protein apo 좌표는 존재할 수 있지만 ligand 상대 좌표의 실험 정답은 없습니다.

## 공통 manifest

CSV 헤더 중 처음 여섯 열은 필수입니다. `write_manifest`는 모든 표준 열을 출력하며 비어 있는 선택 열은 읽을 때 생략합니다. 아래 표에 없는 열은 `notes`처럼 무해해 보여도 오류입니다. 무시된 `Quality`/`AssayType` 열 때문에 QC 실패·phenotypic 관측이 `xray:hit` 학습 label이 되는 일을 막기 위해서입니다. 대소문자·공백·하이픈만 다른 이름에는 바꿀 표준 이름을 알려 줍니다.

| 필드 | 규칙 |
|---|---|
| `sample_id` | 관측 고유 ID; 영문·숫자·`_-.`. Packet 파일 이름이므로 대소문자만 다른 ID도 중복입니다. 동일 측정 중복 수집은 사전에 제거 |
| `target_id` | construct별 ID. 서열은 targets JSON에서 제공 |
| `ligand_id`, `smiles` | 통일한 화학 ID와 isomeric SMILES. 같은 ID의 SMILES/그룹은 모든 표적에서 일치 |
| `chem_group` | scaffold/화학 유사성 그룹; 전체 corpus에서 먼저 정의 |
| `label` | 문자열 `1`, `0`, `uncertain`, `unknown` |
| `split` | 분할 전 빈칸, 이후 `train`, `val`, `test`, both의 교차쌍은 `excluded` |
| `structure_id`, `protein_chain_id`, `ligand_chain_id` | 실험 양성 복합체만 지정. upstream index의 ID/chain과 일치 |
| `target_group` | 서열·pocket 유사성을 고려한 표적 그룹 |
| `assay_id`, `assay_type`, `endpoint` | 출처 내 assay ID, 표의 네 종류, 소문자 snake_case endpoint |
| `campaign_id`, `source`, `source_url` | 실험 캠페인, 공급자, 원본 페이지/식별 링크 |
| `quality` | `pass`, `fail`, `uncertain`. `pass`만 학습 |
| `weight` | 선택, 기본 1, 범위 0–1. BCE에만 적용; 0이면 학습·지표 제외 |
| `concentration`, `concentration_unit` | 선택, 양수와 `M/mM/uM/nM/pM` |
| `measurement_value`, `measurement_relation`, `measurement_unit` | 선택, 수치와 `=`, `<`, `<=`, `>`, `>=` 관계·단위. 검열값을 그대로 보존 |
| `split_strategy` | `chemistry`, `target`, `both` |

확장 열을 하나라도 쓰면 `target_group`부터 `quality`까지 8개 관측 metadata 열이 모두 필요합니다. 이전 최소 schema는 X-ray hit, QC pass로 해석하므로 새 공개 결과에 사용하지 마세요. 수치 열(`weight`, `concentration`, `measurement_value`)의 오류는 sample_id와 열 이름을 알려 줍니다.

현재 모델은 **농도 조건부 모델이나 affinity 회귀 모델이 아닙니다.** 농도·수치는 추적용이며 head 입력에는 들어가지 않습니다. 농도·threshold 정책이 다른 결과를 한 endpoint로 무조건 합치지 말고, 동등한 기준으로 사전 정리하거나 `binding_10um`처럼 별도 endpoint를 사용하세요. `Kd > C`를 임의의 정량 Kd로 변환하지 않습니다. 같은 `source/assay_id/target_id`는 하나의 endpoint/campaign이어야 하므로 여러 endpoint는 별도 assay ID로 구별합니다.

화학 표준화, salt/tautomer/stereochemistry 통일, scaffold/fingerprint 그룹, 서열·pocket 그룹은 사용자가 검토해 제공합니다. 프로그램은 정확히 같은 ID/SMILES와 그룹의 모순을 검사하며 유사도 계산까지 자동 수행하지 않습니다. 추가 동일성 검사는 다음과 같습니다.

- **상충 label**: 학습 가능한 관측 중 같은 `(source, assay_id, target_id, ligand_id)`에 `0`과 `1`이 함께 있으면 manifest를 읽는 모든 명령이 거부합니다. 검토 후 한쪽을 `uncertain`으로 표시하세요. 같은 label의 반복 관측은 허용하며 `validate`/`split`/import 요약의 `replicate_observations`로 보고합니다. 다른 source/assay의 같은 화합물–표적 결과는 별도 관측이므로, 출처 간 중복은 사용자가 정리합니다.
- **분자 동일성**: RDKit가 있으면 `audit`가 SMILES 표기가 달라도 같은 분자(InChIKey)가 하나의 `chem_group`만 갖는지 검사합니다. RDKit가 없으면 `molecule_identity.checked`가 `false`입니다. `pip install -e .`로 설치하면 RDKit가 함께 설치됩니다(설치 없이 소스에서 실행한다면 `python3 -m pip install rdkit`). 공개 데이터(PubChem은 Kekulé SMILES)와 자체 라이브러리(대개 aromatic SMILES)를 합칠 때는 RDKit를 설치한 환경에서 `audit`를 실행하세요. `prepare`는 RDKit(Protenix 의존성)를 요구하고 이 검사를 통과해야 합니다.
- **단백질 동일성**: `inputs`/`prepare`는 같은 단백질 서열이 서로 다른 `target_group`에 있으면 거부합니다. 같은 construct는 같은 target_group으로 합치세요.

## CSV 가져오기

[실행 가능한 mapping](../examples/general_mapping.json)과 [원본 예제](../examples/general_screen.csv)를 참고합니다.

```bash
python3 -m fragment_ft import-csv public.csv \
  --mapping reviewed_mapping.json --compounds reviewed_compounds.csv \
  --output data/imported_campaign
```

`--compounds`는 raw 화합물 ID를 공통 ID/SMILES/그룹으로 연결할 때만 사용합니다. `source_id` 열이 있으면 그 값을, 없으면 `ligand_id`를 key로 씁니다. Key의 앞뒤 공백은 제거하며, 빈 table, 빈 key·중복 key, 필드 수가 header와 다른 행은 오류입니다. 쉼표가 있는 SMILES는 따옴표로 감쌉니다.

```csv
source_id,ligand_id,smiles,chem_group
101,L001,CCO,G001
102,L002,CCN,G002
```

Mapping JSON의 구성:

```json
{
  "columns": {"ligand_id": "compound_id", "quality": "QC"},
  "constants": {
    "target_id": "T001", "target_group": "family_001",
    "assay_id": "screen_001", "assay_type": "direct_binding",
    "endpoint": "binding", "campaign_id": "study_001",
    "source": "reviewed_export", "source_url": "https://example.org/study"
  },
  "outcome_column": "outcome",
  "outcomes": {"Active": "1", "Inactive": "0", "Inconclusive": "uncertain", "NotTested": "unknown"},
  "quality_map": {"OK": "pass", "Failed": "fail", "Pending": "uncertain"}
}
```

- `columns`는 manifest 열 → 원본 열, `constants`는 검토한 고정 metadata입니다. 두 항목에 같은 필드를 쓰지 않습니다.
- `outcomes`를 명시해야 합니다. 미등록 outcome은 자동 음성화하지 않고 오류로 중단합니다. Label을 `columns/constants`로 직접 지정할 수 없습니다.
- 최상위 key는 `columns`, `constants`, `outcome_column`, `outcomes`, `quality_map`, `skip_rows`, `conflicting_labels`만 허용합니다. 오타 key, 중복 key, 문자열이 아닌 값(숫자·`null`)은 오류입니다. 연결한 원본 열, `outcome_column`, `skip_rows`의 열은 입력 header에 있어야 합니다.
- QC 열이나 `quality` constant를 연결하지 않으면 모든 행이 `quality=uncertain`이 되어 학습에서 제외되며, 이때 `quality_map`은 적용되지 않습니다. 원본 QC가 없다는 이유로 일괄 `pass`로 바꾸지 않습니다.
- `"conflicting_labels": "error"`(기본)는 같은 `(source, assay_id, target_id, ligand_id)`의 학습 가능 행에 `0`과 `1`이 함께 있으면 import를 중단합니다. `"uncertain"`이면 해당 key의 학습 가능 행을 모두 `uncertain`으로 바꿔 학습에서 제외합니다. 어느 쪽도 한 label을 골라 음성으로 만들지 않습니다.
- `sample_id`를 생략하면 입력 파일 내용의 SHA256, 행 번호, 변환된 관측에서 결정적으로 생성합니다. 파일 경로는 쓰지 않으므로 파일을 옮기거나 복사해도 ID가 같고, 같은 파일을 같은 mapping으로 다시 import하면 같은 ID가 생겨 `merge`가 중복으로 거부합니다. 파일 내용이나 mapping이 바뀌면 ID도 바뀌므로 장기 corpus에는 고정 관측 ID를 권합니다.
- 결과 디렉터리에는 `manifest.csv`, `raw.jsonl`, `provenance.json`이 생깁니다. `raw.jsonl`의 각 줄은 `{"row", "sample_id", "raw"}`로 원본 행과 manifest 행을 연결합니다(건너뛴 행은 `sample_id: null`). `provenance.json`은 입력 파일(지정한 경로 그대로)의 SHA256, mapping과 `mapping_sha256`, compound table hash, 출력 manifest hash, 건너뛴 행 수, `conflicting_labels` 정책, `replicate_keys`·`conflicting_keys` 수를 기록합니다.
- Duplicate CSV header, 필드 수가 header와 다른 행, 모순된 compound join, 알 수 없는 outcome, 기존 출력 경로는 오류입니다. 원본 CSV는 별도 보존하세요.

공개 import가 끝나면 **분할 전에** corpus를 합칩니다.

```bash
python3 -m fragment_ft merge data/study1/manifest.csv data/study2/manifest.csv \
  --output data/corpus.csv
python3 -m fragment_ft validate data/corpus.csv
```

분할한 manifest와 분할하지 않은 manifest를 섞어 합치면 거부합니다. 이미 분할한 corpus를 합칠 때는 같은 전략과 전역 그룹 배정을 사용해야 합니다. `merge`는 입력 파일별 SHA256과 요약을 출력하므로 corpus 버전 기록에 남기세요. 자동으로 라벨 충돌을 다수결 처리하거나 자료를 삭제하지 않습니다.

## 공개 자료별 사용 경로

| 자료 | 가져오기 | 해석 경계 |
|---|---|---|
| [PubChem BioAssay](https://pubchem.ncbi.nlm.nih.gov/docs/bioassays) | `fetch-pubchem` → description 검토 → `import-csv` | AID별 측정 의미·QC·outcome 정책 확인 |
| [LIT-PCBA](https://lab.drugdesign.unistra.fr/datasets/lit-pcba/) | active/inactive `.smi` → SID join → `import-lit` | 공급자 active/inactive가 직접 결합을 뜻하는지 별도 검토 |
| [NCATS repurposing 자료](https://zenodo.org/records/20429398) | 원본 CSV와 data dictionary 명시 다운로드 → `import-csv` | 표적·assay·농도·endpoint별 mapping 작성 |
| [PRISM](https://depmap.org/repurposing/) | 원본 보존 또는 phenotypic import | 세포 반응을 protein binding 음성으로 변환하지 않음 |
| 공개 X-ray fragment screening | full screening/QC 목록과 구조 ID 연결 → `import-csv` | 구조 저장소에 없는 화합물을 자동 음성으로 만들지 않음 |

Dataset 페이지의 라이선스·재배포 조건은 corpus 버전에 함께 기록합니다. 다운로드가 성공해도 label 검토가 끝난 것은 아닙니다. 현재 연결부는 범용 CSV/SMILES import이며 NCATS·PRISM의 모든 버전에 대한 자동 column 추론을 제공하지 않습니다.

### PubChem

```bash
python3 -m fragment_ft fetch-pubchem --aid YOUR_AID \
  --with-compounds --output data/pubchem_AID
```

실제 정수 AID를 사용합니다. Description JSON, assay CSV, 선택적 CID→SMILES 파일과 각 다운로드 영수증이 저장됩니다. `compounds.csv`의 `chem_group`은 의도적으로 비워 두므로 검토 후 채워야 합니다. CID 없는 substance는 자동 추론하지 않으며, 필요한 경우 별도 SID join을 제공합니다. 출력 디렉터리는 새로 만들어야 하므로 실패 후에는 새 `--output`을 사용합니다. 화합물 batch가 실패하면 `compounds.csv.partial`만 남고 `compounds.csv`는 만들지 않습니다.

PubChem CSV의 `PUBCHEM_RESULT_TAG`에는 설명 행이 포함될 수 있습니다. 원본을 확인하고 다음처럼 **정확한 값 목록**만 mapping에 추가합니다.

```json
{"skip_rows": {"PUBCHEM_RESULT_TAG": ["RESULT_TYPE", "RESULT_DESCR", "RESULT_UNIT", "RESULT_IS_ACTIVE_CONCENTRATION", "RESULT_ATTR_CONC_MICROMOL"]}}
```

이 snippet은 전체 mapping이 아닙니다. 보통 `ligand_id`는 `PUBCHEM_CID`, outcome 열은 `PUBCHEM_ACTIVITY_OUTCOME`에 연결합니다. 헤더명과 실제 outcome을 해당 파일에서 확인하세요. 건너뛴 설명 행도 `raw.jsonl`과 skipped count에 남습니다. 확인하지 않은 결과 문자열은 mapping에 추가하기 전까지 import되지 않습니다. 요청 형태의 공식 근거는 [PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)입니다.

### LIT-PCBA

```bash
python3 -m fragment_ft import-lit \
  --active data/lit/actives.smi --inactive data/lit/inactives.smi \
  --mapping data/lit/reviewed_mapping.json --compounds data/lit/compounds.csv \
  --output data/lit_imported
```

각 `.smi` 행은 `SMILES SID` 두 열입니다. Compound table의 `source_id`는 SID입니다. Mapping은 위와 같은 `constants`와 **명시적 `outcomes`**를 포함합니다. Importer가 SMILES/SID와 파일의 Active/Inactive 표식을 읽으므로 `columns/outcome_column`은 정해진 형식을 사용합니다. 검토 결과 Active도 불확실하면 `"Active":"uncertain"`을 사용하며 공급자 이름 때문에 다시 1로 바꾸지 않습니다.

Manifest의 SMILES는 `.smi` 파일의 값입니다. Compound table의 `smiles`는 `.smi` 값과 정확히 같거나 비어 있어야 하며, 다르면 import가 중단됩니다. 염 제거·표준화가 필요하면 import 전에 `.smi` 파일에서 수행하고 원본 파일은 따로 보존하세요. 같은 SID가 active와 inactive 목록에 모두 있으면(같은 파일을 두 번 지정한 경우 포함) mapping이나 `conflicting_labels`와 무관하게 거부하므로 목록을 먼저 고칩니다.

### 일반 다운로드

```bash
python3 -m fragment_ft fetch 'https://provider.example/data.csv' \
  --output data/raw.csv --sha256 EXPECTED_SHA256 --max-mb 512
```

실제 URL/공급자 checksum으로 바꿉니다. `--sha256`은 선택적입니다. 크기 한도·Content-Length·선택적 SHA256을 확인하고 `.receipt.json`에 URL/최종 URL/hash/크기/시각과 함께 `sha256_verified`(기대 SHA256과 대조했는지), `content_length`(서버가 알린 크기)를 기록합니다. `content_length`가 `null`이고 `sha256_verified`가 `false`이면 잘림 여부를 확인할 수 없었다는 뜻입니다. Redirect는 요청 전에 검사해 HTTPS는 HTTPS로만, HTTP는 HTTP(S)로만 따라갑니다. 실패한 `.partial`은 보존하고 성공본으로 승격하지 않습니다. 출력·`.partial`·영수증 중 하나라도 남아 있으면 같은 `--output`의 재다운로드를 거부하며 오류가 남은 파일을 알려 줍니다. `.partial`을 보관 위치로 옮기거나 새 `--output`을 지정한 뒤 다시 실행하세요. HTML 로그인 페이지나 공급자 오류 메시지의 의미를 자동 판단하지는 않으므로 원본 형식도 검토합니다.

## 일반화 평가 분할

```bash
python3 -m fragment_ft split data/corpus.csv --strategy chemistry --output data/cold_chemistry.csv
python3 -m fragment_ft split data/corpus.csv --strategy target --output data/cold_target.csv
python3 -m fragment_ft split data/corpus.csv --strategy both --output data/cold_both.csv
python3 -m fragment_ft audit data/cold_both.csv --output data/cold_both.audit.json
```

- `chemistry`: 전 표적에서 같은 화학 그룹을 같은 split에 배치합니다.
- `target`: 같은 표적 유사성 그룹을 같은 split에 배치합니다. 화합물의 재등장은 허용하는 평가입니다.
- `both`: 두 그룹 축을 각각 분리합니다. train/val/test가 일치하는 쌍만 사용하고 다른 쌍은 `excluded`로 남깁니다. 많은 관측이 학습에서 빠질 수 있습니다.

분할 비율(`--validation-fraction`, `--test-fraction`, 기본 각 0.15)은 **학습 가능한 관측 수에 대한 상한**입니다.

- 학습 가능한 관측이 있는 그룹만 나눕니다. Phenotypic·uncertain·fail·weight 0 관측만 있는 그룹은 그 축에서 항상 train입니다.
- 정렬한 그룹 목록을 seed로 섞은 뒤 그 순서대로 val, test 할당량에 들어가는 그룹만 채웁니다. 할당량보다 큰 scaffold/표적 그룹은 train에 남습니다.
- 들어가는 그룹이 없으면 가장 작은 그룹 하나를 배정하므로 val/test는 요청 비율보다 작을 수 있습니다. 예: 12행 표적들에 25% 할당량이면 2개가 아니라 1개 표적입니다. Val/test는 각각 최소 1개 그룹을 받고 train에는 최소 1개 그룹이 남습니다.
- `chemistry`/`both`에서 한 `structure_id`가 여러 chem_group에 걸치면 그 chem_group을 합치라는 오류로 중단합니다.
- 결과의 train/val/test 중 하나라도 학습 가능한 관측이 없으면 거부합니다. 작거나 성긴 표적×화학 행렬의 `both`에서 자주 발생하므로 chemistry나 target을 사용하세요.
- 이 알고리즘은 이전 버전과 달라 같은 seed·입력에서도 다른 배정이 나올 수 있습니다. 이미 분할한 manifest는 재분할을 거부하므로 기존 split 열이 보존됩니다. 저장소의 `data/demo_*.csv`는 새 코드로 다시 만들어도 동일했습니다.

각 분할 축에는 학습 가능한 관측이 있는 그룹이 최소 3개 필요합니다. **두 표적만으로 target/both의 독립 train/val/test를 만들 수 없습니다.** 공개 다중 표적 corpus에서 일반화를 검증하고, 두 표적 개인 데이터에는 chemistry split을 적용합니다.

`audit`는 그룹 누출, split별 사용량, task별 양성·음성 수와 분자 동일성(`molecule_identity`)을 확인합니다. 분할 전 manifest에는 split 명령을 먼저 실행하라는 오류를 냅니다. 자동 split은 label을 보고 test가 좋아지는 seed를 선택하지 않습니다. 각 task의 train/val에 두 class가 없으면 학습은 중단됩니다. 테스트 양성을 보고 seed를 반복 선택하지 말고 사전에 분할 정책과 최소 데이터량을 정하세요.

출처·캠페인·시간을 통째로 holdout하는 외부 평가는 manifest를 별도로 설계합니다. 프로그램이 자동 campaign/time split을 만드는 것은 아닙니다. Protenix 사전학습 데이터 중복, 유사 pocket, 화학 표준화 오류는 그룹 검사를 통과해도 남을 수 있습니다.

## Corpus 간 누출 감사

공개 사전학습 corpus와 개인 corpus처럼 따로 분할한 manifest는 서로의 split을 모릅니다. 전이 실험 전에 평가 corpus를 참조 corpus와 대조합니다.

```bash
python3 -m fragment_ft audit private/personal.split.csv \
  --against data/public.split.csv --output private/personal.cross.audit.json
```

`cross_corpus`는 이 manifest의 val/test 행이 참조 manifest의 train 행과 겹치는 SMILES, `ligand_id`, `chem_group`, `target_group`, `structure_id`의 개수와 예시(최대 20개)를 보고합니다. `leakage`는 이 manifest의 분할 전략이 holdout하는 축(chemistry: SMILES·ligand_id·chem_group, target: target_group, both: 모두)과 `structure_id`의 중복만 셉니다. 보고만 하며 exit 0이므로 결과를 보고 manifest를 고칩니다. 참조 manifest에는 train 행이 있어야 합니다. `ligand_id`·`chem_group` 비교는 두 corpus가 같은 명명 체계를 쓴다고 가정하므로 SMILES 중복도 함께 확인하세요. 유사 scaffold·유사 pocket은 정확한 값 비교로 잡히지 않습니다.

# 데이터 수집과 관측 라벨

학습 단위는 **표적 construct × 화합물 × assay 관측**입니다. 공급자의 active/inactive를 원본과 함께 보존하고, 검토한 mapping으로 task별 label을 만듭니다. 본 프로그램은 논문에서 음성 목록을 자동 추론하거나, 측정되지 않은 protein–compound 조합을 음성으로 생성하지 않습니다.

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

CSV 헤더 중 처음 여섯 열은 필수입니다. `write_manifest`는 모든 표준 열을 출력하며 비어 있는 선택 열은 읽을 때 생략합니다.

| 필드 | 규칙 |
|---|---|
| `sample_id` | 관측 고유 ID; 영문·숫자·`_-.`. 동일 측정 중복 수집은 사전에 제거 |
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

확장 열을 하나라도 쓰면 `target_group`부터 `quality`까지 8개 관측 metadata 열이 모두 필요합니다. 이전 최소 schema는 X-ray hit, QC pass로 해석하므로 새 공개 결과에 사용하지 마세요.

현재 모델은 **농도 조건부 모델이나 affinity 회귀 모델이 아닙니다.** 농도·수치는 추적용이며 head 입력에는 들어가지 않습니다. 농도·threshold 정책이 다른 결과를 한 endpoint로 무조건 합치지 말고, 동등한 기준으로 사전 정리하거나 `binding_10um`처럼 별도 endpoint를 사용하세요. `Kd > C`를 임의의 정량 Kd로 변환하지 않습니다. 같은 `source/assay_id/target_id`는 하나의 endpoint/campaign이어야 하므로 여러 endpoint는 별도 assay ID로 구별합니다.

화학 표준화, salt/tautomer/stereochemistry 통일, scaffold/fingerprint 그룹, 서열·pocket 그룹은 사용자가 검토해 제공합니다. 프로그램은 정확히 같은 ID/SMILES와 그룹의 모순을 검사하며 유사도 계산까지 자동 수행하지 않습니다. 다른 자료에서 같은 관측을 가져오는 중복이나 상충 결과는 자동 해결하지 않습니다.

## CSV 가져오기

[실행 가능한 mapping](../examples/general_mapping.json)과 [원본 예제](../examples/general_screen.csv)를 참고합니다.

```bash
python3 -m fragment_ft import-csv public.csv \
  --mapping reviewed_mapping.json --compounds reviewed_compounds.csv \
  --output data/imported_campaign
```

`--compounds`는 raw 화합물 ID를 공통 ID/SMILES/그룹으로 연결할 때만 사용합니다.

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
- QC를 제공하지 않으면 `uncertain`이 되어 학습에서 제외됩니다. 원본 QC가 없다는 이유로 일괄 `pass`로 바꾸지 않습니다.
- `sample_id`를 생략하면 파일 출처·행·변환 관측에서 결정적으로 생성합니다. 파일 이동·내용 변경·mapping 변경은 ID를 바꿀 수 있으므로 장기 corpus에는 고정 관측 ID를 권합니다.
- 결과 디렉터리에는 `manifest.csv`, `raw.jsonl`, `provenance.json`이 생깁니다. 원본 파일 hash, mapping, compound table hash, 출력 hash를 기록합니다.
- Duplicate CSV header, 모순된 compound join, 알 수 없는 outcome, 기존 출력 경로는 오류입니다. 원본 CSV는 별도 보존하세요.

공개 import가 끝나면 **분할 전에** corpus를 합칩니다.

```bash
python3 -m fragment_ft merge data/study1/manifest.csv data/study2/manifest.csv \
  --output data/corpus.csv
python3 -m fragment_ft validate data/corpus.csv
```

이미 분할한 corpus를 합칠 때는 같은 전략과 전역 그룹 배정을 사용해야 합니다. 자동으로 라벨 충돌을 다수결 처리하거나 자료를 삭제하지 않습니다.

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

실제 정수 AID를 사용합니다. Description JSON, assay CSV, 선택적 CID→SMILES 파일과 각 다운로드 영수증이 저장됩니다. `compounds.csv`의 `chem_group`은 의도적으로 비워 두므로 검토 후 채워야 합니다. CID 없는 substance는 자동 추론하지 않으며, 필요한 경우 별도 SID join을 제공합니다.

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

### 일반 다운로드

```bash
python3 -m fragment_ft fetch 'https://provider.example/data.csv' \
  --output data/raw.csv --sha256 EXPECTED_SHA256 --max-mb 512
```

실제 URL/공급자 checksum으로 바꿉니다. `--sha256`은 선택적입니다. 크기 한도·Content-Length·선택적 SHA256을 확인하고 `.receipt.json`에 URL/hash/시각을 기록합니다. 실패한 `.partial`은 보존하고 성공본으로 승격하지 않습니다. HTML 로그인 페이지나 공급자 오류 메시지의 의미를 자동 판단하지는 않으므로 원본 형식도 검토합니다.

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

각 분할 축에는 최소 3개 그룹이 필요합니다. **두 표적만으로 target/both의 독립 train/val/test를 만들 수 없습니다.** 공개 다중 표적 corpus에서 일반화를 검증하고, 두 표적 개인 데이터에는 chemistry split을 적용합니다.

`audit`는 그룹 누출, split별 사용량, task별 양성·음성 수를 확인합니다. 자동 split은 label을 보고 test가 좋아지는 seed를 선택하지 않습니다. 각 task의 train/val에 두 class가 없으면 학습은 중단됩니다. 테스트 양성을 보고 seed를 반복 선택하지 말고 사전에 분할 정책과 최소 데이터량을 정하세요.

출처·캠페인·시간을 통째로 holdout하는 외부 평가는 manifest를 별도로 설계합니다. 프로그램이 자동 campaign/time split을 만드는 것은 아닙니다. Protenix 사전학습 데이터 중복, 유사 pocket, 화학 표준화 오류는 그룹 검사를 통과해도 남을 수 있습니다.

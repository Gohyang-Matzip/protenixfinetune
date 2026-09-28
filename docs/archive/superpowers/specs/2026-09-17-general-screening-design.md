# 범용 screening 학습 설계

사용자가 승인한 방향: 공개 fragment/repurposing screening의 양성과 음성을 assay 의미를 보존해 학습하고, 미관측 단백질과 화합물에 일반화하는 프로그램. Protenix 설치와 weight 다운로드는 하지 않는다.

- 기존 CSV와 head/joint/synthetic 경로를 유지한다. 확장 관측 CSV는 assay 종류, endpoint, 품질, 출처, 농도/측정값, 표적 유사성 그룹을 보존한다.
- 공개 데이터는 명시적 fetch 명령으로 원본과 SHA256을 보존한다. PubChem AID 수집, 범용 CSV mapping/join, LIT-PCBA active/inactive SMILES import를 제공한다. 공급자의 active/inactive 해석은 사용자 검토한 mapping으로만 적용한다. 미지 outcome은 오류로 중단하고 실패/불확실 결과는 학습에서 제외한다.
- binary task는 `assay_type:endpoint`로 구분한다. X-ray, 직접 결합, biochemical task의 출력을 공유 표현 위에서 분리한다. Phenotypic 결과는 보존하지만 단백질 결합 학습·예측에서 제외한다. 실험 농도는 기록하며 현재 head는 농도 조건부 모델이 아니므로 endpoint 정의에 동일한 threshold 정책을 사용한다.
- 구조 supervision은 관측 양성 복합체만 사용한다. 인공 음성 비교는 QC를 통과한 X-ray 음성으로 한정한다.
- chemistry / target / both split을 제공한다. Both는 두 축을 독립적으로 분할한 뒤 같은 split의 교차점만 사용하고 나머지를 excluded로 남긴다. 사용자 제공 chem_group/target_group의 생물학적·화학적 타당성을 자동 증명하지 않는다.
- task/target/assay별 균형 sampling 및 품질 weight를 지원한다. 평가도 assay/task별로 분리하며 validation task macro log loss로 checkpoint를 고른다.
- 공개 checkpoint를 개인 데이터의 초기값으로 쓰는 init-checkpoint와 정확한 이어학습 resume를 구분한다. 태스크 대응을 보존하며 새로운 task 출력은 초기화한다.
- 데이터 수집·분할은 표준 라이브러리만 사용한다. 기존 PyTorch로 gradient, loss routing, 재개·전이, 통합 workflow를 검사한다. 실제 Protenix/GPU 검증 여부를 구분해서 기록한다.

현재 범위 밖: 자동 문헌 해석, 임의 target pair 음성 생성, phenotype-to-binding 변환, 자동 affinity 회귀, 자동 서열/pocket clustering, 대규모 corpus 실제 학습. 원본 다운로드만으로 label 품질이 검증되었다고 주장하지 않는다.

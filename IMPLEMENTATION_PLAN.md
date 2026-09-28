# 구현 상태 — 0.2.0

승인된 범용 screening 방향을 기존 프로그램에 구현했습니다. Protenix 설치·모델 다운로드는 수행하지 않았습니다.

- [x] 공개 데이터 명시적 fetch, PubChem AID+CID 수집, SHA256/원본 기록
- [x] 검토한 CSV mapping, compound join, LIT active/inactive import
- [x] Assay/endpoint/QC/출처/농도 schema와 실패·불확실·phenotypic 제외
- [x] Chemistry/target/both 분할, 제외 관측 보존과 task class coverage 감사
- [x] 다중 task 출력, task/target/assay sampling, 관측 BCE weight
- [x] 공개→개인 전이 초기화, task 대응, optimizer 재개, 동결한 전이 backbone 보존
- [x] 기존 양성 구조 학습과 X-ray 인공 음성 비교 유지
- [x] CLI 실행 예제, 한국어 데이터·학습·연구 문서와 이전 문서 보관
- [x] 회귀/학습 테스트, 데이터 CLI 실행, 작은 공개 API 실수집, 코드 검토
- [ ] 실제 Protenix checkpoint·CIF/apo native 실행
- [ ] 실제 corpus 성능, CUDA/BF16, 16 H100·다중 node 성능 검증

마지막 두 항목은 실제 환경·자료로 검증할 항목이며 구현 테스트 통과로 대체하지 않습니다. [검증 기록](docs/VALIDATION.md)에 확인한 사실과 한계를 기록합니다.

상세 [설계](docs/superpowers/specs/2026-09-17-general-screening-design.md), [구현 계획](docs/superpowers/plans/2026-09-17-general-screening.md), [이전 구현 기록](docs/archive/implementation_0.1.md).

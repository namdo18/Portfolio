</br>

# 프로젝트 포트폴리오
</br>

### 프로젝트1. [다양한 콘텐츠 소비 유도를 위한 웹소설 추천시스템 개발](https://github.com/namdo18/Attention4NovelRec)

Li et al.2022[[1](#reference)] 에서 제안된 세션 기반 웹소설 추천시스템 `NovelNet` 의 한계점 극복
- `NovelNet` 한계점
  - 반복 소비에 편향된 연구 조건에서 모델 학습, 다양한 콘텐츠 소비 유도라는 추천시스템 목적[[2](#reference)]에 부합하지 못함
  - 유효한 소비로 볼 수 없는 소설들까지 추천 대상으로 학습함에 따라, 비즈니스 실효성을 보장하지 못함
- 개발 방향 : 유효한 소비가 예상되는 새로운 소설을 추천하는 모델
- 모델 개발 및 구현
  - 학습할 유효소비 데이터 선별 및 과거에 소비한 소설들에 대한 마스킹 알고리즘 구현(모델 내부) 
  - `NovelNet`, `GRU4Rec`[[3](#reference)], `TransformerEncoder`[[4](#reference)] 기반의 모델 개발 및 구현
  - `Attention4NovelRec` : 사전학습된 장르정보를 이용하는 어텐션 메커니즘 기반의 별도의 모델 개발
- 프로젝트 결과
  - `Attention4NovelRec` 모델이 최고 성능을 기록
  - 유효한 소비가 예상되는 새로운 소설을 추천함에 따라 실효성 및 목적 적합성 충족
  - 업데이트 필요 주기가 길며 단순한 연산만으로 구현된 모델로, 유지보수 및 예측 비용이 상대적으로 저렴한 효율적인 모델
  - 다양한 지표로 비즈니스 실효성 평가가 가능할 것 
    - e.g. 새로운 소설에 대한 유효소비 주기 변화, 추천된 새로운 소설과 이용자가 탐색한 새로운 소설 간의 유효소비 비율
> 전체 내용 확인 [[프로젝트1 레파지토리](https://github.com/namdo18/Attention4NovelRec)]    

</br>

### 프로젝트2. [웹소설 추천을 위한 반복 소비 행동 모델링](https://github.com/namdo18/NovelNet)

2022.09 텐센트 소속 저자들이 발표한 세션 기반 웹소설 추천시스템 논문[[1](#reference)] 구현 프로젝트
- 논문 선정 이유 : 최신 연구 경향 및 실용적 기술 습득 기대
- 논문에서 제안하는 모델 `NovelNet`
  - `PointerNetwork` 와 `BahdanauAttention` 의 HybirdNetwork 구조
  - `PointerNetwork` 의 단점을 보완한 `pointwiseLoss` 도입
  - 과거에 소비한 소설과 새로운 소설들에 대한 확률을 동시에 고려하여 다음 읽을 소설을 추천
- 추가 실험 진행
  - 과거에 소비한 소설과 새로운 소설들에 대한 확률에 임의적 편향을 도입하여 성능 변화 관찰
  - 과거에 소비한 소설에 가중치가 높을 경우 모델 성능 증가
- 프로젝트 결과
  - 효과성은 검증되었으나, 한계점 존재
    - 반복 소비에 편향된 연구 조건으로, 모델의 성능이 다양한 콘텐츠 소비 유도라는 추천시스템 목적[[2](#reference)]과 무관
    - 유효한 소비로 볼 수 없는 소설들까지 추천 대상으로 학습함에 따라 실효성을 보장하지 못함
> 전체 내용 확인 [[프로젝트2 레파지토리](https://github.com/namdo18/NovelNet)]    

</br>

## Reference
###### [1]Yuncong Li, et al, 2022, Modeling User Repeat Consumption Behavior for Online Novel Recommendation, RecSys’22 September 18–23
###### [2]김대원, 2019, 추천 알고리즘의 개념과 적용 그리고 발전의 양상, Broadcasting Trend & Insight October 2019 Vol.20, 한국콘텐츠 진흥원
###### [3]Balazs Hidasi, et al, 2016, Session-Based Recommendations with Recurrent Neural Networks, ICLR 2016
###### [4]A.Vaswani, et al, 2017, Attention is all you need, NIPS 2017



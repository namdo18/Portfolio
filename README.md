</br>

# 프로젝트 포트폴리오
</br>

### 프로젝트1. [다양한 콘텐츠 소비 유도를 위한 웹소설 추천시스템 개발](#다양한-콘텐츠-소비-유도를-위한-웹소설-추천시스템-개발)

Li et al.2022[[1](#reference)] 에서 제안된 세션 기반 웹소설 추천시스템 `NovelNet` 의 한계점 극복
- `NovelNet` 한계점
  - 반복 소비에 편향된 연구 조건에서 모델 학습, 다양한 콘텐츠 소비 유도라는 추천시스템 목적[[2](#reference)]에 부합하지 못함
  - 유효한 소비로 볼 수 없는 소설들까지 추천 대상으로 학습함에 따라, 비즈니스 실효성을 보장하지 못함
- 개발 방향 : 유효한 소비가 예상되는 새로운 소설을 추천하는 모델
- 모델 개발 및 구현
  - 학습할 유효소비 데이터 선별 및 과거에 소비한 소설들에 대한 마스킹 알고리즘 구현(모델 내부) 
  - `NovelNet`, `GRU4Rec`[[3](#reference)], `TransformerEncoder`[[4](#reference)] 기반의 모델 개발 및 구현
  - `Attention4NovelRec` : 사전학습된 소설 장르정보를 이용하는 어텐션 메커니즘 기반의 별도의 모델 개발
- 프로젝트 결과
  - `Attention4NovelRec` 모델이 최고 성능을 기록
  - 유효한 소비가 예상되는 새로운 소설을 추천함에 따라 실효성 및 목적부합성 충족
  - 업데이트 필요 주기가 길며 단순한 연산만으로 구현된 모델로, 유지보수 및 예측 비용이 상대적으로 저렴한 효율적인 모델
  - 다양한 지표로 비즈니스 실효성 평가가 가능할 것 
    - e.g. 새로운 소설들의 유효소비 주기 변화, 추천된 새로운 소설과 이용자가 탐색한 새로운 소설 간의 유효소비 비율

###### [프로젝트1 레파지토리 이동]()

</br>

### 프로젝트2. [웹소설 추천을 위한 반복 소비 행동 모델링](#웹소설-추천을-위한-반복-소비-행동-모델링)
###### 프로젝트 레파지토리 이동 (클릭 $\uparrow$)
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
    - 반복 소비에 편향된 연구 조건으로, 모델의 성능이 다양한 콘텐츠 소비 유도라는 추천시스템 목적과 무관
    - 유효한 소비로 볼 수 없는 소설들까지 추천 대상으로 학습함에 따라 실효성을 보장하지 못함

###### [프로젝트2 레파지토리 이동]()  

</br>

# [다양한 콘텐츠 소비 유도를 위한 웹소설 추천시스템 개발](#naver.com)
`python==3.8.15`, `pytorch==1.12,1+cu113`, `프로젝트v.1.0`


### 주요 사용 알고리즘
- `NovelNet` : Li et al.2022[[1](#reference)], `biGRU`, `BahdanauAttention`  
- `GRU4Rec` : Hidashi et al.2016[[3](#reference)] `GRU`  
- `TransformerEncoder` : Vaswani et al.2017[[4](#reference)] `MultiHeadAttention`,  `PositionalEncoding`  
- 프로젝트 개발 알고리즘 
    - `SelfEncoding` : `BahdanauAttention` 에서 아이디어 착안, 하나의 대표 시퀀스를 얻을 수 있도록(정보압축) 간단한 알고리즘 구현
    - `SubGenreEncoding` : `PositionalEncoding` 에서 아이디어 착안, 모델 규모의 변화 없이 추가정보 전달을 위해 장르정보 임베딩 사전학습 후, 시퀀스 데이터에 더하여 정보전달
    - `Attention4NovelRec` : 새로운 어텐션 기반 모델 아키텍쳐 개발


### 주요 수행 내역  
Li et al.2022[[1](#reference)] 에서 제안된 세션 기반 웹소설 추천시스템 `NovelNet` 의 한계점 극복을 위한 연구  

`NovelNet` 한계점  
    - 탐색 목적의(e.g. 읽은 시간이 1분 미만) 인터렉션 까지 추천 대상으로 학습. 이에 따라 비즈니스 실효성을 보장할 수 없음.  
    - 이전 소비 내역 추천에 초점을 맞추며, 이용자들에게 다양한 콘텐츠 소비를 유도하고자 하는 추천시스템의 목적[[2](#reference)]을 제대로 반영하지 못하고 있음.   
    - 이용자들의 탐색시간을 경감시켜줄 수 있으며, 실효성과 목적부합성을 충족시킬 수 있는 모델 개발 필요  
    
- 유효소비가 예상되는 콘텐츠들을 추천할 수 있도록, 모델이 학습하게 될 타겟데이터 선별
- 다양한 콘텐츠들의 소비를 유도할 수 있도록, 과거 소비 이력이 있는 소설들에 대한 마스킹 알고리즘 구현(모델 내부)
- `NewNovelRec` : `NovelNet` 에서 제안된 새로운 소설 추천을 위한 알고리즘을 별도의 모델로 구현
- `Attention4Rec` : `Transformer` 에서 제안된 인코더 구조를 기반으로 하는 모델 개발(FF를 통한 다중분류 학습)
- `GRU4Rec` : 성능 비교를 위해, 세션 기반 추천 task 에서 좋은 성능을 보이는 `GRU4Rec` 기반 모델 개발
- `SubGenreEncoding` : 모델 규모 변화 없이 성능 향상을 위해 장르정보(임베딩) 사전학습 진행 후 시퀀스에 전달. 하위장르 임베딩에 대해 상위장르 정보를 지닐 수 있도록 설계
- `Attention4NovelRec` : 사전학습된 장르정보를 이용하는 어텐션 메커니즘 기반의 별도의 모델 개발


### Trouble Shooting
- 과적합 발생 이슈
    - 문제 : 기본 실험 모델 `NewNovelRec`, `Attention4Rec`, `GRU4Rec` 구현 및 옵션 테스트 시, 학습 성능과 검증 성능의 차이가 크게 벌어지는 문제 발생
    - 연구 : 유효소비 데이터 선별 과정에서 데이터 규모는 크게 줄었으나, 기존 논문들에서 제안된 모델 규모가 유지되었던 것이 문제로 추정, 정보 손실을 야기시켜 해결 가능할 것으로 추정
    - 해결 :
        1.  모델규모 및 드랍아웃 비율 조절
        2. `GRU4Rec` 모델에 대해 마지막 히든값을 시퀀스인코딩 결과값으로 이용(i.e. seq2vec)
        3. `SelfEncoding` : 하나의 대표 시퀀스로 정보를 압축하기 위해 `BahdanauAttention` 에서 아이디어 착안, 간단한 알고리즘 구현. `Attention4Rec` 모델에 적용
            - $f_{\text{enc}} \triangleq (H \cdot W)^T \cdot H,\quad H \in \mathbb{R}^{\text{|session|}\times |h|}, \ W \in \mathbb{R}^{|h|}$
    - 결과 : 검증 성능의 소폭 향상 및 학습 성능의 큰 폭의 하락. 다만, 성능 절대치가 낮다는 문제

- 낮은 성능 이슈
    - 문제 : 최고 검증 성능 `GRU4Rec` 0.11(recall@20) 으로 성능 향상 필요
    - 연구 : 
        1. `NewNovelNet` 모델의 경우, 반복 소비 추천을 위해 추출된 특징값[[1](#reference)] 이용이 문제로 추정(특징값 임베딩 후 병합하여 이용, 프로젝트 task 에 부합하지 않는 정보 전달이 원인인 것으로 추정)
        2. 다른 모델의 경우 정보 손실 야기에 따른 구조적 한계가 있을 것으로 추정.
    - 해결 :
        1. `NovelNet` 모델에 대해, FeatureSelection 진행(Subset 테스트)
        2. `GRU4Rec` 모델에 대해 히든값 차원을 상향조정(seq2vec 을 이용하는 만큼, 일반화 된 정보를 더 모을 수 있을 것으로 판단)
        3. `SubGenreEncoding` : 모델의 규모 변화 없이, 추가 정보 전달을 위해 장르정보(임베딩) 사전학습 후 시퀀스에 전달. 하위장르 임베딩에 대해 고정된 FF 를 통해 상위장르 다중분류로 학습. 유사한 장르 정보를 얻을 수 있도록 유도. `Attention4Rec` 모델에 적용
        4. `Attention4NovelRec` : 드랍아웃 영향을 줄이는 대신, 모델 규모를 축소하기 위한 방향으로 새로운 모델 아키텍쳐 개발. 
            - 핵심 알고리즘 :
                - 소설 시퀀스 임베딩 + `SubGenreEncoding`
                - `PositionalEncoding`
                - `BahdanauAttention`(단, $S_{t-1} = H$)
                - 새로운 소설 추천을 위한 마스킹
                - 피드포워드를 통한 소설별 점수 도출
    - 결과 : 큰 폭의 성능향상 기록.


### 프로젝트 결과
- Li et al.2022[[1](#reference)] 에서 사용된 텐센트 QQ브라우저의 웹소설 이용 데이터 이용
    - 2021.11.11 ~ 2021.11.22 기간 동안의 랜덤샘플링 된 863,000 여개의 인터렉션 데이터
- 평가기준 : Hidashi et al.2016[[3](#reference)] 에서 주 평가지표로 사용된 recall@20 사용

|model|train|valid|test|
|----|----|----|----|
|Attention4Rec|0.17549|0.12877|0.12608|
|NewNovelRec|0.15403|0.13999|0.14156|
|GRU4Rec|0.49632|0.19592|0.19198|
|Attention4NovelRec|0.76474|0.20258|0.20333|

- 평가결과
    - `Attention4NovelRec` 모델이 최고 성능을 기록
    - `GRU4Rec`, `Attention4NovelRec` 모델에 대해 과적합 현상 재발생하였으나, 유의미한 검증 성능의 향상도 있었음
    - `Attention4NovelRec` 모델의 경우, 업데이트 필요 주기가 매우 긴 장르정보를 이용함에 따라 업데이트 비용이 낮은 모델. 또한, 최소한의 행렬연산만으로 실행됨에 따라 계산비용이 낮은 모델. 
    - 셀프어텐션 기법으로 콘텐츠 추천에서 중요한 short-term intention 을 반영가능(e.g. 소비자의 감정)
    - 유효소비가 예상되는 새로운 소설을 추천하며, 비즈니스 실효성 및 목적부합성 충족
    - 서로 다른 콘텐츠 유효소비 간의 시간차, 추천된 새로운 콘텐츠와 소비자가 탐색한 새로운 콘텐츠의 유효소비 비율 등의 지표로 비즈니스 실효성을 평가할 수 있을 것

- 향후 과제
    - [ ] cross-validation 기법을 이용하여 재학습 후 성능비교 
    - [ ] Hidasi et al.2018[[5](#reference)] 에서 제안된 샘플링 기법 및 다양한 손실함수를 이용하여 성능비교 평가  
</br>



</br>

# 웹소설 추천을 위한 반복 소비 행동 모델링
Li et al.2022[[1](#reference)] 논문 구현 프로젝트  
`python==3.10.6`, `pytorch==1.11.0`

### 주요 사용 알고리즘
- `biGRU`, `BahdanauAttention`, `PointerNetwork`, `pointwiseLoss`


### 논문 리뷰 요약
- 논문 선정 이유
    - 2022년 9월 나온 논문으로 최신 연구경향을 담고 있을 것으로 기대
    - 주요 저자들이 텐센트 소속으로 비즈니스 목적의 실용적인 연구가 진행되었을 것으로 기대
- 핵심 내용 
    - 이용자들의 서비스 이용 패턴에서 특징값을 추출. 모델의 입력으로 이용하여 도메인 특화 추천시스템 구현 가능(인코딩 후 병합하여 시퀀스화)
    - 이용자들 대부분이 반복 소비 패턴을 보임에 따라, 이전 소비 내역을 추천하는 것이 중요 
    - 신규 이용자들은 선호 작품을 찾는 것이 친숙하지 않기 때문에, 그들을 위한 추천 시스템 필요(세션 기반 추천과 신규 이용자를 위한 추천은 동일)
    - 이용 패턴 및 반복 소비를 고려해 신규이용자들에게 웹소설을 추천해주는 세션 기반 추천시스템 `NovelNet` 제안
    - 앱 환경에서 모달을 띄워 한 개의 작품을 추천하는 경우를 가정, recall@1 을 평가지표로 `NovelNet` 0.4702 성능 달성
        - `GRU4Rec`[[3](#reference)] 0.4220, `RecentNovel` 0.4448 대비 우수한 성능 기록
        - `RecentNovel` 은 직전 소비한 소설을 추천하는 규칙 기반 알고리즘
    

### 핵심 알고리즘
- `biGRU` 를 통한 세션 정보 시퀀스 인코딩
    - 특징값 별로 개별 임베딩 레이어를 통해 벡터화
    - 세션 단위로 임베딩 값 병합 후 `biGRU` 레이어를 통해 시퀀스 인코딩
        - 인코딩 시퀀스 $S = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_L], \quad \mathbf{h}_l = [\mathbf{\overset{\rightarrow}{h}}_l; \mathbf{\overset{\leftarrow}{h}}_l]$ 
        - 각 방향별 마지막 히든값 병합, $H = [\mathbf{\overset{\rightarrow}{h}}_L ; \mathbf{\overset{\leftarrow}{h}}_1]$

- `BahdanauAttention` 를 통한 새로운 소설에 대한 추천 확률 도출
    - 시퀀스 정보 압축, $\text{score}(S, H) = \text{tanh}(S \cdot W_s + H \cdot W_h) \cdot V,\quad W \in \mathbb{R}^{|\mathbf{h}| \times |\mathbf{h}|}, V \in \mathbb{R}^{|\mathbf{h}|}$
    - 대표 시퀀스 $r^S \in \mathbb{R}^{|batch| \times |\mathbf{h}|}$ 인코딩, $\text{value}(\text{score}, S) = \text{softmax}(\text{score}^T)S$
    - 소설임베딩 값과 유사도 계산 및 마스킹을 통한 새로운 소설에 대한 추천 확률 도출, $p^{new} = \text{softmax}(r^SW^T_{\text{emb}} \circledcirc m^c),\quad c=\text{consumed}$


- `PointerNetwork` 를 통한 이전 소비 내역에 대한 추천 확률 도출
    - 시퀀스 정보 압축, $\text{score}(S, H) = \text{tanh}(S \cdot W^c_s + H \cdot W^c_h) \cdot V^c,\quad W^c \in \mathbb{R}^{|\mathbf{h}| \times |\mathbf{h}|}, V^c \in \mathbb{R}^{|\mathbf{h}|}$
    - 빈도 수를 고려한 소설별 확률값 도출, $p^{listwise} = \text{softmax}(\text{score}) \cdot E_{\text{novel}},\quad E = \text{1 of N encoding}$
    - 인터렉션별 확률 도출, $p^{pointwise} = \text{sigmoid}(\text{score})$
        - $p^{listwise}$ 에 대해 세션 길이가 1인 경우, 실제 소비 의도와 상관 없이 하나의 소설에 대한 확률이 1로 설정되는 것을 방지하기 위해 사용
        - 실제 추천 시에는 $p^{pointwise} \cdot E_{\text{novel}}$ 이용( $p^{new}$ 와 비교하여 가장 높은 확률 소설 추천)

- `pointwiseLoss` 및 손실함수 설정 $\mathcal{L} = \mathcal{L} + \mathcal{L}_{pointwise}$
    - $\mathcal{L} = \text{NLL-loss}(p), \quad p = p^{listwise} \ \text{if} \ y \in N^c \ \text{else} \ p^{new}$
    - $\mathcal{L}_{pointwise} = \lambda \cdot \text{BCE}(m \circledcirc p^{pointwise})$
        - $\lambda$ 를 통해 `pointwiseLoss` 에 가중치를 부여하여 학습
        - $m$ 은 마지막 인터렉션(세션 정보 누적)에 대해서만 연산할 수 있도록 하는 마스크 
        

### 프로젝트 수행결과

- 텐센트 QQ브라우저의 웹소설 이용 데이터 이용
    - 2021.11.11~22 기간에서 랜덤샘플링 된 세션 중 1000 개의 세션에 대한 인터렉션 데이터들을 학습 데이터셋으로 이용
    - 2021.11.23~24 기간에서 랜덤샘플링 된 세션 중 300 개의 세션에 대한 인터렉션 데이터들을 테스트 데이터셋으로 이용
    
- 알고리즘 구현 및 효과성 검증 실험 결과
    - 총 2,503 명의 신규 이용자들 각각에 대해, 2,760 개의 소설 중 다음으로 읽을 소설을 정확히(recall@1) 예측할 확률
    - 0.5825 의 성능 기록(경우의 수가 줄어들면서 성능향상 추정)
    
- 추가 실험 진행
    - $\text{CrossEntropy}(\text{softmax}((1-\alpha) \cdot p^{new} + \alpha \cdot p^{listwise}))$ 를 통해 학습
        - 이전 소비 이력에 대한 가중치 $\alpha$ 를 고려하여 밸런싱한 확률 기반 학습 진행 
        - 이전 소비 이력에 대한 추천이 성능에 미치는 영향 조사 목적(짧은 세션 오류 감안)
        - 극단적 상황 $\alpha = 0.9$ 상황에서도 오히려 모델 성능은 0.7013 으로 +0.1188p 향상
        
- 프로젝트 결과
    - `RecentNovel` 의 성능이 0.4448[[1](#reference)] 이라는 것은 직전에 읽었던 소설을 바로 다시 읽은 데이터가 44.48% 에 달한다는 것
        - `NovelNet` 이 이러한 데이터들을 학습하여 잘 추천해도, 이용자들 입장에서 직전에 읽은 소설이 추천(모달)되는 것은 바로가기 기능에 불과할 것
    - 추가 실험 진행 결과, 사실상 이전 소비 내역만 고려했을 때의 recall@1 성능이 70% 수준
    - 위와 같은 상황에서 학습된 모델의 성능은 이용자들에게 다양한 콘텐츠 소비를 유도하고자 하는 추천시스템의 목적[[3](#reference)]에 부합하지 못할 것
    - 또한, Li et al.2022[[1](#reference)] 에서는 실제로 유효 소비라고 볼 수 없는 소설들에 대해서도 추천하도록 학습이 진행됨. 이에 따라 실효성을 기대하기 어려울 것.
    - 추천 시스템 목적[[2](#reference)] 에 부합하며, 실효성을 충족시킬 수 있는 모델 개발 필요

</br>



</br>

## Reference
###### [1]Yuncong Li, et al, 2022, Modeling User Repeat Consumption Behavior for Online Novel Recommendation, RecSys’22 September 18–23
###### [2]김대원, 2019, 추천 알고리즘의 개념과 적용 그리고 발전의 양상, Broadcasting Trend & Insight October 2019 Vol.20, 한국콘텐츠 진흥원
###### [3]Balazs Hidasi, et al, 2016, Session-Based Recommendations with Recurrent Neural Networks, ICLR 2016
###### [4]A.Vaswani, et al, 2017, Attention is all you need, NIPS 2017
###### [5]Balazs Hidasi, et al, 2018, Recurrent Neural Networks woth Top-k Gains for Session-based Recommendations, CIKM'18 October 22-26



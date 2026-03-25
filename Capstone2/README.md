# [미디어/NLP] LLM 기반 기사 헤드라인 선정성 평가 및 생성 방법론
`Python | PyTorch | HuggingFace Transformers`
- 선정적 표현의 헤드라인으로 인한 피해자에 대한 2차 가해 방지
- 기존 선정적 헤드라인 대비 생성 헤드라인의 선정성 수치 74.7% 감소
<br>





## 목차
### 1. 개요
### 2. 데이터 수집 및 전처리
### 3. 선정성 판별 및 수치화
### 4. 헤드라인 생성
### 5. 실험 결과
### 6. 기대효과 및 인사이트
### 7. 어려웠던 점과 해결방법 
<br>







## 1. 개요
**프로젝트 성과**
- ICT 플랫폼 학회 우수 논문상 수상 (2024.12)
- 기사 본문을 반영한 헤드라인을 생성하여 기존 선정적 헤드라인 대비 선정성 수치 74.7% 감소
<br>


**프로젝트의 배경 및 목표**
- 신문윤리위원회의 온라인 신문 심의 결정 분석 결과, 전체 1063건 중 57.2%가 선정성과 관련된 제재를 받음
- 젠더데스크를 운영하던 언론사에서도 선정적 표현의 헤드라인을 사용하여 피해자에 대한 2차 가해 우려 제기
<br>


**문제정의**
- “ 헤드라인의 선정성을 판별 및 수치화하고 선정성 수치가 감소된 새로운 헤드라인을 생성하는 방법론 제안 ”
<br>


**프로젝트 기간 / 인원**
- 2024.09 ~ 2024.12 (4개월) / 6명
<br>


**프로젝트 진행 과정**
<br>
<img src="https://github.com/user-attachments/assets/a6a7ca06-a86f-46be-b8e0-68dcbe8fd6f4" width="600"/>


**주요 역할**
- 뉴스 기사 데이터 크롤링 및 전처리
- 판별 모델 구현 (LMkor-BERT)
- 생성 모델 프롬프트 설계 (GPT-4 Turbo)
<br>





## 2. 데이터 수집 및 전처리
 <img width="475" height="162" alt="228453786-047157d1-0bd3-4780-9ab9-e8c9effdae19" src="https://github.com/user-attachments/assets/e0df36e3-3f1a-45e7-a2d8-0d8dc4289b78" />
<br>

### 데이터 수집

**1) AI-hub - ‘텍스트 윤리검증 데이터’**
- 판별 모델 학습 데이터 : 비선정적 및 선정적 문장의 비율이 약 18:1인 총 408,369행의 데이터셋을 구축
- 한국어 문장 열과 비윤리성 유형 정보 열을 사용

**2) 네이버 뉴스 기사 크롤링**
- 기사 데이터 : 언론사, URL, 제목, 본문으로 구성된 총 13,839행의 데이터셋 구축
- 4년 간의 5개 언론사의 기사를 크롤링 수행, ‘텍스트 윤리검증 데이터’의 선정적 문장 주요 어휘 키워드로 검색하여 수집
- [`소스 코드 (뉴스 기사 데이터 크롤링_세계일보)`](https://github.com/Gyeong-Eun/portpolio/blob/master/Capstone2/code/뉴스기사_크롤링_세계일보.ipynb)


### 크롤링 데이터 전처리
- 본문이 한 문장으로 되어 있고, 제목과 본문 간의 차이가 거의 없는 단신성 기사 삭제
- 본문과 관련 없는 내용인 기자명, 이메일, 부제목 및 요약 문장, 사진 설명, 기호, 광고 등을 제거
<br>


## 3. 선정성 판별 및 수치화 
- 트랜스포머 기반 사전학습 언어모델을 활용해 헤드라인 선정성 분류 모델을 구축
- 헤드라인이 선정적 문장으로 분류될 확률을 ‘선정성 수치’로 정의하여 정량적 평가 지표로 활용
<br>

**선정성 분류 모델 별 최종 하이퍼파라미터**

<p>
<img src="https://github.com/user-attachments/assets/171f0a14-a4a5-436e-ba51-03c265d7f147" width="500"/>
</p>
<br>

**평가지표**
- 학습에 사용한 데이터셋이 18:1로 불균형하기 때문에, 모델이 선정적인 문장을 정확하게 예측하는 것이 중요하다고 판단하여 재현율과 PR_AUC를 성능 평가 지표로 사용
<br>

**선정성 분류 모델**


<img src="https://github.com/user-attachments/assets/546b1122-d7a3-4a2b-9f51-b5d75a717b75" width="300"/>

- KcELECTRA 모델이 재현율, PR_AUC 모두 다른 모델에 비해 가장 높은 성능을 나타냄
- KcELECTRA, KoELECTRA, KLUE-RoBERTa 모델이 PR_AUC에서 비슷한 수준으로 높은 성능을 보였으나 KcELECTRA 모델이 재현율에서 다른 모델에 비해 우수한 성능을 기록
- KcELECTRA를 헤드라인 선정성 판별 및 수치화 모델로 선정
- [`소스 코드 (헤드라인 선정성 판별_KcELECTRA)`](https://github.com/Gyeong-Eun/portpolio/blob/master/Capstone2/code/KcELECTRA.ipynb)
<br>

<br>





## 4. 헤드라인 생성
- 트랜스포머 디코더 기반의 LLM 중 프롬프트 엔지니어링이 가능한 모델 사용
- 선정성 판별 및 수치화 과정을 통해 도출된 선정성 수치 상위 100개 기사 헤드라인을 비선정적으로 생성
- 프롬포트 엔지니어링을 통해 선정성 수치가 높은 헤드라인을 비선정적으로 생성
- [`소스 코드 (KULLM3&KoBART 생성 프롬프트)`](https://github.com/Gyeong-Eun/portpolio/blob/master/Capstone2/prompt/gen_kullm3&KoBART.py)
- [`소스 코드 (GPT-4 Turbo 생성 프롬프트)`](https://github.com/Gyeong-Eun/portpolio/blob/master/Capstone2/prompt/gen_gpt4turbo.py)
<br>


**생성 평가지표**

**1) 선정성 수치**
- 비선정적으로 헤드라인을 생성하였는지 확인

**2) BERTScore**
- 헤드라인이 본문 내용을 잘 반영하였는지 확인

**3) GPT-4 Turbo 모델 평가**
- 생성된 헤드라인이 헤드라인의 형식에 부합하는지 확인
- 0점에서 5점까지의 헤드라인 형식 평가 척도를 통해 실제로 사용 가능한 헤드라인의 형태로 생성되었는지 평가
- [`소스 코드 (헤드라인 형식 평가 프롬프트)`](https://github.com/Gyeong-Eun/portpolio/blob/master/Capstone2/prompt/evaluation_head.py)


<p>
<img src="https://github.com/user-attachments/assets/3a2bcd8d-b7f6-4335-bf46-e6ecadd3c4b0" width="300"/>
</p>
<br>








## 5. 실험 결과

**BERTScore & 헤드라인 형식**

<img src="https://github.com/user-attachments/assets/7dd651d1-8371-45ad-9f70-0833dec9f390" width="360"/>

- 선정성 판별 및 수치화 과정을 통해 도출된 선정성 수치 상위 100개 기사 헤드라인을 비선정적으로 생성
- 세 모델 모두 비선정적인 헤드라인을 생성했지만, GPT-4 Turbo가 본문의 내용을 효과적으로 반영하고 헤드라인 형식에 가장 적합하게 생성
- GPT-4 Turbo를 헤드라인 생성 모델로 채택
- [`소스 코드 (헤드라인 생성_GPT-4 Turbo)`](https://github.com/Gyeong-Eun/portpolio/blob/master/Capstone2/code/gpt4turbo.ipynb)
<br>




**생성된 헤드라인 예시**

<p>
<img src="https://github.com/user-attachments/assets/c0e9347b-92ad-4c07-b108-9d34719ef36a" width="600"/>
</p>
<br>




## 6. 기대효과 및 인사이트 
- 선정적으로 분류된 헤드라인 상위 100개를 대상으로 생성 모델을 비교한 결과, GPT-4 Turbo가 평균 선정성 수치를 약 74.7%   감소시키며 가장 효과적인 생성 성능을 보였다. 또한, 기사 본문의 내용을 적절히 반영한 헤드라인 생성 성능을 확인하였다.
- 이를 통해 한국어 기사 헤드라인의 선정성 검출 및 완화 생성 모델을 구축하여 언론사의 헤드라인 작성 및 검토 과정에 활용하는 것을 기대해 볼 수 있다.
<br>




## 7. 어려웠던 점과 해결방법 
- 기사 헤드라인의 선정적인 정도를 정량적으로 어떻게 수치화할 것인가에 대한 어려움이 있었다. 이를 해결하기 위해 다양한 선행논문을 참고하여 선정적인 문장으로 분류될 확률을 ‘선정성 수치’로 사용하였다.
- GPT-4 Turbo 모델의 경우 헤드라인을 비선정적으로 생성하지 못하는 문제가 있었다. KULLM3와 KoBART 모델과 같이 단순한 지시 프롬프트 대신 선정성 수치와 헤드라인 예시 등을 함께 제공하는 방식으로 프롬프트를 설계하였고 비선정적인 헤드라인 생성을 더 효과적으로 지원하여 생성 성능을 개선할 수 있었다.  


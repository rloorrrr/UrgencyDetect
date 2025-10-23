# UrgencySense - 김원

- 본 프로젝트는 멋쟁이사자처럼 AI 자연어처리(NLP)단기 심화 교육에서 진행한 긴급도 판단 모델 구축 프로젝트입니다.

# 학습 방식 

## EDA 핵심 특징 2가지

### 1. 대화 턴 전환 횟수 (Conversation Turn Count)

이것은 **신고자와 수보자 사이에 대화가 몇 번이나 오갔는지**를 측정합니다.

이것을 **핑퐁 랠리** 🏓에 비유할 수 있습니다. 긴급한 통화는 "불이야!" → "어디세요?" → "아파트요!"처럼 짧고 빠른 말이 계속 오가는 빠른 랠리와 같습니다. 반면, 덜 긴급한 통화는 길게 설명하고 답하는 느린 랠리와 비슷합니다. 저희 분석에 따르면, 긴급도가 높을수록 이 턴 전환 횟수가 증가하는 경향이 있어, 통화가 얼마나 역동적인지를 판단하는 좋은 지표가 됩니다.

---

### 2. 수보자 초기 응답 시간 (Agent's Initial Response Time)

이것은 신고자가 문제 상황에 대한 첫 설명을 마친 후, 수보자가 첫 번째 의미 있는 반응(질문, 지시 등)을 하기까지의 **'침묵' 시간**을 측정합니다. 즉, 상황을 파악하는 데 걸리는 '생각하는 시간'입니다.

이것을 **응급실 의사의 반응 시간** 🩺에 비유할 수 있습니다. "숨을 못 쉬겠다"는 위급 환자에게는 의사가 즉시 반응합니다. 반면, 덜 위급한 환자에게는 잠시 생각한 후 다음 질문을 하죠. 저희 분석 결과, '중', '상' 등급의 통화에서 이 반응 시간이 '하' 등급보다 눈에 띄게 짧았습니다. 따라서 이 지표는 상황이 얼마나 심각한지를 예측하는 매우 강력한 단서가 됩니다.

## 결과(Colab .ipynb 링크)
- [EDA](https://colab.research.google.com/drive/1YKSupCTp0c6rzeMOqQxhX2DyhmmfPysA?usp=sharing)
- [데이터 전처리](https://drive.google.com/file/d/19vZ8cUREOpNvoCK_GnGDUbpgB9H0NvZQ/view?usp=sharing)
- [모델 학습](https://colab.research.google.com/drive/1XZRGRb7w8_OaWlzltY8NaJEnnXmXwK_f?usp=sharing)
- [인퍼런스](https://drive.google.com/file/d/1b4PzgilX-f_u2Dj9zLJ2lqvbg_Yr5kFM/view?usp=sharing)

# 욕설 감지기
인공지능을 통해 욕설을 감지하는 프로젝트입니다.

이 repository에는 모델을 예측할 수 있는 Python 코드가 포함되어 있습니다.

학습된 모델 파일은 [여기](https://drive.google.com/file/d/1gO_5Pltn9vEVVyOW3gTTR4e7_DdjKPrL/view?usp=sharing)에서 다운로드 받을 수 있습니다. (모델 예측을 위해선 학습 모델을 src/models 폴더에 옮겨 주셔야 합니다)


## 모델
인공지능 모델에는 Bidirectional-GRU를 활용하였으며, 라이브러리는 keras를 사용했습니다.

자세한 정보는 [코드](https://github.com/2runo/Curse-detection/blob/master/src/model.py)를 참조하세요.


## 데이터
학습 데이터로는 [욕설 감지 데이터셋](https://github.com/2runo/Curse-detection-data)을 사용했습니다.

욕설 감지 데이터셋은 약 6000개의 문장에 대해 욕설 여부를 분류한 데이터셋입니다.


## 예측 과정
욕설을 감지하는 과정은 다음과 같습니다.
#### 1. 전처리
- 특수문자, 영어 제거
- 연속적인 글자 단축 (ㅋㅋㅋㅋ->ㅋ)
- 초성, 중성, 종성으로 분리 (안녕 -> ㅇㅏㄴㄴㅕㅇ)
#### 2. 임베딩
- AutoEncoder를 활용하여 비슷한 생김새의 글자는 비슷한 벡터 값을 갖도록 임베딩
![autoencoder](/imgs/autoencoder.jpg)
#### 3. 예측
- 모델로 예측하면, 욕설 여부를 확률로 반환


## 사용법
#### 예측을 하기 위해선 학습된 모델(weights.h5 파일)이 src/models 폴더에 옮겨져 있어야 합니다!

#### 학습된 모델은 [여기](https://drive.google.com/file/d/1gO_5Pltn9vEVVyOW3gTTR4e7_DdjKPrL/view?usp=sharing)에서 다운로드 받을 수 있습니다.

예측에는 src 폴더의 curse_detector.py 파일을 사용합니다.
```python3
from curse_detector import CurseDetector

curse = CurseDetector()

# [욕설이 아닐 확률, 욕설일 확률]
print(curse.predict('씨발'))  # [0.0023 0.9976] -> 99.76% 확률로 욕설
```
다음은 실제 커뮤니티 사이트의 댓글들을 분류한 예시입니다:
```python3
print(curse.predict('ㅆ발'))    # [0.0159 0.9840]
print(curse.predict('^^ㅣ발'))  # [0.0570 0.9429]

print(curse.predict(['이게 뭔,,, 개소리야... 아오...',
                     '전자발찌도 차야 함..',
                     '부정적평가 하는 씨발것들은 머죠??',
                     '보는내내  너무 화가났었어요.  소시오패스같아요.',
                     '아침부터 시야흐려지네.ㅠㅠ']))
# [[0.0043 0.9956]
#  [0.7492 0.2507]
#  [0.0102 0.9897]
#  [0.9164 0.0835]
#  [0.7681 0.2318]]
```

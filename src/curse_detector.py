from model import *
import text_preprocessing as pre
import embedding as emb


class CurseDetector():
    def __init__(self, path='models/weights.h5'):
        # 모델 가져오기
        self.model = build_model()
        self.load_weights(path)

    def load_weights(self, path):
        # 학습된 weights 가져오기
        self.model.load_weights(path)

    def embedding(self, texts):
        # 전처리, 임베딩 수행
        texts = pre.preprocess(texts)
        embed = emb.embedding(texts)
        embed = emb.padding(embed)
        return embed

    def predict(self, texts):
        # 욕설 분류 수행
        one = False
        if isinstance(texts, str):
            texts = [texts]
            one = True

        # 예측
        embed = self.embedding(texts)
        pred = self.model.predict(embed)

        if one:
            pred = pred[0]
        return pred


curse = CurseDetector()

print(curse.predict('씨발'))    # [0.0023 0.9976]
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

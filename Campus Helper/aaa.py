import json
import numpy as np # 수학적 연산, 통계, 선형대수학 등 다양한 과학 계산에 사용됩니다.
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # 신경망 구성
from tensorflow.keras.layers import Dense, Dropout # 신경망 레이어
from tensorflow.keras.optimizers import Adam # 학습 속도와 안정성
from tensorflow.keras.callbacks import EarlyStopping
from konlpy.tag import Okt # 형태소
import random # 랜덤
import pickle # 파일 저장

# 데이터 로드
with open('C:/Users/Admin/Desktop/Campus Helper/json/intents.json', encoding='utf-8') as file:
    data = json.load(file)

# 한국어 형태소 분석기 초기화
okt = Okt()

words = []
labels = []
docs_x = []
docs_y = []

# 데이터 전처리
for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = okt.morphs(pattern)
        words.extend(tokens)
        docs_x.append(tokens)
        docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

# 단어 중복 제거 및 정렬
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

out_empty = [0] * len(labels)

for x, doc in enumerate(docs_x):
    bag = []
    
    for w in words:
        bag.append(1) if w in doc else bag.append(0)
        
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# 모델 정의 및 컴파일
model = Sequential()
'''
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))
'''
model.add(Dense(256, input_shape=(len(training[0]),), activation='relu'))  # 뉴런 수 증가
model.add(Dropout(0.4))  # Dropout 비율을 줄임
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping 콜백 추가
early_stopping = EarlyStopping(monitor='loss', patience=100, verbose=1)

# 모델 훈련
history = model.fit(training, output, epochs=1500, batch_size=8, verbose=1)

# 모델 저장
model.save("C:/Users/Admin/Desktop/Campus Helper/model_tf2.keras")

# words와 labels 저장
with open("C:/Users/Admin/Desktop/Campus Helper/words.pkl", "wb") as f:
    pickle.dump(words, f)

with open("C:/Users/Admin/Desktop/Campus Helper/labels.pkl", "wb") as f:
    pickle.dump(labels, f)

# 로스 값을 그래프로 시각화
plt.plot(history.history['loss'], label='Loss', color='pink')
plt.title('Model Loss Over Time')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
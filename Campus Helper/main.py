import numpy as np
from tensorflow.keras.models import load_model
from konlpy.tag import Okt
import random
import json
from difflib import SequenceMatcher
import sys
import os
import pickle
import tensorflow as tf  # TensorFlow를 사용하기 위한 추가

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# 한국어 형태소 분석기 초기화
okt = Okt()

# words 리스트와 입력된 문장을 비교해 one-hot encoding 형식으로 변환
def bag_of_words(sentence, words):
    tokens = okt.morphs(sentence)
    bag = [0] * len(words)
    for w in tokens:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array([bag])

# 태그의 유사도를 계산하는 함수
def get_similar_tag(tag, threshold=0.6):
    similar_tags = []
    for intent in data['intents']:
        similarity = SequenceMatcher(None, tag, intent['tag']).ratio()
        if similarity >= threshold:
            similar_tags.append(intent['tag'])
    return similar_tags

# 모델 예측 함수 - @tf.function을 사용해 최적화
@tf.function
def predict(input_data):
    return model(input_data)

def chat(value_received):
    global model, words, labels, data  # 전역 변수 선언
    
    # 모델 로드 (경로 수정 필요)
    model = load_model('C:/Users/Admin/Desktop/Campus Helper/model_tf2.keras')

    # words와 labels 로드 (경로 수정 필요)
    with open("C:/Users/Admin/Desktop/Campus Helper/words.pkl", "rb") as f:
        words = pickle.load(f)

    with open("C:/Users/Admin/Desktop/Campus Helper/labels.pkl", "rb") as f:
        labels = pickle.load(f)

    # 데이터 로드 (경로 수정 필요)
    with open('C:/Users/Admin/Desktop/Campus Helper/json/intents.json', encoding='utf-8') as file:
        data = json.load(file)

    inp = value_received

    # 입력 데이터를 모델에 맞게 변환
    input_data = bag_of_words(inp, words).reshape(1, -1)

    # tf.function으로 감싸진 predict 함수 호출
    results = predict(input_data)
    results_index = np.argmax(results)
    tag = labels[results_index]

    # 예측 확률을 가져옴
    accuracy = results[0][results_index] * 100

    # 임계값 설정 (예: 88%)
    threshold = 88

    # 확률이 임계값보다 낮을 경우 기본 응답
    if accuracy < threshold:
        return f"챗봇: 죄송해요, 이해하지 못했어요. (정확도: {accuracy:.2f}%)"

    similar_tags = get_similar_tag(tag)

    response_found = False
    for tg in similar_tags:
        for intent in data["intents"]:
            if intent['tag'] == tg:
                responses = intent['responses']
                response_found = True
                response = random.choice(responses)
                print(f"정확도: {accuracy:.2f}%")
                return f"챗봇: {response}"

    if not response_found:
        return "챗봇: 죄송해요, 이해하지 못했어요."

# 메인 루프
while True: 
    print("입력 :")
    value_received = sys.stdin.readline().rstrip('\n')

    print("수정본:", value_received)

    # 'chat' 함수를 정의한 후 호ㅋㅋ출
    response = chat(value_received)
    print(response)

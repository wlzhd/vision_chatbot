from flask import Flask, request, jsonify # pip install flask
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from konlpy.tag import Okt
import random
import json
from difflib import SequenceMatcher
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)  # 모든 출처에서의 요청을 허용합니다.

# value_received = sys.stdin.readline().rstrip('\n')
# print("확인 : " , value_received)


# 모델 로드
model = load_model('model_tf2.keras')

# 데이터 로드
with open('intents.json', encoding='utf-8') as file:    
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
        # 토큰화 및 형태소 분석
        tokens = okt.morphs(pattern)
        words.extend(tokens)
        docs_x.append(tokens)
        docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

# 단어 중복 제거 및 정렬
words = sorted(list(set(words)))
labels = sorted(labels)

def bag_of_words(sentence, words):
    # 문장의 형태소 분석
    tokens = okt.morphs(sentence)
    # 문장의 단어 빈도 초기화
    bag = [0]*len(words)  
    # 단어 빈도 카운트
    for w in tokens:
        for i, word in enumerate(words):
            if word == w: 
                bag[i] = 1
    return np.array([bag])

def get_similar_tag(tag, threshold=0.6):
    similar_tags = []
    for intent in data['intents']:
        similarity = SequenceMatcher(None, tag, intent['tag']).ratio()
        if similarity >= threshold:
            similar_tags.append(intent['tag'])
    return similar_tags

def chat(value_received):
    inp = value_received

    results = model.predict([bag_of_words(inp, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]

    similar_tags = get_similar_tag(tag)

    response_found = False
    for tg in similar_tags:
        for intent in data["intents"]:
            if intent['tag'] == tg:
                responses = intent['responses']
                response_found = True
                return random.choice(responses)

    if not response_found:
        return "챗봇: 죄송해요, 이해하지 못했어요."


@app.route('/aaa', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        value_received = request.form['value_received']
        response = chat(value_received)
        print(response)
        return response  # 한글을 그대로 반환
    return 'Hello, please send a POST request with "value_received" parameter.'

if __name__ == '__main__':
    app.run('0.0.0.0', port = 5000, debug=True)

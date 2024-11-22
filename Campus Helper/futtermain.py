from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from konlpy.tag import Okt
import random
import json
from difflib import SequenceMatcher
import pickle
import os
# from pykospacing import Spacing

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)  # 모든 출처의 요청 허용

# 한글 띄어쓰기 교정을 위한 띄어쓰기 초기화
# spacing = Spacing()

# Load model
model = load_model('C:/Users/Admin/Desktop/Campus Helper/model_tf2.keras')

# Load data
with open('C:/Users/Admin/Desktop/Campus Helper/json/intents.json', encoding='utf-8') as file:
    data = json.load(file)

# Load words and labels
with open("C:/Users/Admin/Desktop/Campus Helper/words.pkl", "rb") as f:
    words = pickle.load(f)

with open("C:/Users/Admin/Desktop/Campus Helper/labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Initialize Korean morphological analyzer
okt = Okt()

def bag_of_words(sentence, words):
    tokens = okt.morphs(sentence)
    bag = [0] * len(words)
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

    # Apply spacing correction
    corrected_input =  inp #spacing(inp)

    input_data = bag_of_words(corrected_input, words).reshape(1, -1)
    
    # Predict the response using the loaded model
    results = model.predict(input_data)
    results_index = np.argmax(results)
    tag = labels[results_index]

    # Get prediction accuracy
    accuracy = results[0][results_index] * 100

    # Set threshold (e.g., 88%)
    threshold = 88

    # Return default response if accuracy is below the threshold
    if accuracy < threshold:
        return f"챗봇: 죄송해요, 이해하지 못했어요."

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

@app.route('/', methods=['POST'])
def index():
    if request.is_json:
        req_data = request.get_json()
        value_received = req_data.get('text', '')
        print("플러터: ", value_received)
        response = chat(value_received)
        print(response)
        return jsonify({'message': response})  # Return response as JSON
    return jsonify({'error': 'Invalid input format. Please send JSON data.'}), 400

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)

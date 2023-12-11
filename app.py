import json

import joblib
from flask import Flask, request, jsonify
from keras.src.saving.saving_api import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

from model.data_preparation import clean_string, prepare_dataset
from model.models import train_NN, eval_model

app = Flask(__name__)


def train_model():
    with open('data/comentarios_con_grupo.json', 'r') as json_file:
        dataset = json.load(json_file)
    X_train, X_val, X_test, y_train, y_val, y_test, vect = prepare_dataset(dataset)
    modelNN = train_NN(X_train, y_train)
    result, accSVC = eval_model(modelNN, X_val, y_val)
    joblib.dump(modelNN, 'cc_nlp.joblib')
    return result, accSVC, vect, modelNN


def predict(doc, vect):
    model = joblib.load('cc_nlp.joblib')
    doc_cleaned = clean_string(doc)
    corpus = []
    corpus.append(doc_cleaned)
    test_vect = vect.transform(corpus)
    return model.predict(test_vect)[0]


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/train')
def trainf_model():
    result, accSVC, vect, modelNN = train_model()
    return jsonify({'result': result, 'acc': accSVC})


@app.route('/predict', methods=['POST'])
def predict_comment():
    with open('data/comentarios_con_grupo.json', 'r') as json_file:
        dataset = json.load(json_file)
    X_train, X_val, X_test, y_train, y_val, y_test, vect = prepare_dataset(dataset)
    data = request.get_json()
    comment = data.get('comment')
    prediction = predict(comment, vect)
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run()

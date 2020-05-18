# classifier/application/routes.py
from flask import Flask, request, jsonify
from application import app
from application.spam_classifier import Spam


@app.route('/classify_text', methods=['POST'])
def classify_text():
    data = request.json
    text = data.get('text')
    if text is None:
        params = ', '.join(data.keys())
        return jsonify({'message': f'Parametr "{params}" is invalid'}), 400

    result = Spam().classify(text)
    return jsonify({'result': result})

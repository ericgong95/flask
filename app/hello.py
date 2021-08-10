from flask import Flask
from flask import current_app
from flask import jsonify
from flask import request
from sklearn.metrics.pairwise import cosine_similarity
from marshmallow import Schema, fields, ValidationError
import numpy as np
import tensorflow_hub as hub

# Load Model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

app = Flask(__name__)

# Schemas for each of the endpoints
class EmbeddingSchema(Schema):
    sentence = fields.Str(required=True)

class BulkSchema(Schema):
    sentences = fields.List(fields.String,required=True)

class SimilaritySchema(Schema):
    sentence_1 = fields.String(required=True)
    sentence_2 = fields.String(required=True)

class InvalidAPIUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        super().__init__()
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

@app.errorhandler(InvalidAPIUsage)
def invalid_api_usage(e):
    return jsonify(e.to_dict())

@app.route('/embeddings', methods=['GET'])
def embeddings():
    data = request.args
    schema = EmbeddingSchema()
    try:
        result = schema.validate(data)
    except ValidationError as e:
        raise InvalidAPIUsage(e.messages, status_code=400)
    sentence = data.get('sentence')
    if sentence is None:
        raise InvalidAPIUsage('Did not provide sentence parameter', status_code=400)
    embeddings = embed([sentence])
    embeddings = embeddings.numpy().tolist()[0]
    return jsonify(embedding=embeddings), 200


@app.route('/embeddings/bulk', methods=['POST'])
def bulk():
    data = request.get_json()
    schema = BulkSchema()
    try:
        result = schema.load(data)
    except ValidationError as e:
        raise InvalidAPIUsage(e.messages, status_code=400)
    if data['sentences'] is None:
        raise InvalidAPIUsage('Did not provide sentences parameter', status_code=400)
    sentences = []
    for sentence in data['sentences']:
        sentences.append(sentence)
    embeddings = embed(sentences)
    embeddings = embeddings.numpy().tolist()
    return jsonify(embedding=embeddings), 200

@app.route('/embeddings/similarity', methods=['POST'])
def similarity():
    data = request.get_json()

    schema = SimilaritySchema()
    try:
        result = schema.load(data)
    except ValidationError as e:
        raise InvalidAPIUsage(e.messages, status_code=400)
    sent_1, sent_2 = data['sentence_1'], data['sentence_2']
    embeddings = embed([sent_1,sent_2])
    score = cosine_similarity([embeddings.numpy().astype(np.float64)[0]],[embeddings.numpy().astype(np.float64)[1]])[0][0]
    return jsonify(similarity=score), 200

if __name__ == '__main__':
    print('Loading Model and Flask Starting Server...')
    app.run()

from flask import Flask
from flask import request
from flask import jsonify
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/embeddings', methods=['GET'])
def embeddings():
    #embed = request.args.to_dict()
    sentence = request.args.get('sentence')
    #embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embeddings = embed([
        sentence])
    embeddings = embeddings.numpy().tolist()[0]
    return jsonify(embedding=embeddings)


@app.route('/embeddings/bulk', methods=['POST'])
def bulk():
    #data = request.form.get('sentences')
    data = request.get_json()
    #embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    sentences = []
    for sentence in data['sentences']:
        sentences.append(sentence)
    embeddings = embed(sentences)
    embeddings = embeddings.numpy().tolist()
    return jsonify(embedding=embeddings)

@app.route('/embeddings/similarity', methods=['POST'])
def similarity():
    #data = request.form.get('sentences')
    data = request.get_json()
    #embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    sent_1, sent_2 = data['sentence_1'], data['sentence_2']
    embeddings = embed([sent_1,sent_2])
    score = cosine_similarity([embeddings.numpy().astype(np.float64)[0]],[embeddings.numpy().astype(np.float64)[1]])[0][0]
    return jsonify(similarity=score)

if __name__ == '__main__':
    print('Loading Model and Flask Starting Server...')
    #embed = None
    # global embed
    # embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    app.run()
# curl "http://localhost:5000/embeddings?sentence=the+quick+brown+fox" 
# curl -X POST -H "Content-Type: application/json" -d @payload_2.json http://localhost:5000/embeddings/bulk

from flask import Flask
from flask import request
from flask import jsonify
#import tensorflow_hub as hub


app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

#@app.route('/embeddings', methods=['GET'])
# def embeddings():
#     #embed = request.args.to_dict()
#     sentence = request.args.getlist('sentence')
#     embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#     embeddings = embed([
#         sentence])
#     return jsonify(embedding=embeddings)


@app.route('/embeddings/bulk', methods=['POST'])
def bulk():
    #embed = request.args.to_dict()
    data = request.get_json()
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embedding = []
    for sentence in jsonify(data):
        embeddings = embed([sentence])
    return jsonify(embedding=embeddings)


# curl "http://localhost:5000/embeddings?sentence=the+quick+brown+fox" 
# curl -X POST -H "Content-Type: application/json" -d @payload_2.json http://localhost:5000/embeddings/bulk

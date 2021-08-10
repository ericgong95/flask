from flask import Flask
from flask import request
from flask import jsonify
import tensorflow_hub as hub


app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/embeddings', methods=['GET'])
def embeddings():
    #embed = request.args.to_dict()
    sentence = request.args.getlist('sentence')
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embeddings = embed([
        sentence])
    return jsonify(embedding=embed)


@app.route('/embeddings/bulk', methods=['POST'])
def embeddings():
    #embed = request.args.to_dict()
    data = request.get_json(force=True)
    return data

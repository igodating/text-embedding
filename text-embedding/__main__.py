import json

from flask import Flask, request, jsonify
import yaml
from sentence_transformers import SentenceTransformer
import os
import numpy

app = Flask(__name__)
config = None
sentence_transformer_model_api = None

class ResultItem:

    sentence_id = None
    embedding = None

    def __init__(self, sentence_id, embedding):
        self.sentence_id = sentence_id
        self.embedding = embedding

class Result:

    sentences = None
    global_embedding = None

    def __init__(self, sentences, global_embedding):
        self.sentences = sentences
        self.global_embedding = global_embedding

class ResultEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, Result):
            return obj.__dict__
        if isinstance(obj, ResultItem):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)

class SentenceTransformerService:

    model_name = None
    model_root_path = None
    model = None

    def __init__(self, model_name, model_root_path):
        self.model_name = model_name
        self.model_root_path = model_root_path
        self.init_model()

    def init_model(self):
        model_path = self.model_root_path + "/" + self.model_name
        if not os.path.exists(model_path):
            self.save_model(model_path)
        self.model = SentenceTransformer(model_path)

    def save_model(self, path_to_save):
        model_to_save = SentenceTransformer(self.model_name)
        model_to_save.save(path_to_save)

    def get_embedding(self, sentence):
        return self.model.encode(sentence)

@app.route('/api/v1/embeddings', methods=['POST'])
def embeddings_route():
    sentences = request.json.get('sentences')

    if len(sentences) == 0:
        return Result(None, None)

    result_list = list()

    main_embedding = None

    for sentence in sentences:
        embedding = sentence_transformer_model_api.get_embedding(sentence["value"])

        if main_embedding is None:
            main_embedding = embedding
        else:
            main_embedding = numpy.add(main_embedding, embedding)

        result_list.append(ResultItem(sentence["sentence_id"], embedding.tolist()))

    result = Result(result_list, main_embedding.tolist())

    response = app.response_class(
        response=json.dumps(result, cls=ResultEncoder),
        status=200,
        mimetype='application/json'
    )

    return response

if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.full_load(f)

    sentence_transformer_model_api = SentenceTransformerService(config['model']['name'], config['model']['root-path'])
    app.run()




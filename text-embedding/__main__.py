from flask import Flask, request
import yaml
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
config = None
sentence_transformer_model_api = None

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


    def get_embeddings(self, sentences):
        return self.model.encode(sentences)

@app.route('/api/v1/embeddings', methods=['POST'])
def embeddings_route():
    sentences = request.json.get('sentences')
    embeddings = sentence_transformer_model_api.get_embeddings(sentences)

    result_list = list()

    for embedding in embeddings:
        result_list.append(embedding.tolist())

    return result_list

if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.full_load(f)

    sentence_transformer_model_api = SentenceTransformerService(config['model']['name'], config['model']['root-path'])
    app.run()




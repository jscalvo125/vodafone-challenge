from flask import Flask, request, jsonify
from nlp.recommender import ProductsRecommender
import psutil
import os
import pickle

app = Flask(__name__)
dir_name = os.path.dirname(__file__)
model_path_name = os.path.join(dir_name, "../models/trained_model.pkl")
test_data_name = os.path.join(dir_name, "../data/dataset_en_test.json")

recommender = ProductsRecommender(test_data_name, 'reviewer_id', 'product_id', model="en_core_web_lg")
with open(model_path_name, "rb") as input_file:
    spacy_pipeline = pickle.load(input_file)


def get_profiling_info():
    app.logger.info("Virtual memory = {} %".format(psutil.virtual_memory().percent))
    app.logger.info(
        "Available memory = {} %".format(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total))
    app.logger.info("CPU usage = {} %".format(psutil.cpu_percent()))
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    app.logger.info('App''s memory use (in GB): {}'.format(memoryUse))


@app.route('/')
def hello_world():
    return 'Hi! This is an API for predicting Amazon reviews. Please read README.md to know how to use me!'


@app.route('/predict_category', methods=['POST'])
def predict_category():
    try:
        get_profiling_info()
        test_data = request.get_json()
        app.logger.info("Calling the prediction for review text {}".format(test_data))
        parsed_doc = spacy_pipeline(test_data["review_body"])
        test_data["predicted_categories"] = parsed_doc.cats
        res = {"review": test_data["review_body"],
               "predicted_categories": test_data["predicted_categories"]
               }
        return jsonify(status='200 OK', label=res)
    except Exception as e:
        app.logger.info("Failed to call the prediction method - {}".format(e))
        return jsonify(status='401', label=e)

@app.route('/recommend_products', methods=['POST'])
def recommend_products():
    try:
        get_profiling_info()
        test_data = request.get_json()
        app.logger.info("Calling the recommender for user {}".format(test_data))
        recommended_products = recommender.recommend(test_data['reviewer_id'], 'reviewer_id', 'product_id')

        res = {"reviewer_id": test_data["reviewer_id"],
               "recommended_products": recommended_products
               }
        return jsonify(status='200 OK', label=res)
    except Exception as e:
        app.logger.info("Failed to call the prediction method - {}".format(e))
        return jsonify(status='401', label=e)


app.run(host='0.0.0.0', port=8080)

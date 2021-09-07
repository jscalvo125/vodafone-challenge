# Vodafone ML Challenge

This repository includes a [jupyter notebook](./notebooks/challenge.ipynb) that explores the amazon dataset.

# Components

This repo contains the following packages/folders:

-  [app](./app) The Flask app that exposes and serves the product category prediction (http://localhost:8080/predict_category) and http://localhost:8080/recommend_products.
-  [data](./data) The training and testing dataset used for training and testing the ML models.
-  [models](./models) Contains the trained model used by the flask application to predict in pickle (binary) format.
-  [tests](./tests) Unit tests for the ML model and its pre-processing+prediction pipelines.
-  [titanic_ml](./nlp) package to perform the pre-process+prediction

# How to install

To install, make sure you have Python==3.8.x and pip==latest in your environment.
Run: pip install -r requirements.txt to install the required dependencies.
IMPORTANT: Run python -m spacy download en_core_web_sm and python -m spacy download en_core_web_md to donwload the required spacy Languages used in the whole project

# How to run
Type: python -m app to run the Flask app in your environment. The app points to 0.0.0.0:8080/
Alternatively, run: python main.py to run the app.

The model is loaded(spacy Pipeline object in a pickle file) from the /models folder trained_model.pkl file

# How to use
##Train: To train a new model, run python -m nlp <train_file.json> <test_file.json> to generate a new .pkl file in the /models folder.

## Test:
Call the URI http://localhost:8080/predict_category (change localhost to your IP address/host name) via POST with the following parameters:
Alternatevely, you an call http://localhost:8080/recommend_products to use the product recommender (top 5):

## Headers
content-type: "Application/json"
## Body(an example)
{
    "review_id": "en_0199937",
    "product_id": "product_en_0902516",
    "reviewer_id": "reviewer_en_0097389",
    "stars": "1",
    "review_body": "These are AWFUL. They are see through, the fabric feels like tablecloth, and they fit like children’s clothing. Customer service did seem to be nice though, but I regret missing my return date for these. I wouldn’t even donate them because the quality is so poor.",
    "review_title": "Don’t waste your time!",
    "language": "en",
    "product_category": "apparel"
}

## Response
Product category prediction:
{
    "label": {
        "predicted_categories": {
            "apparel": 0.07987654209136963,
            "automotive": 0.0451403371989727,
            "baby_product": 0.013818043284118176,
            "beauty": 0.07933609932661057,
            "book": 0.02054043859243393,
            "camera": 0.04655729979276657,
            "digital_ebook_purchase": 0.034544624388217926,
            "drugstore": 0.03475000336766243,
            "electronics": 0.05887753516435623,
            "grocery": 0.02932930365204811,
            "jewelry": 0.08957303315401077,
            "kitchen": 0.04971007630228996,
            "lawn_and_garden": 0.05587203800678253,
            "office_product": 0.06298227608203888,
            "other": 0.049658097326755524,
            "pc": 0.06272172182798386,
            "pet_products": 0.06349823623895645,
            "sports": 0.031166568398475647,
            "toy": 0.024613747373223305,
            "video_games": 0.03977556899189949,
            "watch": 0.027658427134156227
        },
        "review": "These are AWFUL. They are see through, the fabric feels like tablecloth, and they fit like children’s clothing. Customer service did seem to be nice though, but I regret missing my return date for these. I wouldn’t even donate them because the quality is so poor."
    },
    "status": "200 OK"
}

Top Products recommendation:
"label": {
        "recommended_products": [
            "product_en_0906230",
            "product_en_0377343",
            "product_en_0250578",
            "product_en_0559225",
            "product_en_0233183"
        ],
        "reviewer_id": "reviewer_en_0097389"
    },
    "status": "200 OK"
}

# Deployment in container/ Productionizing

If you want to deploy into a container, [Dockerfile](./Dockerfile) contains a deployment into a python 3.8 image with all the necessary components in it.
you can test it using [start.sh](./start.sh) if you run it in your linux env.
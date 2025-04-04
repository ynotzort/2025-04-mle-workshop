import pickle
from flask import Flask, request, jsonify
import os

MODEL_PATH = os.getenv("MODEL_PATH", "model.bin")
VERSION = os.getenv("VERSION", "not defined")

with open(MODEL_PATH, "rb") as f_in:
    model = pickle.load(f_in)

# trip = {
#     "PULocationID": "43",
#     "DOLocationID": "238",
#     "trip_distance": 1.16,
# }
# prediction = model.predict(trip)

# "feature engineering"
def prepare_features(ride):
    features = dict()
    features["PULocationID"] = str(ride["PULocationID"])
    features["DOLocationID"] = str(ride["DOLocationID"])
    features["trip_distance"] = float(ride["trip_distance"])
    return features

def predict(features):
    preds = model.predict(features)
    return float(preds[0])

app = Flask("duration-preidction")


@app.route("/version", methods=["GET"])
def version():
    return VERSION

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    prediction = predict(features)
    
    result = {
        "prediction": {"duration" : prediction}
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
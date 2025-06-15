from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the model and scaler
model = pickle.load(open("land_price_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Label encoders (must match training)
location_map = {"Chennai": 0, "Coimbatore": 1, "Madurai": 2, "Salem": 3}
highway_map = {"Yes": 1, "No": 0}

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    price = None
    total_price = None
    if request.method == "POST":
        location = request.form["location"]
        area = float(request.form["area"])
        road_width = float(request.form["road_width"])
        distance = float(request.form["distance"])
        highway = request.form["highway"]

        # Encode categorical variables
        loc_encoded = location_map.get(location, 0)
        highway_encoded = highway_map.get(highway, 0)

        # Prepare features
        features = np.array([[area, road_width, distance, loc_encoded, highway_encoded]])
        features_scaled = scaler.transform(features)
        price_per_sqft = np.expm1(model.predict(features_scaled)[0])

        price = round(price_per_sqft, 2)
        total_price = round(price_per_sqft * area, 2)

    return render_template("index.html", price=price, total_price=total_price)

if __name__ == "__main__":
    app.run(debug=True)

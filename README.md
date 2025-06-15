# Land-Price-Prediction-Using-Linear-Regression
To develop a web-based application that predicts the land price per square foot in Tamil Nadu using linear regression based on user inputs like area, road width, location, and distance from the city center.
# 🏡 Land Price Prediction Web App (Tamil Nadu)

This project is a **web application** that predicts the **land price per square foot** in Tamil Nadu based on user inputs using **Linear Regression**. The model was trained on synthetic and real-world inspired data with features like area, road width, location, and distance from the city center.

---

## 📌 Features

- Predict **price per square foot** for land
- Calculate **total price** based on entered area
- User-friendly **web interface** using Flask
- Trained using **Linear Regression** with feature scaling
- Supports dynamic predictions using real-time user input

---

## 🚀 Demo

![Land Price Prediction Screenshot](./screenshot.png)

Try it locally:  
Open in your browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS
- **ML**: Scikit-learn, Pandas, NumPy
- **Model Saving**: Pickle

---

## 📂 Dataset

**File**: `tamilnadu_land_price_dataset.csv`

### Sample Columns:
- `Location` (e.g., Chennai, Madurai)
- `Area_sqft` (e.g., 3000)
- `Road_Width_ft` (e.g., 24)
- `Distance_from_citycenter_km` (e.g., 5.6)
- `Near_Highway` (Yes/No)
- `Price_per_sqft` (target)

---

## 📈 Model Training (Steps)

1. Encode categorical features (`Location`, `Near_Highway`)
2. Remove outliers using IQR on `Price_per_sqft`
3. Apply `StandardScaler` for scaling
4. Use `LinearRegression` from scikit-learn
5. Evaluate using:
   - **MAE**
   - **MSE**
   - **R² Score**
6. Save the model and scaler using `pickle`

---

## 🧪 Evaluation Metrics

| Metric | Score |
|--------|-------|
| MAE    | ~294.44 |
| MSE    | ~118880.33 |
| R²     | ~0.8644 |

---

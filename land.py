import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r"F:\land\tamilnadu_land_price_dataset.csv")  # Adjust the path if needed

# ----------------------
# 1. DATA PREPARATION
# ----------------------

# Encode categorical variables
le_location = LabelEncoder()
le_highway = LabelEncoder()

df['Location_encoded'] = le_location.fit_transform(df['Location'])
df['Near_Highway_encoded'] = le_highway.fit_transform(df['Near_Highway'])

# Drop missing values (if any)
df.dropna(inplace=True)

# ----------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Plot: Price vs Area
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Area_sqft", y="Price_per_sqft", hue="Location")
plt.title("Price vs Area")
plt.tight_layout()
plt.show()

# Plot: Location vs Price
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Location", y="Price_per_sqft")
plt.title("Location vs Price per sqft")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Box plots to identify outliers
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[["Area_sqft", "Road_Width_ft", "Distance_from_citycenter_km", "Price_per_sqft"]])
plt.title("Boxplot for Numerical Features")
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("tamilnadu_land_price_dataset.csv")

# Encode categorical variables
le_location = LabelEncoder()
le_highway = LabelEncoder()
df['Location_encoded'] = le_location.fit_transform(df['Location'])
df['Near_Highway_encoded'] = le_highway.fit_transform(df['Near_Highway'])

# Drop missing values
df.dropna(inplace=True)

# Features and target
X = df[["Area_sqft", "Road_Width_ft", "Distance_from_citycenter_km", "Location_encoded", "Near_Highway_encoded"]]
y = df["Price_per_sqft"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Model: Ridge Regression (Regularized Linear Regression)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Actual vs Predicted Plot with Fit Line
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='green', label="Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Fit")
plt.xlabel("Actual Price per sqft")
plt.ylabel("Predicted Price per sqft")
plt.title("Actual vs Predicted Price (with Ridge Regression)")
plt.legend()
plt.tight_layout()
plt.show()

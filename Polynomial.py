# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load a real-life housing dataset 
data = pd.read_csv('housing_dataset.csv')  # Load your dataset here

# Select relevant features and the target variable
X = data[['SquareFootage']]
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features
degree = 2  # Degree of the polynomial
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Create a polynomial regression model
model = LinearRegression()

# Fit the model to the polynomial training data
model.fit(X_train_poly, y_train)

# Make predictions on the polynomial test data
y_pred = model.predict(X_test_poly)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model coefficients and evaluation metrics
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2): {r2}')

# Visualize the predictions
plt.scatter(X_test, y_test, label="Actual Prices")
plt.scatter(X_test, y_pred, color='red', label="Predicted Prices")
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("Square Footage vs. Price (Polynomial Regression)")
plt.legend()
plt.show()

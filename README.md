# linear-regression-house-prices-
A simple Python project demonstrating linear regression for predicting house prices based on features like square meters and number of bedrooms. Includes data simulation, model building using scikit-learn, model evaluation (RMSE), and a visualization comparing true vs. predicted prices.
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(0)
n = 200
square_meters = np.random.normal(100, 15, n)
num_bedrooms = np.random.randint(1, 5, n)
price = 3000 * square_meters + 10000 * num_bedrooms + np.random.normal(0, 20000, n)

df = pd.DataFrame({
    'square_meters': square_meters,
    'num_bedrooms': num_bedrooms,
    'price': price
})

X = df[['square_meters', 'num_bedrooms']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: ${rmse:,.0f}")

# Visualization: True vs. Predicted prices
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('True vs. Predicted House Prices')
plt.grid(True)
plt.tight_layout()
plt.show()

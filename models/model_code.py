#import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
from sklearn.preprocessing import StandardScaler

# Load the synthetic dataset on carbon emission
df = pd.read_csv('Carbon Emission.csv')
df = df.dropna()
df = df.drop_duplicates()
df = df.drop(columns=['Recycling', 'Cooking_With'])
df.head()

# Transforming categorical fields according to the provided logic
df["Body Type"] = df["Body Type"].map({'underweight':0, 'normal':1, 'overweight':2, 'obese':3})
df["Sex"] = df["Sex"].map({'female':0, 'male':1})
df = pd.get_dummies(df, columns=["Diet","Heating Energy Source","Transport","Vehicle Type"], dtype=int)
df["How Often Shower"] = df["How Often Shower"].map({'less frequently':0, 'daily':1, "twice a day":2, "more frequently":3})
df["Social Activity"] = df["Social Activity"].map({'never':0, 'sometimes':1, "often":2})
df["Frequency of Traveling by Air"] = df["Frequency of Traveling by Air"].map({'never':0, 'rarely':1, "frequently":2, "very frequently":3})
df["Waste Bag Size"] = df["Waste Bag Size"].map({'small':0, 'medium':1, "large":2,  "extra large":3})
df["Energy efficiency"] = df["Energy efficiency"].map({'No':0, 'Sometimes':1, "Yes":2})

df.head()

#Create XGBoost model to predict 'CarbonEmission' based on input features

X = df.drop(columns=['CarbonEmission'])
y = df['CarbonEmission']

# Convert categorical features to 'category' dtype
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the scaler to the DataFrame's data
scaler = StandardScaler().fit(X_train)

# Initialize the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, enable_categorical=True)

# Train the model
model.fit(scaler.transform(X_train), y_train)

# Predict on the test set
y_pred = model.predict(scaler.transform(X_test))

# Show model summary
print(model)

# Plot the predicted values against the actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Calculate percentage mean absolute error
mean_actual = np.mean(y_test)
percentage_mae = (mae / mean_actual) * 100
print(f"Percentage Mean Absolute Error: {percentage_mae:.2f}%")

# Plot feature importance
xgb.plot_importance(model)
plt.title('Feature Importance')
plt.show()

# Save the model into a .sav file
with open('model.sav', 'wb') as f:
    pickle.dump(model, f)

# Save the fitted scaler object to a .sav file
with open('scaler.sav', 'wb') as file:
    pickle.dump(scaler, file)
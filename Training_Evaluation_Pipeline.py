import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

# Load your preprocessed data
file_path = ''
df = pd.read_csv(file_path)

# Split the data into features (X) and target variable (y)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest Regressor model with lower computing parameters
rf_model_low_resources = RandomForestRegressor(
    n_estimators=50,  # You can further reduce this number
    max_depth=20,      # Adjust as needed, lower values for shallower trees
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Train the model
rf_model_low_resources.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model_low_resources.predict(X_test)

# Evaluate the model
rmsle_score = mean_squared_log_error(y_test, y_pred) ** 0.5
print("RMSLE Score:", rmsle_score)

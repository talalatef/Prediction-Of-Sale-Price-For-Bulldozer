import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer


# Step 1: Read the file into a DataFrame
def read_file(file_path):
    return pd.read_csv(file_path)

# Step 2: Drop unnecessary columns
def drop_columns(df, columns_to_keep):
    return df[columns_to_keep]

# Step 3: Handling missing values
def impute_missing_data(df, numeric_columns_with_missing, categorical_columns_with_missing):
    # Numeric columns imputation with mean
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns_with_missing] = numeric_imputer.fit_transform(df[numeric_columns_with_missing])

    # Categorical columns imputation with most frequent or missing
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns_with_missing] = categorical_imputer.fit_transform(df[categorical_columns_with_missing])

    return df

# Step 4: Feature engineering on 'saledate'
def feature_engineering_saledate(df):
    df['saledate'] = pd.to_datetime(df['saledate'])
    df['sale_year'] = df['saledate'].dt.year
    df['sale_month'] = df['saledate'].dt.month
    df['sale_day'] = df['saledate'].dt.day
    df['sale_dayofweek'] = df['saledate'].dt.dayofweek
    # df['saledate'] = df['saledate'].dt.date  # Convert back to date (optional)
    return df.drop(['saledate'], axis=1)

# Step 5: Drop rows where 'YearMade' < 1900
def filter_yearmade(df):
    return df[df['YearMade'] >= 1900]

# Step 5: label encoding (you can change between them and see the impact)
def label_encoding(df, features_to_encode):
    label_encoder = LabelEncoder()
    for feature in features_to_encode:
        df[feature + '_label_encoded'] = label_encoder.fit_transform(df[feature])
    return df.drop(features_to_encode, axis=1)

# # Step 5: Frequency encoding
# def frequency_encoding(df, features_to_encode):
#     for feature in features_to_encode:
#         frequency_encoding = df[feature].value_counts().to_dict()
#         df[feature + '_freq_encoded'] = df[feature].map(frequency_encoding)
#     return df.drop(features_to_encode, axis=1)

# Step 6: Save the preprocessed data to a specified path
def save_preprocessed_data(df, file_path):
    df.to_csv(file_path, index=False)

# Step 7: train_and_evaluate_model
def train_and_evaluate_model(df, file_path):
    # Load your preprocessed data
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
    return rmsle_score

def handle_mixed_data_types(df, columns_to_check):
    for column in columns_to_check:
        unique_types = df[column].apply(type).unique()
        if len(unique_types) > 1:
            print(f"Column '{column}' has mixed data types: {unique_types}")
            
            # Handle the mixed data types, for example, convert to numeric or handle separately
            # For simplicity, let's convert the entire column to strings
            df[column] = df[column].astype(str)

    return df

# Define the columns to keep
columns_to_keep = ['SalesID', 'SalePrice', 'MachineID', 'ModelID', 'datasource',
       'auctioneerID', 'YearMade', 'MachineHoursCurrentMeter', 'UsageBand',
       'saledate', 'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc',
       'fiModelSeries', 'fiModelDescriptor', 'ProductSize',
       'fiProductClassDesc', 'state', 'ProductGroup', 'ProductGroupDesc',
       'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control',
       'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension',
       'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics',
       'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size',
       'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow',
       'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb',
       'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type',
       'Travel_Controls', 'Differential_Type', 'Steering_Controls']

# Define the columns with missing values (both numeric and categorical)
numeric_columns_with_missing = ['SalesID', 'SalePrice', 'MachineID', 'ModelID', 'datasource','auctioneerID', 'YearMade', 'MachineHoursCurrentMeter']
categorical_columns_with_missing = ['UsageBand', 'fiModelDesc', 'fiBaseModel',
       'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor', 'ProductSize',
       'fiProductClassDesc', 'state', 'ProductGroup', 'ProductGroupDesc',
       'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control',
       'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension',
       'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics',
       'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size',
       'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow',
       'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb',
       'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type',
       'Travel_Controls', 'Differential_Type', 'Steering_Controls']

# Define the features to encode
features_to_encode = ['UsageBand', 'fiModelDesc', 'fiBaseModel',
       'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor', 'ProductSize',
       'fiProductClassDesc', 'state', 'ProductGroup', 'ProductGroupDesc',
       'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control',
       'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension',
       'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics',
       'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size',
       'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow',
       'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb',
       'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type',
       'Travel_Controls', 'Differential_Type', 'Steering_Controls']

# Define the columns to be normalized
columns_to_normalize = ['SalesID', 'MachineID', 'ModelID', 'datasource',
       'auctioneerID', 'YearMade', 'MachineHoursCurrentMeter', 'UsageBand',
       'saledate', 'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc',
       'fiModelSeries', 'fiModelDescriptor', 'ProductSize',
       'fiProductClassDesc', 'state', 'ProductGroup', 'ProductGroupDesc',
       'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control',
       'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension',
       'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics',
       'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size',
       'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow',
       'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb',
       'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type',
       'Travel_Controls', 'Differential_Type', 'Steering_Controls']

# Create the pipeline
data_preparation_pipeline = Pipeline([
    ('read_file', FunctionTransformer(read_file)),
    ('drop_columns', FunctionTransformer(drop_columns, kw_args={'columns_to_keep': columns_to_keep})),
    ('missing_data_imputation', FunctionTransformer(impute_missing_data, kw_args={
        'numeric_columns_with_missing': numeric_columns_with_missing,
        'categorical_columns_with_missing': categorical_columns_with_missing
    })),
    ('feature_engineering_saledate', FunctionTransformer(feature_engineering_saledate)),
    ('filter_yearmade', FunctionTransformer(filter_yearmade)),
    ('handle_mixed_data_types', FunctionTransformer(handle_mixed_data_types, kw_args={'columns_to_check': features_to_encode})),
    ('label_encoding', FunctionTransformer(label_encoding, kw_args={'features_to_encode': features_to_encode})),
    ('save_preprocessed_data', FunctionTransformer(save_preprocessed_data, kw_args={'file_path': 'preprocessed_data.csv'})),
    ('train_model', FunctionTransformer(train_and_evaluate_model, kw_args={'file_path': 'preprocessed_data.csv'}))
])

# Example usage:
file_path = 'Train.csv'
evaluation_result = data_preparation_pipeline.transform(file_path)
print("Evaluation Result:", evaluation_result)

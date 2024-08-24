import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'supply_chain_data.csv'
df = pd.read_csv(file_path)

# Display the dataset in the app
st.subheader('Dataset Overview')
st.write(df.head())

# Sidebar for user input parameters
st.sidebar.header('User Input Parameters')

# Convert categorical columns to numeric if necessary
df = pd.get_dummies(df, drop_first=True)

# Define available columns for selection
available_columns = list(df.columns)

# Select target column
target_column = st.sidebar.selectbox('Select Target Column for Prediction', available_columns)

# Select features to use for prediction
selected_features = st.sidebar.multiselect('Select Features to Use for Prediction', available_columns, default=available_columns)

# Drop the target column from features
if target_column in selected_features:
    selected_features.remove(target_column)

# Define toggles and dropdowns for user input
def user_input_features(selected_features):
    features = {}
    for feature in selected_features:
        if df[feature].dtype in ['float64', 'int64']:
            features[feature] = st.sidebar.slider(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
        else:
            unique_values = df[feature].unique()
            features[feature] = st.sidebar.selectbox(feature, unique_values)
   
    features_df = pd.DataFrame(features, index=[0])
    return features_df

df_input = user_input_features(selected_features)

st.subheader('User Input parameters')
st.write(df_input)

# Prepare features and target variable
X = df[selected_features]
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
prediction = model.predict(df_input)

st.subheader('Prediction')
st.write(f"Predicted {target_column}: {prediction[0]}")

# Optionally, display model performance metrics
y_pred = model.predict(X_test)
st.subheader('Model Performance')
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
st.write(f"R-squared: {r2_score(y_test, y_pred)}")


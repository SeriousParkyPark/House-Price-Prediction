##THIS HERE IS JUST ME TRYING SOMETHING:


import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Function for loading and cleaning the dataset
@st.cache_data
def load_data():
    url = "https://data.insideairbnb.com/canada/qc/montreal/2024-09-13/visualisations/listings.csv"
    df = pd.read_csv(url)
    # Remove unnecessary columns and filter prices above 9000
    columns_to_remove = ['id', 'host_id', 'license', 'neighbourhood_group', 'host_name', 'last_review', 'reviews_per_month']
    df = df.drop(columns=columns_to_remove, errors='ignore')
    df = df[df['price'] <= 9000]
    df = df.dropna()
    df = df.replace([float('inf'), float('-inf')], float('nan'))
    df = df.dropna()
    return df

# Function for preprocessing and training the model
@st.cache_resource
def train_model(df):
    categorical_features = ['room_type', 'neighbourhood']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    df = pd.concat([df, encoded_df], axis=1)

    # Prepare features (X) and target (y)
    X = df[['latitude', 'longitude'] + list(encoder.get_feature_names_out(categorical_features))]
    y = df['price']
    combined_df = pd.concat([X, y], axis=1).dropna()
    X = combined_df.drop('price', axis=1)
    y = combined_df['price']
    assert len(X) == len(y), f"Inconsistent lengths: X={len(X)}, y={len(y)}"
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, encoder, mse, r2

# Page for exploring the dataset
def dataset_page(df):
    st.title("Explore the Dataset")
    st.write("This page displays the Montreal Airbnb dataset used for training the model.")
    st.write(df.head(10))
    st.write("Dataset Summary:")
    st.write(df.describe())

# Page for predictions
def prediction_page(df, model, encoder, mse, r2):
    st.title("Predict Airbnb Rental Price")

    #st.subheader("Model Metrics")
    #st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    #st.write(f"R-squared (R²): {r2:.2f}")

    st.subheader("Make a Prediction")
    latitude = st.number_input('Latitude', value=df['latitude'].mean())
    longitude = st.number_input('Longitude', value=df['longitude'].mean())
    room_type = st.selectbox('Room Type', df['room_type'].unique())
    neighbourhood = st.selectbox('Neighbourhood', df['neighbourhood'].unique())

    # Prepare input data
    input_data = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'room_type': [room_type],
        'neighbourhood': [neighbourhood]
    })

    # Encode input data
    encoded_input = encoder.transform(input_data[['room_type', 'neighbourhood']])
    encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['room_type', 'neighbourhood']))
    input_data = pd.concat([input_data[['latitude', 'longitude']], encoded_input_df], axis=1)

    # Predict
    prediction = model.predict(input_data)
    st.write(f"Predicted Rental Price: ${prediction[0]:.2f}")

# Main Streamlit app
def main():
    st.sidebar.title("Navigation")
    pages = ["Dataset", "Predict Price"]
    choice = st.sidebar.radio("Go to", pages)

    # Load and clean data
    df = load_data()

    # Train the model
    model, encoder, mse, r2 = train_model(df)

    if choice == "Dataset":
        dataset_page(df)
    elif choice == "Predict Price":
        prediction_page(df, model, encoder, mse, r2)

if __name__ == "__main__":
    main()

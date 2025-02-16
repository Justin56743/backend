import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from joblib import dump, load

# Paths for saving and loading models
MODEL_PATH = r'backend\models\kmeans_model.joblib'
SCALER_PATH = r'backend\models\scaler.joblib'
DATA_PATH = r'C:\Users\anuj5\Documents\Main_website\backend\data-final.xls'

# Ensure the models directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def load_and_preprocess_data(filepath: str):
    data_raw = pd.read_csv(filepath, sep='\t')
    data = data_raw.copy()

    # Drop unnecessary columns and missing values
    data.drop(data.columns[50:107], axis=1, inplace=True)
    data.drop(data.columns[51:], axis=1, inplace=True)
    data.dropna(inplace=True)

    # Normalize the data
    df = data.drop('country', axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_normalized = scaler.fit_transform(df)

    return df_normalized, scaler

def train_kmeans_model(data_normalized, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    k_fit = kmeans.fit(data_normalized)
    return k_fit, kmeans

def save_model_and_scaler(model, scaler):
    dump(model, MODEL_PATH)
    dump(scaler, SCALER_PATH)

def load_model_and_scaler():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load(MODEL_PATH)
        scaler = load(SCALER_PATH)
        return model, scaler
    else:
        raise FileNotFoundError("Model or scaler files not found. Please train the model first.")

def predict_cluster(model, scaler, new_data):
    new_data_normalized = scaler.transform(new_data)
    return model.predict(new_data_normalized)

# Function to train and save the model if not already saved
def train_and_save_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        data_normalized, scaler = load_and_preprocess_data(DATA_PATH)
        
        # Determine optimal number of clusters using Elbow Method
        kmeans = KMeans()
        visualizer = KElbowVisualizer(kmeans, k=(2, 15))
        visualizer.fit(data_normalized[:5000])
        optimal_clusters = visualizer.elbow_value_
        
        k_fit, kmeans = train_kmeans_model(data_normalized, optimal_clusters)
        save_model_and_scaler(kmeans, scaler)
        print("Model and scaler saved successfully.")
    else:
        print("Model and scaler already exist.")

# Train and save the model when the script is run
if __name__ == "__main__":
    train_and_save_model()
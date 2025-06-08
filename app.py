from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load artifacts
imputer = joblib.load('personality_imputer.joblib')
scaler = joblib.load('personality_scaler.joblib')
model = joblib.load('personality_rf_model.joblib')

PERSONALITY_TRAITS = {
    0: "Type A - Analytical Leader",
    1: "Type B - Creative Explorer",
    2: "Type C - Social Coordinator",
    3: "Type D - Practical Implementer",
    4: "Type E - Balanced Mediator"
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate request
        data = request.json
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        # Convert to numpy array
        features = np.array(data['features']).reshape(1, -1)
        
        # Convert to DataFrame with feature names
        if hasattr(scaler, 'feature_names_in_'):
            feature_names = scaler.feature_names_in_
        else:
            # Fallback if feature names aren't available
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        # Validate feature count
        if len(feature_names) != features.shape[1]:
            return jsonify({
                'error': f'Expected {len(feature_names)} features, got {features.shape[1]}'
            }), 400
            
        features_df = pd.DataFrame(features, columns=feature_names)
        
        # Preprocess
        features_imputed = imputer.transform(features_df)
        features_scaled = scaler.transform(features_df)  # Now has proper feature names
        
        # Predict
        prediction = model.predict(features_scaled)
        cluster = int(prediction[0])
        
        return jsonify({
            'cluster': cluster,
            'traits': PERSONALITY_TRAITS.get(cluster, "Unknown Personality Type")
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
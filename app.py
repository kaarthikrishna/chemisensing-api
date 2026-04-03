from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import os

# Initialize the web server
app = Flask(__name__)

# Load the model into memory
print("Loading model...")
model = load_model('peroxide_cnn_model.h5')

# !!! STOP: YOU MUST CHANGE THESE TWO NUMBERS !!!
TRAIN_MEAN = 5.79880805333569e-06  
TRAIN_STD = 2.379785709247936e-06  

# Create the route where the Arduino will send data
@app.route('/predict', methods=['POST'])
def predict_peroxide():
    try:
        # Get the JSON data sent by the Arduino
        data = request.get_json()
        
        if not data or 'sensor_array' not in data:
            return jsonify({'error': 'No sensor_array found in JSON'}), 400
            
        raw_signal = data['sensor_array']
        
        if len(raw_signal) != 202:
            return jsonify({'error': f'Expected 202 points, got {len(raw_signal)}'}), 400

        # Standardize and reshape the data
        signal_array = np.array(raw_signal)
        scaled_signal = (signal_array - TRAIN_MEAN) / TRAIN_STD
        reshaped_signal = scaled_signal.reshape(1, 202, 1)

        # Run the AI Prediction
        prediction = model.predict(reshaped_signal)
        probability = float(prediction[0][0])
        is_peroxide = bool(probability > 0.5)

        # Send the answer back to the Arduino!
        return jsonify({
            'peroxide_detected': is_peroxide,
            'confidence_score': probability
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# This part is required for local testing, but Render uses Gunicorn
if __name__ == '__main__':
    # Use the port Render assigns, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
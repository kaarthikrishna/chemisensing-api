from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load the highly optimized TFLite model
print("Loading TinyML TFLite model...")
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get the exact memory addresses for the input and output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# !!! STOP: YOU MUST CHANGE THESE TWO NUMBERS !!!
TRAIN_MEAN = 0.0  
TRAIN_STD = 1.0   

@app.route('/predict', methods=['POST'])
def predict_peroxide():
    try:
        data = request.get_json()
        
        if not data or 'sensor_array' not in data:
            return jsonify({'error': 'No sensor_array found'}), 400
            
        raw_signal = data['sensor_array']
        if len(raw_signal) != 202:
            return jsonify({'error': f'Expected 202 points, got {len(raw_signal)}'}), 400

        # Standardize the data
        signal_array = np.array(raw_signal)
        scaled_signal = (signal_array - TRAIN_MEAN) / TRAIN_STD
        
        # Reshape and explicitly convert to float32 (TFLite is very strict on data types!)
        reshaped_signal = scaled_signal.reshape(1, 202, 1).astype(np.float32)

        # Run the blazing fast TFLite Prediction
        interpreter.set_tensor(input_details[0]['index'], reshaped_signal)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        probability = float(prediction[0][0])
        is_peroxide = bool(probability > 0.5)

        return jsonify({
            'peroxide_detected': is_peroxide,
            'confidence_score': probability
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

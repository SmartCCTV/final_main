# flask_app.py
from flask import Flask, Response, jsonify
import cv2
import numpy as np
import onnxruntime
import json

app = Flask(__name__)

camera = cv2.VideoCapture(0)

onnx_model_path = 'First_MLP.onnx'
session = onnxruntime.InferenceSession(onnx_model_path)

input_name = session.get_inputs()[0].name

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
    return np.expand_dims(frame_transposed, axis=0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            input_tensor = preprocess_frame(frame)
            predictions = session.run(None, {input_name: input_tensor})

            predicted_class = np.argmax(predictions[0], axis=1)[0]
            prediction_confidence = np.max(predictions[0])
            prediction_result = {
                "class": int(predicted_class),
                "confidence": float(prediction_confidence)
            }

            yield (b'--frame\r\n'
                   b'Content-Type: application/json\r\n\r\n' +
                   json.dumps(prediction_result).encode('utf-8') + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture frame from camera"}), 500

    input_tensor = preprocess_frame(frame)
    predictions = session.run(None, {input_name: input_tensor})

    predicted_class = np.argmax(predictions[0], axis=1)[0]
    prediction_confidence = np.max(predictions[0])
    prediction_result = {
        "class": int(predicted_class),
        "confidence": float(prediction_confidence)
    }

    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(debug=True)

import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

set_global_policy('mixed_float16')

import numpy as np
import cv2
import json
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"], "allow_headers": "*"}})

# Initialize MediaPipe Hands - using same parameters as handtrack.py
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Changed to False to match handtrack.py
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def detect_hand_with_mediapipe(image):
    """Use MediaPipe to detect and crop the hand from an image"""
    # Convert to RGB (MediaPipe requires RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb_image.shape

    # Process the image with MediaPipe
    results = hands.process(rgb_image)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand bounding box
            x_list = []
            y_list = []
            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            # Add padding around hand
            padding = 30
            xmin, xmax = max(0, min(x_list) - padding), min(w, max(x_list) + padding)
            ymin, ymax = max(0, min(y_list) - padding), min(h, max(y_list) + padding)

            # Extract hand region
            if xmin < xmax and ymin < ymax:
                hand_region = image[ymin:ymax, xmin:xmax]
                return hand_region, True, {"xmin": int(xmin), "ymin": int(ymin), "xmax": int(xmax), "ymax": int(ymax)}

    # Return None for image when no hand is detected (instead of original image)
    return None, False, None


class SignLanguageDetector:
    def __init__(self, model_path, class_mapping_path):
        print("Loading model and configuration...")
        self.model = load_model(model_path)

        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)

        print("First 5 items in class mapping:", dict(list(self.class_mapping.items())[:5]))

        # Simplify class mapping to match handtrack.py approach
        self.REV_CLASS_MAP = {int(idx): name for name, idx in self.class_mapping.items()}

        print(f"Model output shape: {self.model.output_shape}")
        print(f"Number of classes in mapping: {len(self.class_mapping)}")

        self.img_size = (224, 224)
        # Remove prediction window for direct predictions
        self.inference_times = []

        # Model warmup
        print("Warming up model...")
        dummy_input = np.random.rand(1, *self.img_size, 3).astype('float32')
        self.model.predict(dummy_input, verbose=0)

    def preprocess_image(self, image_data):
        if isinstance(image_data, str) and ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        # Use MediaPipe for hand detection and cropping
        processed_img, hand_detected, hand_box = detect_hand_with_mediapipe(img)

        # Debug output
        if os.getenv('DEBUG_IMAGES'):
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = int(time.time())
            cv2.imwrite(f'{debug_dir}/{timestamp}_input.jpg', img)
            if processed_img is not None:
                cv2.imwrite(f'{debug_dir}/{timestamp}_processed.jpg', processed_img)

        # Return early if no hand is detected
        if not hand_detected or processed_img is None:
            return None, hand_detected, hand_box

        # Resize and normalize the image (matching handtrack.py preprocessing)
        resized = cv2.resize(processed_img, self.img_size)
        normalized = resized.astype('float32') / 255.0
        return np.expand_dims(normalized, axis=0), hand_detected, hand_box

    def predict(self, image_data):
        try:
            start_time = time.time()
            processed_image, hand_detected, hand_box = self.preprocess_image(image_data)

            # If no hand detected, return empty result
            if not hand_detected or processed_image is None:
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                return [("No hand detected", 0.0)], inference_time, False, None

            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)

            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # Get top predictions directly without smoothing
            top_3_idx = np.argsort(prediction[0])[-3:][::-1]
            top_3_probs = prediction[0][top_3_idx]

            top_3_classes = []
            for idx in top_3_idx:
                idx_int = int(idx)
                if idx_int in self.REV_CLASS_MAP:
                    top_3_classes.append(self.REV_CLASS_MAP[idx_int])
                else:
                    print(f"Warning: Index {idx_int} not found in class mapping")
                    top_3_classes.append(f"Unknown-{idx_int}")

            return list(zip(top_3_classes, top_3_probs.tolist())), inference_time, hand_detected, hand_box

        except Exception as e:
            import traceback
            print("Detailed error:", traceback.format_exc())
            raise e


detector = SignLanguageDetector(
    model_path="best_model.h5",
    class_mapping_path="class_mapping.json"
)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        image_data = request.json['image']
        predictions, inference_time, hand_detected, hand_box = detector.predict(image_data)

        formatted_predictions = [
            {'class': cls, 'confidence': float(conf)}
            for cls, conf in predictions
        ]

        response = {
            'predictions': formatted_predictions,
            'hand_detected': hand_detected,
            'inference_time_ms': round(inference_time * 1000, 2),
            'timestamp': time.time()
        }

        # Include hand box coordinates if available
        if hand_box:
            response['hand_box'] = hand_box

        return jsonify(response)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'avg_inference_time_ms': round(np.mean(detector.inference_times) * 1000, 2) if detector.inference_times else 0,
        'total_predictions': len(detector.inference_times)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

# SignBridge

This repository contains the core components for a real-time hand gesture recognition system that translates hand gestures into Malayalam alphabetic letters. The project aims to provide a fast and efficient way to convert sign language gestures into written Malayalam text.

## ✨ Project Highlights ✨

Our system is engineered for speed and accuracy, capable of:

👁️ Detecting hand gestures from live video feeds with precision.

✍️ Recognizing specific hand gestures that correspond to individual Malayalam alphabetic characters.

⚡ Translating recognized gestures into their respective Malayalam characters in a matter of seconds, enabling fluid communication.

This project stands as a powerful tool for enhancing accessibility and supporting the learning and application of Malayalam sign language.
## 📂 Repository Structure 📂
```
SignBridge/
├── best_model.h5
├── class_mapping.json
├── inference22.py
├── requirements.txt
├── environment.yml
├── train.py
└── README.md
```
## 🌟 Key Features at a Glance 🌟

⚡ Real-time Recognition: Experience near-instantaneous translation of gestures into text.

🗣️ Malayalam Alphabet Support: Specifically fine-tuned to understand and convert gestures for the complete Malayalam alphabet.

🌐 Robust Backend API: inference22.py offers a clean Flask API endpoint, making frontend integration a breeze.

✋ Advanced Hand Detection: Utilizes sophisticated techniques (like MediaPipe) to precisely isolate and focus on the hand region for accurate gesture analysis.

🧠 Deep Learning Powered: Built upon a highly efficient MobileNetV2 model, ensuring both speed and accuracy in classification.
## 🛠️ Setup and Installation 🛠️

Get your project up and running with these simple steps:

### 1. Clone the repository:
```bash
    git clone <your-repository-url>
    cd <your-repository-name>
```
### 2. Create and activate the Conda environment (recommended for dependency management):
```bash
    conda env create -f environment.yml
    conda activate py310_env
```
*Alternatively, if you prefer `pip`:*
```bash
    pip install -r requirments.txt  
```
### 3. Verify model and mapping files:
Ensure `best_model.h5` and `class_mapping.json` are located in the root directory. If not, adjust the file paths within `inference22.py` and `train.py` accordingly.

## 🏃‍♀️ Running the Backend (Inference) 🏃‍♂️

To fire up the real-time gesture recognition backend:
```bash
    python inference22.py
```
This command will launch a Flask server, typically accessible at `http://127.0.0.1:5000`. You can then send POST requests containing image data to the `/predict` endpoint for real-time translation.
## 🚀 Backend API Endpoints 🚀

`POST /predict`

Accepts: A POST request with a JSON payload containing base64-encoded image data.
```bash
{
  "image": "data:image/jpeg;base64,..."
}
```
#### Response Example:
```bash
{
  "predictions": [
    {"class": "Hello", "confidence": 0.91},
    {"class": "Thanks", "confidence": 0.06},
    {"class": "Yes", "confidence": 0.03}
  ],
  "hand_detected": true,
  "inference_time_ms": 34.12,
  "timestamp": 1720000000.0,
  "hand_box": {
    "xmin": 120,
    "ymin": 80,
    "xmax": 300,
    "ymax": 260
  }
}
```

The `hand_box` field provides the bounding box coordinates of the detected hand, if available.
## 📈 Training the Model 📊

Interested in retraining the model or training it with new data?
#### 1.Prepare your dataset:
Organize your image dataset in a structure compatible with `ImageDataGenerator` in `train.py` (e.g., data/train and data/validation directories, each containing subdirectories for every distinct class/letter).

#### 2.Execute the training script:
```
python train.py
```

Upon completion, a new model will be saved as `sign_language_model_gray.h5`. Remember to update `inference22.py` to point to this new model if you intend to deploy it.
## 🔗 Usage: Frontend Integration 🔗

This repository provides the robust backend and training infrastructure. To create a complete, interactive real-time application, you'll typically develop a separate frontend (e.g., using HTML/CSS/JavaScript, React, or a mobile development framework). This frontend would:

1.Access the user's webcam feed.

2.Capture video frames continuously.

3.Send these frames (or pre-processed image data) to the `/predict` endpoint of your running `inference22.py` backend.

4.Receive the predicted Malayalam character and dynamically display it to the user.
## 🤝 Contributing 🤝

We welcome contributions of all kinds! If you have ideas for enhancements, new features, or discover any bugs, please don't hesitate to open an issue or submit a pull request. Your input helps us improve!


## 🙏 Acknowledgements 🙏

A big thank you to the creators and maintainers of the following tools and resources that made this project possible:

* [MediaPipe](https://mediapipe.dev/): For robust hand detection capabilities.

* [TensorFlow](https://www.tensorflow.org/): The powerful machine learning framework.

* [MobileNetV2](https://www.google.com/search?q=https://keras.io/api/applications/mobilenet/%23mobilenetv2-function): The efficient deep learning model architecture.

* [OpenCV](https://opencv.org/): For image processing functionalities.

* [Flask](https://flask.palletsprojects.com/): For the lightweight web server.
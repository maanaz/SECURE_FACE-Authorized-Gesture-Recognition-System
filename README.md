# 🔐 Secure Face-Authorized Gesture Recognition System

A real-time computer vision and deep learning system that combines **face authentication** with **hand gesture recognition**. Gesture recognition is enabled only for an authorized user, with continuous face verification during operation.

## 🚀 Features

- Face detection and recognition using `face_recognition`
- Continuous verification of the authorized user
- Gesture recognition using transfer learning
- Three supported gestures:
  - 👍 Thumbs Up
  - ✌️ Peace
  - ✋ Open Palm
- Confidence-based gesture prediction
- Real-time camera processing using OpenCV
- Streamlit-based user interface
- Gesture recognition automatically disabled when authentication fails

## 🧠 Machine Learning Approach

The gesture classifier uses **MobileNetV2 with Transfer Learning**.

A MobileNetV2 model pretrained on ImageNet is used as a feature extractor. Its original classification layer is removed and replaced with a custom three-class classifier for the required gestures.

```text
Camera Frame
     │
     ▼
Face Authentication
     │
     ├── Unauthorized ──► Access Denied
     │
     ▼
Authorized User
     │
     ▼
MobileNetV2
     │
     ▼
Feature Extraction
     │
     ▼
Custom Classifier
     │
     ▼
Gesture + Confidence
```

The pretrained MobileNetV2 layers are frozen while the custom classification head is trained using a custom gesture dataset.

## 📊 Dataset

A custom dataset was collected using a camera with **455 images** across three classes:

```text
gesture_dataset/
├── open_palm/
├── peace/
└── thumbs_up/
```

The dataset was split into:

- Training: 364 images
- Validation: 91 images

The final model achieved approximately **97.8% validation accuracy**.

## 🔐 Continuous Face Verification

Authentication is not performed only once.

After the initial login, the system continuously checks the face in the live camera feed.

```text
Authorized User
      │
      ▼
Continuous Face Verification
      │
 ┌────┴─────┐
 │          │
Verified   Failed
 │          │
 ▼          ▼
Gesture    Access
Enabled    Denied
```

If the authorized user leaves or another person appears, gesture recognition is immediately disabled.

## 🛠️ Technologies

- **Python**
- **TensorFlow / Keras**
- **MobileNetV2**
- **OpenCV**
- **face_recognition**
- **NumPy**
- **Streamlit**

## 📁 Project Structure

```text
SECURE_FACE_AUTHORISED_GESTURE_RECOGNITION_SYSTEM/
│
├── app.py
├── gesture_model.keras
├── README.md
│
├── Images/
│   └── registered_user.jpg
│
└── gesture_dataset/
    ├── open_palm/
    ├── peace/
    └── thumbs_up/
```

## ⚙️ Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install tensorflow
pip install opencv-python
pip install face-recognition
pip install numpy
pip install streamlit
```

## ▶️ Running the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

## 📷 Camera Configuration

The application currently uses camera index `1`, configured for DroidCam:

```python
CAMERA_INDEX = 1
```

Change the value if a different camera index is required.

## 🔄 Application Workflow

1. The user opens the Streamlit application.
2. The camera captures the user's face.
3. The face is compared with registered face encodings.
4. Access is granted only if the user is recognized.
5. Gesture recognition becomes available.
6. The user's identity is continuously verified.
7. If verification fails, gesture recognition is disabled.
8. For an authorized user, MobileNetV2 predicts the gesture and confidence score.

## 🎯 Model Output

The classifier predicts one of three classes:

```text
open_palm
peace
thumbs_up
```

A confidence threshold is used to reject uncertain predictions.

## 🔮 Future Improvements

- Liveness detection to prevent photo-based authentication
- More gesture classes
- Larger and more diverse gesture dataset
- Data augmentation
- Fine-tuning MobileNetV2
- Database-based user management
- Improved real-time video streaming

## 👨‍💻 Author

**Maanaz K Antony**

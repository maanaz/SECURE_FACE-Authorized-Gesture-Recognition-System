import streamlit as st
import cv2
import numpy as np
import face_recognition as fr
import tensorflow as tf
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Secure Gesture Recognition",
    page_icon="🔐",
    layout="wide"
)


# =========================================================
# CONFIGURATION
# =========================================================

CAMERA_INDEX = 1

CLASS_NAMES = [
    "open_palm",
    "peace",
    "thumbs_up"
]

GESTURE_CONFIDENCE_THRESHOLD = 0.60

# Lower distance = stricter face matching
FACE_DISTANCE_THRESHOLD = 0.50


# =========================================================
# LOAD REGISTERED FACES
# =========================================================

@st.cache_resource
def load_face_data():

    path = "Images"

    images = []
    names = []

    for filename in os.listdir(path):

        image_path = os.path.join(path, filename)

        image = cv2.imread(image_path)

        if image is not None:
            images.append(image)
            names.append(filename.split(".")[0])

    encoded_faces = []
    valid_names = []

    for image, name in zip(images, names):

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        face_locations = fr.face_locations(
            rgb_image
        )

        if len(face_locations) == 0:
            continue

        encoding = fr.face_encodings(
            rgb_image,
            face_locations
        )[0]

        encoded_faces.append(encoding)
        valid_names.append(name)

    return encoded_faces, valid_names


# =========================================================
# LOAD GESTURE MODEL
# =========================================================

@st.cache_resource
def load_gesture_model():

    return tf.keras.models.load_model(
        "gesture_model.keras"
    )


known_face_encodings, known_names = load_face_data()

gesture_model = load_gesture_model()


# =========================================================
# FACE VERIFICATION
# =========================================================

def verify_face(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    rgb = np.ascontiguousarray(
        rgb,
        dtype=np.uint8
    )

    face_locations = fr.face_locations(rgb)

    face_encodings = fr.face_encodings(
        rgb,
        face_locations
    )

    # No face
    if len(face_encodings) == 0:
        return False, None, face_locations

    # More than one face
    if len(face_encodings) > 1:
        return False, None, face_locations

    face_encoding = face_encodings[0]

    face_distances = fr.face_distance(
        known_face_encodings,
        face_encoding
    )

    best_match_index = np.argmin(
        face_distances
    )

    best_distance = face_distances[best_match_index]

    # Require both:
    # 1. Face match
    # 2. Distance below threshold

    if best_distance <= FACE_DISTANCE_THRESHOLD:

        name = known_names[best_match_index]

        return True, name, face_locations

    return False, None, face_locations


# =========================================================
# GESTURE PREDICTION
# =========================================================

def predict_gesture(frame):

    image = cv2.resize(
        frame,
        (224, 224)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    # SAME preprocessing used during training
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    predictions = gesture_model.predict(
        image,
        verbose=0
    )

    predicted_index = np.argmax(
        predictions[0]
    )

    confidence = predictions[0][predicted_index]

    gesture = CLASS_NAMES[predicted_index]

    return gesture, confidence


# =========================================================
# SESSION STATE
# =========================================================

if "authorized" not in st.session_state:

    st.session_state.authorized = False

if "authorized_name" not in st.session_state:

    st.session_state.authorized_name = None


# =========================================================
# PAGE 1 — FACE AUTHENTICATION
# =========================================================

if not st.session_state.authorized:

    st.title("🔐 Face Authentication")

    st.write(
        "Authenticate yourself before accessing "
        "gesture recognition."
    )

    st.warning(
        "Gesture recognition is locked until "
        "your identity is verified."
    )

    start_auth = st.button(
        "▶ Start Authentication"
    )

    if start_auth:

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        while True:

            success, frame = camera.read()

            if not success:
                status_placeholder.error(
                    "Camera frame not received."
                )
                break

            authorized, name, face_locations = verify_face(
                frame
            )

            # Draw face boxes

            for location in face_locations:

                top, right, bottom, left = location

                if authorized:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    color,
                    2
                )

            if authorized:

                cv2.putText(
                    frame,
                    f"AUTHORIZED: {name}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                frame_placeholder.image(
                    cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    ),
                    channels="RGB"
                )

                status_placeholder.success(
                    f"Access granted. Welcome {name}!"
                )

                st.session_state.authorized = True
                st.session_state.authorized_name = name

                camera.release()

                st.rerun()

            else:

                cv2.putText(
                    frame,
                    "ACCESS DENIED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

                frame_placeholder.image(
                    cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    ),
                    channels="RGB"
                )

                status_placeholder.error(
                    "Unauthorized user"
                )

        camera.release()


# =========================================================
# PAGE 2 — CONTINUOUSLY VERIFIED GESTURE RECOGNITION
# =========================================================

else:

    st.title("🖐️ Secure Gesture Recognition")

    st.success(
        f"Authenticated User: "
        f"{st.session_state.authorized_name}"
    )

    st.info(
        "Your face is continuously verified. "
        "Gesture recognition will stop if another "
        "person is detected."
    )

    start_gesture = st.button(
        "▶ Start Secure Gesture Recognition"
    )

    if start_gesture:

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        while True:

            success, frame = camera.read()

            if not success:

                status_placeholder.error(
                    "Camera frame not received."
                )

                break

            # =========================================
            # CONTINUOUS FACE VERIFICATION
            # =========================================

            current_authorized, current_name, face_locations = verify_face(
                frame
            )

            # Draw face box

            for location in face_locations:

                top, right, bottom, left = location

                if current_authorized:

                    box_color = (0, 255, 0)

                else:

                    box_color = (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    box_color,
                    2
                )

            # =========================================
            # AUTHORIZED
            # =========================================

            if current_authorized:

                # Make sure it is the SAME registered user
                if current_name == st.session_state.authorized_name:

                    cv2.putText(
                        frame,
                        "IDENTITY VERIFIED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    # =================================
                    # GESTURE RECOGNITION
                    # =================================

                    gesture, confidence = predict_gesture(
                        frame
                    )

                    if confidence >= GESTURE_CONFIDENCE_THRESHOLD:

                        cv2.putText(
                            frame,
                            f"Gesture: {gesture}",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Confidence: {confidence * 100:.1f}%",
                            (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                        status_placeholder.success(
                            f"Gesture: {gesture} | "
                            f"Confidence: {confidence * 100:.1f}%"
                        )

                    else:

                        cv2.putText(
                            frame,
                            "Gesture uncertain",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 165, 255),
                            2
                        )

                        status_placeholder.warning(
                            "Gesture confidence is too low."
                        )

                else:

                    # Different registered user

                    cv2.putText(
                        frame,
                        "ACCESS DENIED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3
                    )

                    status_placeholder.error(
                        "Different user detected. "
                        "Gesture recognition disabled."
                    )

            # =========================================
            # UNAUTHORIZED / NO FACE
            # =========================================

            else:

                cv2.putText(
                    frame,
                    "ACCESS DENIED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    "GESTURE RECOGNITION DISABLED",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                status_placeholder.error(
                    "Identity verification failed. "
                    "Gesture recognition disabled."
                )

            # =========================================
            # DISPLAY FRAME
            # =========================================

            frame_placeholder.image(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                ),
                channels="RGB"
            )

        camera.release()

    # =============================================
    # LOGOUT
    # =============================================

    if st.button("🔒 Logout"):

        st.session_state.authorized = False
        st.session_state.authorized_name = None

        st.rerun()
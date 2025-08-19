import streamlit as st
import cv2
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from liveness import is_live_face

st.title("😊 Real-Time Emotion Detection with Anti-Spoofing")
st.markdown("Using **DeepFace (pre-trained)** + **Liveness Detection** for secure emotion recognition.")

class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.frame_skip = 5  # process every 5th frame
        self.count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.count += 1

        try:
            if self.count % self.frame_skip == 0:
                if is_live_face(img):
                    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                    dominant_emotion = result[0]['dominant_emotion']
                    cv2.putText(img, f"Emotion: {dominant_emotion}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "⚠ Spoof Detected!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            cv2.putText(img, f"Error: {str(e)}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img

# Streamlit WebRTC
webrtc_streamer(key="emotion", video_transformer_factory=EmotionDetector)

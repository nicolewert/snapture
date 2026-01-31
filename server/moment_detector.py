import logging
import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MomentDetector:
    """
    Detects moments and signals using MediaPipe Tasks (FaceLandmarker, GestureRecognizer).
    """
    def __init__(self, model_path: str = "models"):
        # Base options
        self.base_options = python.BaseOptions(model_asset_path=f"{model_path}/face_landmarker.task")
        
        # 1. Face Landmarker
        self.face_options = vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(self.face_options)
        
        # 2. Gesture Recognizer
        self.gesture_options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=f"{model_path}/gesture_recognizer.task"),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.gesture_recognizer = vision.GestureRecognizer.create_from_options(self.gesture_options)
        
        # State tracking
        self.last_process_time = 0
        
    def process_frame(self, base64_data: str) -> Dict[str, Any]:
        """
        Process a base64 encoded image frame.
        """
        try:
            # Decode image
            image_bytes = base64.b64decode(base64_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {}

            # Convert to MediaPipe Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            signals = {
                "smile_score": 0.0,
                "gesture": None,
                "pucker_score": 0.0,
                "surprise_score": 0.0,
                "brow_score": 0.0
            }

            # 1. Detect Gestures (Hands)
            gesture_result = self.gesture_recognizer.recognize(mp_image)
            if gesture_result.gestures:
                # Get top gesture from first hand
                top_gesture = gesture_result.gestures[0][0]
                if top_gesture.category_name != "None":
                    signals["gesture"] = top_gesture.category_name

            # 2. Detect Face Blendshapes (Expressions)
            face_result = self.face_landmarker.detect(mp_image)
            if face_result.face_blendshapes:
                blendshapes = face_result.face_blendshapes[0]
                
                # Manual extraction for specific emotions
                for category in blendshapes:
                    name = category.category_name
                    score = category.score * 100 # Normalize to 0-100
                    
                    if name in ["mouthSmileLeft", "mouthSmileRight"]:
                        signals["smile_score"] += (score / 2)
                    elif name == "mouthPucker":
                        signals["pucker_score"] = score
                    elif name == "browInnerUp":
                        signals["surprise_score"] = score
                    elif name in ["browDownLeft", "browDownRight"]:
                        signals["brow_score"] += (score / 2)
                
            return signals

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {}

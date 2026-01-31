import logging
import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class MomentType(str, Enum):
    """Types of detectable moments."""
    SMILE = "smile"
    SURPRISE = "surprise"
    WINK = "wink"
    GOOD_FRAMING = "good_framing"
    THUMBS_UP = "thumbs_up"
    PEACE_SIGN = "peace_sign"
    WAVE = "wave"
    POINT = "point"
    ARMS_UP = "arms_up"
    T_POSE = "t_pose"
    HANDS_ON_HIPS = "hands_on_hips"
    LEAN_IN = "lean_in"

@dataclass
class DetectedMoment:
    """Represents a detected moment with metadata."""
    type: MomentType
    confidence: float
    timestamp: float
    
    def to_dict(self):
        return {
            "type": self.type.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }

class MomentDetector:
    """
    Detects moments and signals using MediaPipe Tasks (FaceLandmarker, GestureRecognizer, Pose).
    """
    def __init__(self, model_path: str = None):
        # Get absolute path to models directory
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "models")
        
        # Ensure absolute path
        model_path = os.path.abspath(model_path)
        logger.info(f"Loading models from: {model_path}")
        
        # Verify models exist
        face_model = os.path.join(model_path, "face_landmarker.task")
        gesture_model = os.path.join(model_path, "gesture_recognizer.task")
        
        if not os.path.exists(face_model):
            raise FileNotFoundError(f"Face landmarker model not found at: {face_model}")
        if not os.path.exists(gesture_model):
            raise FileNotFoundError(f"Gesture recognizer model not found at: {gesture_model}")
        
        logger.info(f"✓ Found face_landmarker.task")
        logger.info(f"✓ Found gesture_recognizer.task")
        
        self.base_options = python.BaseOptions(model_asset_path=face_model)
        
        # 1. Face Landmarker
        try:
            self.face_options = vision.FaceLandmarkerOptions(
                base_options=self.base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(self.face_options)
            logger.info("✓ Face Landmarker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Face Landmarker: {e}")
            raise
        
        # 2. Gesture Recognizer
        try:
            self.gesture_options = vision.GestureRecognizerOptions(
                base_options=python.BaseOptions(model_asset_path=gesture_model),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.gesture_recognizer = vision.GestureRecognizer.create_from_options(self.gesture_options)
            logger.info("✓ Gesture Recognizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gesture Recognizer: {e}")
            raise
        
        # 3. Pose Detector
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7
        )
        
        # State tracking for pose consistency
        self.last_process_time = 0
        self.pose_frame_buffer = []  # Track pose across frames
        self.max_buffer = 5
        
        # Moment buffer for ranking
        self.moment_buffer: List[DetectedMoment] = []
        self.moment_weights = {
            MomentType.SMILE: 1.0,
            MomentType.THUMBS_UP: 1.2,
            MomentType.ARMS_UP: 1.1,
            MomentType.PEACE_SIGN: 0.9,
            MomentType.WAVE: 0.8,
            MomentType.SURPRISE: 0.95,
            MomentType.GOOD_FRAMING: 0.5,
            MomentType.WINK: 0.7,
            MomentType.POINT: 0.8,
            MomentType.T_POSE: 0.9,
            MomentType.HANDS_ON_HIPS: 0.85,
            MomentType.LEAN_IN: 0.6,
        }
        
    def process_frame(self, base64_data: str) -> Dict[str, Any]:
        """
        Process a base64 encoded image frame and extract signals.
        Returns blendshapes for expression detection and gesture data.
        """
        try:
            # Decode image
            image_bytes = base64.b64decode(base64_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {}

            # Get image dimensions for framing checks
            height, width = image.shape[:2]

            # Convert to MediaPipe Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            signals = {
                "smile_score": 0.0,
                "gesture": None,
                "pucker_score": 0.0,
                "surprise_score": 0.0,
                "wink_score": 0.0,
                "face_center_score": 0.0,
                "blendshapes": {},
                "pose": {}
            }

            # 1. Detect Gestures (Hands)
            gesture_result = self.gesture_recognizer.recognize(mp_image)
            if gesture_result.gestures:
                # Get top gesture from first hand
                top_gesture = gesture_result.gestures[0][0]
                if top_gesture.category_name != "None":
                    signals["gesture"] = top_gesture.category_name

            # 2. Detect Pose (Body)
            pose_signals = self._detect_pose(image_rgb)
            signals["pose"] = pose_signals

            # 3. Detect Face Blendshapes (Expressions)
            face_result = self.face_landmarker.detect(mp_image)
            if face_result.face_blendshapes:
                blendshapes = face_result.face_blendshapes[0]
                
                # Store all blendshapes for flexibility
                blendshapes_dict = {}
                
                # Manual extraction for specific emotions
                for category in blendshapes:
                    name = category.category_name
                    score = category.score * 100  # Normalize to 0-100
                    blendshapes_dict[name] = score
                    
                    if name in ["mouthSmileLeft", "mouthSmileRight"]:
                        signals["smile_score"] += (score / 2)
                    elif name == "mouthPucker":
                        signals["pucker_score"] = score
                    elif name in ["browInnerUpLeft", "browInnerUpRight"]:
                        signals["surprise_score"] += (score / 2)
                    elif name in ["eyeBlinkLeft", "eyeBlinkRight"]:
                        # Track wink (one eye closed, other open)
                        if name == "eyeBlinkLeft":
                            signals["wink_score"] = max(signals.get("wink_score", 0), score)
                        elif name == "eyeBlinkRight":
                            signals["wink_score"] = max(signals.get("wink_score", 0), score)
                
                signals["blendshapes"] = blendshapes_dict
                
                # 3. Check face framing (center of face in rule of thirds)
                if face_result.face_landmarks:
                    landmarks = face_result.face_landmarks[0]
                    if landmarks:
                        # Get face center from landmarks
                        face_x = sum(lm.x for lm in landmarks) / len(landmarks)
                        face_y = sum(lm.y for lm in landmarks) / len(landmarks)
                        
                        # Rule of thirds: center should be in middle 1/3
                        x_in_center = 0.33 < face_x < 0.67
                        y_in_center = 0.33 < face_y < 0.67
                        
                        if x_in_center and y_in_center:
                            signals["face_center_score"] = 1.0
                        elif x_in_center or y_in_center:
                            signals["face_center_score"] = 0.5
                        else:
                            signals["face_center_score"] = 0.0
                
            return signals

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {}
    
    def _detect_pose(self, image_rgb: np.ndarray) -> Dict[str, float]:
        """
        Detect body pose using MediaPipe Pose.
        Returns pose signals: arms_up, t_pose, hands_on_hips, lean_in.
        """
        pose_signals = {
            "arms_up": 0.0,
            "t_pose": 0.0,
            "hands_on_hips": 0.0,
            "lean_in": 0.0,
        }
        
        try:
            # Process with pose detector
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                return pose_signals
            
            landmarks = results.pose_landmarks.landmark
            
            # Key landmark indices (MediaPipe Pose)
            NOSE = 0
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_ELBOW = 13
            RIGHT_ELBOW = 14
            LEFT_WRIST = 15
            RIGHT_WRIST = 16
            LEFT_HIP = 23
            RIGHT_HIP = 24
            
            # Get positions
            left_wrist = landmarks[LEFT_WRIST]
            right_wrist = landmarks[RIGHT_WRIST]
            left_shoulder = landmarks[LEFT_SHOULDER]
            right_shoulder = landmarks[RIGHT_SHOULDER]
            left_hip = landmarks[LEFT_HIP]
            right_hip = landmarks[RIGHT_HIP]
            
            # Only process if confidence is high
            if all(lm.visibility > 0.5 for lm in [left_wrist, right_wrist, left_shoulder, right_shoulder]):
                
                # 1. ARMS_UP: Both wrists above shoulders
                left_wrist_above = left_wrist.y < left_shoulder.y
                right_wrist_above = right_wrist.y < right_shoulder.y
                
                if left_wrist_above and right_wrist_above:
                    # Check how much above (confidence)
                    avg_distance = ((left_shoulder.y - left_wrist.y) + 
                                   (right_shoulder.y - right_wrist.y)) / 2
                    pose_signals["arms_up"] = min(1.0, avg_distance * 3)
                
                # 2. T_POSE: Arms extended horizontally
                # Check if elbows and wrists are at similar y-level (horizontal)
                left_arm_y = [landmarks[LEFT_ELBOW].y, left_wrist.y]
                right_arm_y = [landmarks[RIGHT_ELBOW].y, right_wrist.y]
                
                left_horizontal = max(left_arm_y) - min(left_arm_y) < 0.1
                right_horizontal = max(right_arm_y) - min(right_arm_y) < 0.1
                
                # And wrists should be away from body (x-distance)
                left_extension = abs(left_wrist.x - left_shoulder.x)
                right_extension = abs(right_wrist.x - right_shoulder.x)
                
                if (left_horizontal and right_horizontal and 
                    left_extension > 0.15 and right_extension > 0.15):
                    pose_signals["t_pose"] = 0.9
                
                # 3. HANDS_ON_HIPS: Wrists near hip landmarks
                if all(lm.visibility > 0.5 for lm in [left_hip, right_hip]):
                    left_hip_distance = np.sqrt(
                        (left_wrist.x - left_hip.x)**2 + 
                        (left_wrist.y - left_hip.y)**2
                    )
                    right_hip_distance = np.sqrt(
                        (right_wrist.x - right_hip.x)**2 + 
                        (right_wrist.y - right_hip.y)**2
                    )
                    
                    # If wrists are close to hips (normalized distance < 0.15)
                    if left_hip_distance < 0.15 and right_hip_distance < 0.15:
                        pose_signals["hands_on_hips"] = 0.9
                
                # 4. LEAN_IN: Shoulders closer to camera than baseline
                # This requires tracking baseline - for now, check if shoulders are forward
                # (y coordinate increasing means closer to camera in MediaPipe)
                shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
                # If shoulder y > 0.35, they're in lower part of frame (closer)
                if shoulder_avg_y > 0.4:
                    pose_signals["lean_in"] = min(1.0, (shoulder_avg_y - 0.35) * 2)
        
        except Exception as e:
            logger.error(f"Error detecting pose: {e}")
        
        return pose_signals
    
    def detect_moments(self, signals: Dict[str, Any]) -> List[DetectedMoment]:
        """
        Analyze signals and return any detected moments.
        Moments are high-confidence combinations of signals.
        """
        moments = []
        timestamp = datetime.now().timestamp()
        
        # Face expressions
        smile_score = signals.get("smile_score", 0.0)
        if smile_score > 60:
            moments.append(DetectedMoment(
                type=MomentType.SMILE,
                confidence=min(1.0, smile_score / 100),
                timestamp=timestamp
            ))
        
        surprise_score = signals.get("surprise_score", 0.0)
        if surprise_score > 50:
            moments.append(DetectedMoment(
                type=MomentType.SURPRISE,
                confidence=min(1.0, surprise_score / 100),
                timestamp=timestamp
            ))
        
        wink_score = signals.get("wink_score", 0.0)
        if wink_score > 60:
            moments.append(DetectedMoment(
                type=MomentType.WINK,
                confidence=min(1.0, wink_score / 100),
                timestamp=timestamp
            ))
        
        # Framing
        face_center = signals.get("face_center_score", 0.0)
        if face_center == 1.0:
            moments.append(DetectedMoment(
                type=MomentType.GOOD_FRAMING,
                confidence=face_center,
                timestamp=timestamp
            ))
        
        # Gestures
        gesture = signals.get("gesture")
        if gesture and gesture != "None":
            # Map gesture names to MomentTypes
            gesture_map = {
                "Thumb_Up": MomentType.THUMBS_UP,
                "Peace": MomentType.PEACE_SIGN,
                "Pointing_Up": MomentType.POINT,
                "Waving": MomentType.WAVE,
            }
            if gesture in gesture_map:
                moments.append(DetectedMoment(
                    type=gesture_map[gesture],
                    confidence=0.9,  # Gestures are high confidence when detected
                    timestamp=timestamp
                ))
        
        # Poses
        pose_data = signals.get("pose", {})
        if pose_data:
            if pose_data.get("arms_up", 0.0) > 0.5:
                moments.append(DetectedMoment(
                    type=MomentType.ARMS_UP,
                    confidence=min(1.0, pose_data["arms_up"]),
                    timestamp=timestamp
                ))
            
            if pose_data.get("t_pose", 0.0) > 0.7:
                moments.append(DetectedMoment(
                    type=MomentType.T_POSE,
                    confidence=pose_data["t_pose"],
                    timestamp=timestamp
                ))
            
            if pose_data.get("hands_on_hips", 0.0) > 0.7:
                moments.append(DetectedMoment(
                    type=MomentType.HANDS_ON_HIPS,
                    confidence=pose_data["hands_on_hips"],
                    timestamp=timestamp
                ))
        
        # Add to buffer
        self.moment_buffer.extend(moments)
        # Keep buffer size manageable (last 300 moments = ~10 sec at 30fps)
        if len(self.moment_buffer) > 300:
            self.moment_buffer = self.moment_buffer[-300:]
        
        return moments
    
    def get_best_moments(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Rank moments by weighted confidence and return top N.
        """
        if not self.moment_buffer:
            return []
        
        # Score each moment
        scored = []
        for moment in self.moment_buffer:
            weight = self.moment_weights.get(moment.type, 0.5)
            score = moment.confidence * weight
            scored.append((moment, score))
        
        # Sort by score and return top N
        scored.sort(key=lambda x: x[1], reverse=True)
        top_moments = [m[0] for m in scored[:top_n]]
        
        return [m.to_dict() for m in top_moments]
    
    def clear_buffer(self):
        """Clear the moment buffer (useful at end of session)."""
        self.moment_buffer = []

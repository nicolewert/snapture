import logging
import base64
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MomentDetector:
    """
    Detects moments and signals from video frames using MediaPipe.
    Analyzes facial expressions and body pose to compute metrics.
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Signal history for smoothing/energy calc
        self.prev_nose_y = None
        
    def process_frame(self, base64_data: str) -> Dict[str, Any]:
        """
        Process a base64 encoded image frame.
        Returns a dictionary of calculated signals.
        """
        try:
            # Decode image
            image_bytes = base64.b64decode(base64_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {}

            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            signals = {
                "smile_score": 0.0,
                "energy_score": 0.0,
                "is_centered": False
            }

            # Process Face
            face_results = self.face_mesh.process(image_rgb)
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0]
                signals["smile_score"] = self._calculate_smile(landmarks)
                
            # Process Pose (for energy/centering)
            # Optimization: distinct pose call might be heavy, maybe skip every other frame?
            # For now, run it.
            pose_results = self.pose.process(image_rgb)
            if pose_results.pose_landmarks:
                signals["energy_score"] = self._calculate_energy(pose_results.pose_landmarks)
                signals["is_centered"] = self._check_centering(pose_results.pose_landmarks)

            return signals

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {}

    def _calculate_smile(self, landmarks) -> float:
        """
        Calculate smile score based on mouth landmarks.
        Simple heuristic: distance between mouth corners vs mouth width.
        """
        # Lip corners: 61 and 291
        # Upper lip top: 13
        # Lower lip bottom: 14
        
        # Convert landmarks to numpy array for easier math if needed, 
        # but accessing .x .y directly is fast enough for simple distance
        
        left_corner = landmarks.landmark[61]
        right_corner = landmarks.landmark[291]
        
        # Simple width
        mouth_width = ((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)**0.5
        
        # This is a very basic heuristic; a "smile" often widens the mouth
        # A better one might compare to a neutral face, but we don't have calibration.
        # Let's return raw width for now, or normalize roughly.
        # Typical face width might be around 0.5 of frame? 
        # Let's just return the raw width as a score for now.
        return float(mouth_width) 

    def _calculate_energy(self, landmarks) -> float:
        """
        Calculate energy score based on movement.
        """
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        
        energy = 0.0
        if self.prev_nose_y is not None:
             # Vertical movement
            delta = abs(nose.y - self.prev_nose_y)
            # Amplify small movements
            energy = min(delta * 100, 1.0) 
            
        self.prev_nose_y = nose.y
        return float(energy)

    def _check_centering(self, landmarks) -> bool:
        """Check if user is roughly centered."""
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        return 0.4 < nose.x < 0.6

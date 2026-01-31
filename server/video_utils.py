import base64
import time
import cv2
import numpy as np
import logging
from collections import deque
from pathlib import Path
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class FrameBuffer:
    def __init__(self, max_duration_sec: int = 60, fps: int = 4):
        self.fps = fps
        self.max_frames = max_duration_sec * fps
        # Buffer stores (timestamp, base64_frame)
        self.buffer = deque(maxlen=self.max_frames)
    
    def add_frame(self, frame_b64: str):
        """Add a frame to the buffer with current timestamp."""
        self.buffer.append((time.time(), frame_b64))
    
    def get_frames(self, duration_sec: int = 10) -> List[str]:
        """Get frames from the last N seconds."""
        now = time.time()
        start_time = now - duration_sec
        
        frames = []
        for ts, frame in self.buffer:
            if ts >= start_time:
                frames.append(frame)
        return frames

    def clear(self):
        self.buffer.clear()

def downscale_frame(base64_frame: str, target_width: int = 640) -> str:
    """
    Downscale a base64 encoded JPEG image to a target width, maintaining aspect ratio.
    Returns base64 encoded JPEG.
    """
    try:
        # Decode base64 to bytes
        img_data = base64.b64decode(base64_frame)
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return base64_frame

        # Calculate new dimensions
        height, width = img.shape[:2]
        if width <= target_width:
            return base64_frame
            
        ratio = target_width / width
        new_height = int(height * ratio)
        
        # Resize
        resized = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Encode back to JPEG
        _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(buffer).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Error downscaling frame: {e}")
        return base64_frame

def create_video_clip(frames_b64: List[str], output_path: str, fps: int = 4) -> Optional[str]:
    """
    Create an MP4 video clip from a list of base64 frames.
    Returns the path to the created video file.
    """
    if not frames_b64:
        return None
        
    try:
        # Decode first frame to get dimensions
        first_frame_data = base64.b64decode(frames_b64[0])
        nparr = np.frombuffer(first_frame_data, np.uint8)
        first_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if first_frame is None:
            logger.error("Failed to decode first frame for clip")
            return None
            
        height, width = first_frame.shape[:2]
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize video writer
        # 'avc1' (H.264) is better for HTML5 browsers than 'mp4v'
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        except:
            # Fallback to mp4v if avc1 is not available
            logger.warning("avc1 codec not available, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_b64 in frames_b64:
            img_data = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                out.write(img)
                
        out.release()
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating video clip: {e}")
        return None

class VideoRecorder:
    def __init__(self, output_path: str, fps: int = 24):
        self.output_path = output_path
        self.fps = fps
        self.width = None
        self.height = None
        self.writer = None
        self.is_recording = False
        self._initialized = False
        
    def start(self):
        """Start recording to the output path."""
        try:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            self.is_recording = True
            self._initialized = False
            logger.info(f"Recording enabled, waiting for first frame: {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
    
    def _init_writer(self, width: int, height: int):
        """Initialize the video writer with detected dimensions."""
        self.width = width
        self.height = height
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        except:
            logger.warning("avc1 codec not available, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        self._initialized = True
        logger.info(f"Initialized video writer: {self.width}x{self.height} @ {self.fps}fps")
            
    def write_frame(self, frame_b64: str):
        """Write a base64 frame to the video file."""
        if not self.is_recording:
            return
            
        try:
            img_data = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Initialize writer on first frame with actual dimensions
                if not self._initialized:
                    height, width = img.shape[:2]
                    self._init_writer(width, height)
                
                # Resize if dimensions don't match (shouldn't happen normally)
                if img.shape[:2] != (self.height, self.width):
                    img = cv2.resize(img, (self.width, self.height))
                    
                if self.writer:
                    self.writer.write(img)
        except Exception as e:
            logger.error(f"Error writing frame to recording: {e}")
            
    def stop(self) -> Optional[str]:
        """Stop recording and release resources. Returns output path if successful."""
        self.is_recording = False
        if self.writer:
            self.writer.release()
            self.writer = None
            logger.info(f"Stopped recording: {self.output_path}")
            return self.output_path
        return None

def extract_subclip(source_path: str, start_sec: float, end_sec: float, output_path: str) -> Optional[str]:
    """
    Extract a subclip from a video file based on start and end timestamps.
    """
    try:
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            logger.error(f"Failed to open source video: {source_path}")
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        # Initialize writer
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        except:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1
            
        cap.release()
        out.release()
        
        logger.info(f"Created subclip: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error extracting subclip: {e}")
        return None

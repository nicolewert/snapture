"""
Snapture Backend Server

FastAPI WebSocket server that bridges frontend clients to Gemini Live API.
Handles bidirectional audio/video streaming with proper message formatting.
"""

import asyncio
import json
import logging
import os
import uuid
import time
from contextlib import asynccontextmanager
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


from gemini_client import GeminiLiveClient
from video_utils import FrameBuffer, downscale_frame, create_video_clip, VideoRecorder, extract_subclip
from moment_detector import MomentDetector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Server configuration
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CLIP_FPS = int(os.getenv("CLIP_FPS", "24"))
GEMINI_FRAME_STRIDE = int(os.getenv("GEMINI_FRAME_STRIDE", "2"))

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not set - Gemini integration will fail")


# Store active sessions
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, "ClientSession"] = {}

    def create_session(self, session_id: str, websocket: WebSocket) -> "ClientSession":
        session = ClientSession(session_id, websocket)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> "ClientSession | None":
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session {session_id} removed")


class ClientSession:
    """Represents a connected client session with Gemini integration."""

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.gemini_client: GeminiLiveClient | None = None
        self.detector = MomentDetector()
        self._connected = True
        self.gemini_client: GeminiLiveClient | None = None
        self._connected = True
        self.frame_buffer = FrameBuffer(max_duration_sec=60, fps=CLIP_FPS)
        self._video_frame_index = 0
        self.video_recorder: VideoRecorder | None = None
        self.recording_start_time: float | None = None
        self.active_clip_start: float | None = None
        self.bookmarks: list = []
        
        # Audio tracking
        self.audio_chunks_received = 0
        
        # State tracking for debouncing - INDEPENDENT timers per type
        self.last_gesture = None
        self.last_gesture_time = 0
        
        # Expression states with individual timers
        self.is_smiling = False
        self.last_smile_time = 0
        self.is_surprised = False
        self.last_surprise_time = 0
        self.is_puckering = False
        self.last_pucker_time = 0
        
        # Pose state tracking with individual timers
        self.arms_up = False
        self.last_arms_up_time = 0
        self.t_pose = False
        self.last_t_pose_time = 0
        self.hands_on_hips = False
        self.last_hands_on_hips_time = 0
        self.lean_in = False
        self.last_lean_in_time = 0

    async def setup_gemini(self, model: str = "gemini-2.5-flash-native-audio-latest"):
        """Initialize Gemini Live API connection."""
        if not GOOGLE_API_KEY:
            await self.send_error("GOOGLE_API_KEY not configured on server")
            return False

        try:
            self.gemini_client = GeminiLiveClient(GOOGLE_API_KEY, model)

            # Set up callbacks for Gemini responses
            self.gemini_client.on_audio = lambda data: asyncio.create_task(
                self.send_audio(data)
            )
            self.gemini_client.on_text = lambda text: asyncio.create_task(
                self.send_text(text)
            )
            self.gemini_client.on_interrupted = lambda: asyncio.create_task(
                self.send_interrupted()
            )
            self.gemini_client.on_turn_complete = lambda: asyncio.create_task(
                self.send_turn_complete()
            )
            
            # Tool Callbacks
            self.gemini_client.on_bookmark = lambda label, confidence: asyncio.create_task(
                self.send_bookmark(label, confidence)
            )
            self.gemini_client.on_overlay = lambda text, kind, duration: asyncio.create_task(
                self.send_overlay(text, kind, duration)
            )
            
            # Reconnect on disconnect
            self.gemini_client.on_disconnected = lambda: asyncio.create_task(
                self._handle_gemini_disconnect(model)
            )

            await self.gemini_client.connect()
            logger.info(f"Session {self.session_id}: Gemini connected")
            return True

        except Exception as e:
            logger.error(f"Session {self.session_id}: Failed to connect to Gemini: {e}")
            await self.send_error(f"Failed to connect to Gemini: {str(e)}")
            return False

    async def _handle_gemini_disconnect(self, model: str):
        """Handle Gemini disconnection and attempt to reconnect."""
        logger.warning(f"Session {self.session_id}: Gemini disconnected, attempting reconnect...")
        try:
            # Clean up old client
            if self.gemini_client:
                try:
                    await self.gemini_client.disconnect()
                except:
                    pass
            
            # Wait a moment before reconnecting
            await asyncio.sleep(1)
            
            # Reconnect
            if await self.setup_gemini(model):
                logger.info(f"Session {self.session_id}: Gemini reconnected successfully")
            else:
                logger.error(f"Session {self.session_id}: Failed to reconnect to Gemini")
        except Exception as e:
            logger.error(f"Session {self.session_id}: Error during reconnect: {e}")

    async def handle_message(self, message: dict):
        """Process incoming message from client."""
        msg_type = message.get("type")

        if msg_type == "setup":
            config = message.get("config", {})
            model = config.get("model", "gemini-2.5-flash-native-audio-latest")
            if await self.setup_gemini(model):
                await self.send_connected()

        elif msg_type == "audio":
            if self.gemini_client:
                self.audio_chunks_received += 1
                # Log every 100 chunks (~2-4 seconds of audio)
                if self.audio_chunks_received % 100 == 0:
                    logger.info(f"Audio chunks sent to Gemini: {self.audio_chunks_received}")
                await self.gemini_client.send_audio(message.get("data", ""))

        elif msg_type == "video":
            frame_b64 = message.get("data", "")
            video_data = message.get("data", "")
            
            # 1. Process with MomentDetector
            signals = self.detector.process_frame(video_data)
            now = time.time()
            
            # 2. Detect moments from signals
            moments = self.detector.detect_moments(signals)
            
            # Check signals for triggers and update state
            if signals:
                # --- Gesture Logic ---
                current_gesture = signals.get("gesture")
                
                # If gesture stopped (None or "None"), reset the last_gesture tracker
                if not current_gesture or current_gesture == "None":
                    if self.last_gesture is not None and (now - self.last_gesture_time > 0.3):
                        logger.info(f"Gesture ended: {self.last_gesture}")
                        self.last_gesture = None
                        self.last_gesture_time = now
                
                # If new gesture detected and enough time has passed since last gesture
                elif (current_gesture != "None" and 
                      (current_gesture != self.last_gesture or self.last_gesture is None) and 
                      (now - self.last_gesture_time > 0.8)):
                    
                    logger.info(f"Gesture detected: {current_gesture}")
                    self.last_gesture = current_gesture
                    self.last_gesture_time = now
                    
                    if self.gemini_client:
                         await self.gemini_client.send_text(f"[SYSTEM EVENT: User performed gesture '{current_gesture}']")

                # --- Expression Logic (Smile, Surprise, Pucker) - INDEPENDENT debouncing ---
                smile_score = signals.get("smile_score", 0)
                surprise_score = signals.get("surprise_score", 0)
                pucker_score = signals.get("pucker_score", 0)
                
                # Smile detection - 2s debounce
                is_currently_smiling = smile_score > 40
                if is_currently_smiling != self.is_smiling and (now - self.last_smile_time > 2.0):
                    self.is_smiling = is_currently_smiling
                    self.last_smile_time = now
                    if is_currently_smiling:
                         logger.info(f"Hype: Smile detected ({smile_score:.1f}%)")
                         if self.gemini_client:
                             await self.gemini_client.send_text("[SYSTEM EVENT: User is smiling broadly]")
                
                # Surprise detection - 2s debounce
                is_currently_surprised = surprise_score > 50
                if is_currently_surprised != self.is_surprised and (now - self.last_surprise_time > 2.0):
                    self.is_surprised = is_currently_surprised
                    self.last_surprise_time = now
                    if is_currently_surprised:
                         logger.info(f"Hype: Surprise detected ({surprise_score:.1f}%)")
                         if self.gemini_client:
                             await self.gemini_client.send_text("[SYSTEM EVENT: User looks surprised/amazed]")

                # Pucker detection - 2s debounce
                is_currently_puckering = pucker_score > 50
                if is_currently_puckering != self.is_puckering and (now - self.last_pucker_time > 2.0):
                    self.is_puckering = is_currently_puckering
                    self.last_pucker_time = now
                    if is_currently_puckering:
                         logger.info(f"Hype: Pucker face detected ({pucker_score:.1f}%)")
                         if self.gemini_client:
                             await self.gemini_client.send_text("[SYSTEM EVENT: User is making a silly pucker face]")
                
                # --- Pose Logic - INDEPENDENT debouncing ---
                pose_data = signals.get("pose", {})
                
                if pose_data:
                    # ARMS_UP detection - 1.5s debounce
                    arms_up_score = pose_data.get("arms_up", 0.0)
                    if arms_up_score > 0.5 and not self.arms_up and (now - self.last_arms_up_time > 1.5):
                        self.arms_up = True
                        self.last_arms_up_time = now
                        logger.info(f"Pose: Arms up detected ({arms_up_score:.1%})")
                        if self.gemini_client:
                            await self.gemini_client.send_text("[SYSTEM EVENT: User has arms raised up in celebration]")
                    elif arms_up_score <= 0.3 and self.arms_up:
                        self.arms_up = False
                    
                    # T_POSE detection - 1.5s debounce
                    t_pose_score = pose_data.get("t_pose", 0.0)
                    if t_pose_score > 0.7 and not self.t_pose and (now - self.last_t_pose_time > 1.5):
                        self.t_pose = True
                        self.last_t_pose_time = now
                        logger.info(f"Pose: T-pose detected")
                        if self.gemini_client:
                            await self.gemini_client.send_text("[SYSTEM EVENT: User is in T-pose with arms extended horizontally]")
                    elif t_pose_score <= 0.5 and self.t_pose:
                        self.t_pose = False
                    
                    # HANDS_ON_HIPS detection - 1.5s debounce
                    hips_score = pose_data.get("hands_on_hips", 0.0)
                    if hips_score > 0.7 and not self.hands_on_hips and (now - self.last_hands_on_hips_time > 1.5):
                        self.hands_on_hips = True
                        self.last_hands_on_hips_time = now
                        logger.info(f"Pose: Hands on hips detected")
                        if self.gemini_client:
                            await self.gemini_client.send_text("[SYSTEM EVENT: User has hands on hips in confident pose]")
                    elif hips_score <= 0.5 and self.hands_on_hips:
                        self.hands_on_hips = False
                    
                    # LEAN_IN detection - 1.5s debounce
                    lean_score = pose_data.get("lean_in", 0.0)
                    if lean_score > 0.6 and not self.lean_in and (now - self.last_lean_in_time > 1.5):
                        self.lean_in = True
                        self.last_lean_in_time = now
                        logger.info(f"Pose: Lean in detected ({lean_score:.1%})")
                        if self.gemini_client:
                            await self.gemini_client.send_text("[SYSTEM EVENT: User is leaning in toward the camera]")
                    elif lean_score <= 0.4 and self.lean_in:
                        self.lean_in = False

            # 2. Send to Gemini
            if self.gemini_client:
                # 1. Buffer full resolution frame
                self.frame_buffer.add_frame(frame_b64)
                
                # 2. Write to recorder if active
                if self.video_recorder and self.video_recorder.is_recording:
                    # Run in executor to avoid blocking
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self.video_recorder.write_frame, frame_b64)

                # 3. Forward every Nth frame to Gemini to reduce load
                self._video_frame_index += 1
                if self._video_frame_index % GEMINI_FRAME_STRIDE != 0:
                    return

                # 4. Downscale for Gemini (executor avoids blocking event loop)
                loop = asyncio.get_running_loop()
                small_frame = await loop.run_in_executor(
                    None, downscale_frame, frame_b64
                )
                
                # 5. Send to Gemini
                await self.gemini_client.send_video(small_frame)

        elif msg_type == "start_recording":
            timestamp = int(time.time())
            filename = f"full_session_{self.session_id}_{timestamp}.mp4"
            filepath = os.path.join("clips", filename)
            
            self.video_recorder = VideoRecorder(filepath, fps=CLIP_FPS) 
            self.bookmarks = [] # Reset bookmarks
            
            # Start recording
            await asyncio.to_thread(self.video_recorder.start)
            self.recording_start_time = time.time()
            logger.info(f"Session {self.session_id}: Started recording full session")

        elif msg_type == "stop_recording":
            if self.video_recorder and self.video_recorder.is_recording:
                output_path = await asyncio.to_thread(self.video_recorder.stop)
                self.recording_start_time = None
                
                if output_path:
                    # 1. Send full session clip
                    filename = os.path.basename(output_path)
                    clip_url = f"http://localhost:{SERVER_PORT}/clips/{filename}"
                    await self._send({
                        "type": "clip",
                        "url": clip_url,
                        "context": "Full Session Recording"
                    })
                    logger.info(f"Session {self.session_id}: Sent full session clip {clip_url}")
                    
                    # 2. Process Bookmarks
                    if self.bookmarks:
                        logger.info(f"Session {self.session_id}: Processing {len(self.bookmarks)} bookmarks")
                        loop = asyncio.get_running_loop()
                        
                        for i, bookmark in enumerate(self.bookmarks):
                            start = bookmark["start"]
                            end = bookmark["end"]
                            label = bookmark["label"]
                            
                            bookmark_filename = f"clip_{self.session_id}_{int(time.time())}_{i}.mp4"
                            bookmark_path = os.path.join("clips", bookmark_filename)
                            
                            subclip_path = await loop.run_in_executor(
                                None, extract_subclip, output_path, start, end, bookmark_path
                            )
                            
                            if subclip_path:
                                subclip_url = f"http://localhost:{SERVER_PORT}/clips/{bookmark_filename}"
                                await self._send({
                                    "type": "clip",
                                    "url": subclip_url,
                                    "context": f"{label} ({start:.1f}s - {end:.1f}s)"
                                })
                                logger.info(f"Session {self.session_id}: Sent bookmark clip {subclip_url}")
            
            self.video_recorder = None

        elif msg_type == "start_clip":
            if self.recording_start_time:
                current_time = time.time() - self.recording_start_time
                self.active_clip_start = current_time
                logger.info(f"Session {self.session_id}: Clip start marked at {current_time:.2f}s")

        elif msg_type == "stop_clip":
            if self.recording_start_time and self.active_clip_start is not None:
                current_time = time.time() - self.recording_start_time
                start_time = self.active_clip_start
                end_time = current_time
                
                logger.info(f"Session {self.session_id}: Clip stop marked ({start_time:.2f}s - {end_time:.2f}s)")
                self.active_clip_start = None
                
                # Generate clip immediately from frame buffer
                asyncio.create_task(self.generate_clip_from_buffer(start_time, end_time, "Manual Clip"))

        elif msg_type == "end":
            # Generate session summary
            best_moments = self.detector.get_best_moments(3)
            
            # Calculate suggested clip time (first to last moment)
            if self.detector.moment_buffer:
                timestamps = [m.timestamp for m in self.detector.moment_buffer]
                suggested_clip = {
                    "start": min(timestamps),
                    "end": max(timestamps)
                }
            else:
                suggested_clip = {"start": 0, "end": 0}
            
            # Send summary to client
            await self._send({
                "type": "session_summary",
                "data": {
                    "best_moments": best_moments,
                    "suggested_clip": suggested_clip
                }
            })
            
            logger.info(f"Session {self.session_id}: Summary sent with {len(best_moments)} best moments")
            
            await self.cleanup()

        else:
            logger.warning(f"Session {self.session_id}: Unknown message type: {msg_type}")

    async def send_connected(self):
        """Notify client of successful connection."""
        await self._send({"type": "connected", "sessionId": self.session_id})

    async def send_audio(self, data: str):
        """Send audio data to client."""
        await self._send({"type": "audio", "data": data})

    async def send_text(self, content: str):
        """Send text content to client."""
        # Handle dynamic clipping triggers
        if "[START_CLIP]" in content and self.recording_start_time:
            current_time = time.time() - self.recording_start_time
            self.active_clip_start = current_time
            logger.info(f"Session {self.session_id}: Clip start marked at {current_time:.2f}s")
            
        if "[END_CLIP]" in content and self.recording_start_time and self.active_clip_start is not None:
            current_time = time.time() - self.recording_start_time
            start_time = self.active_clip_start
            end_time = current_time
            
            logger.info(f"Session {self.session_id}: Clip bookmark added ({start_time:.2f}s - {end_time:.2f}s)")
            self.active_clip_start = None
            
            # Generate clip immediately from frame buffer
            asyncio.create_task(self.generate_clip_from_buffer(start_time, end_time, "Gemini Highlight"))
            
        # Legacy trigger fallback (optional, maybe remove if strictly following new plan)
        if "[CLIP]" in content and not "[START_CLIP]" in content and not "[END_CLIP]" in content:
             # Just mark a 15s clip ending now
             if self.recording_start_time:
                end_time = time.time() - self.recording_start_time
                start_time = max(0, end_time - 15)
                self.bookmarks.append({
                    "start": start_time,
                    "end": end_time,
                    "label": "Gemini Clip"
                })
                logger.info(f"Session {self.session_id}: Legacy clip trigger processed")

        await self._send({"type": "text", "content": content})

    async def generate_and_send_clip(self, context: str):
        """Generate a video clip and send it to the client."""
        try:
            timestamp = int(time.time())
            filename = f"clip_{self.session_id}_{timestamp}.mp4"
            filepath = os.path.join("clips", filename)
            
            # Get last 10 seconds of frames
            frames = self.frame_buffer.get_frames(duration_sec=10)
            if not frames:
                logger.warning(f"Session {self.session_id}: No frames to clip")
                return

            logger.info(f"Session {self.session_id}: Generating clip with {len(frames)} frames")
            
            # Generate clip in executor
            loop = asyncio.get_running_loop()
            result_path = await loop.run_in_executor(
                None, create_video_clip, frames, filepath, CLIP_FPS
            )
            
            if result_path:
                clip_url = f"http://localhost:{SERVER_PORT}/clips/{filename}"
                await self._send({
                    "type": "clip",
                    "url": clip_url,
                    "context": context
                })
                logger.info(f"Session {self.session_id}: Sent clip {clip_url}")
            else:
                logger.error(f"Session {self.session_id}: Failed to generate clip")
                
        except Exception as e:
            logger.error(f"Session {self.session_id}: Error generating clip: {e}")

    async def generate_clip_from_buffer(self, start_sec: float, end_sec: float, label: str):
        """Generate a clip from the frame buffer for a specific time range."""
        try:
            timestamp = int(time.time())
            filename = f"clip_{self.session_id}_{timestamp}.mp4"
            filepath = os.path.join("clips", filename)
            
            # Get frames from buffer for the specified duration
            # The frame buffer stores (timestamp, frame) tuples
            # We need to calculate frames based on recording timeline
            duration = end_sec - start_sec
            if duration <= 0:
                logger.warning(f"Session {self.session_id}: Invalid clip duration")
                return
                
            # Get recent frames that cover the clip duration
            frames = self.frame_buffer.get_frames(duration_sec=int(duration) + 2)
            if not frames:
                logger.warning(f"Session {self.session_id}: No frames available for clip")
                return

            logger.info(f"Session {self.session_id}: Generating clip with {len(frames)} frames ({start_sec:.1f}s - {end_sec:.1f}s)")
            
            # Generate clip in executor
            loop = asyncio.get_running_loop()
            result_path = await loop.run_in_executor(
                None, create_video_clip, frames, filepath, CLIP_FPS
            )
            
            if result_path:
                clip_url = f"http://localhost:{SERVER_PORT}/clips/{filename}"
                await self._send({
                    "type": "clip",
                    "url": clip_url,
                    "context": f"{label} ({start_sec:.1f}s - {end_sec:.1f}s)"
                })
                logger.info(f"Session {self.session_id}: Sent clip {clip_url}")
            else:
                logger.error(f"Session {self.session_id}: Failed to generate clip from buffer")
                
        except Exception as e:
            logger.error(f"Session {self.session_id}: Error generating clip from buffer: {e}")

    async def send_interrupted(self):
        """Notify client that model was interrupted."""
        await self._send({"type": "interrupted"})

    async def send_turn_complete(self):
        """Notify client that model turn is complete."""
        await self._send({"type": "turnComplete"})
        
    async def send_bookmark(self, label: str, confidence: float):
        """Notify client of a bookmark."""
        logger.info(f"Sending bookmark: {label}")
        await self._send({
            "type": "bookmark", 
            "label": label, 
            "confidence": confidence
        })

    async def send_overlay(self, text: str, kind: str, duration: float):
        """Notify client to show overlay."""
        logger.info(f"Sending overlay: {text}")
        await self._send({
            "type": "overlay", 
            "text": text, 
            "kind": kind,
            "duration": duration
        })

    async def send_error(self, message: str):
        """Send error message to client."""
        await self._send({"type": "error", "message": message})

    async def _send(self, message: dict):
        """Send message to WebSocket client."""
        if self._connected:
            try:
                await self.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Session {self.session_id}: Failed to send message: {e}")
                self._connected = False

    async def cleanup(self):
        """Clean up session resources."""
        self._connected = False
        if self.gemini_client:
            await self.gemini_client.disconnect()
            self.gemini_client = None
        self.frame_buffer.clear()
        logger.info(f"Session {self.session_id}: Cleaned up")


# Application setup
session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Ensure clips directory exists
    os.makedirs("clips", exist_ok=True)
    
    logger.info(f"Starting Snapture server on {SERVER_HOST}:{SERVER_PORT}")
    yield
    logger.info("Shutting down Snapture server")
    # Clean up all sessions
    for session_id in list(session_manager.sessions.keys()):
        session = session_manager.get_session(session_id)
        if session:
            await session.cleanup()
        session_manager.remove_session(session_id)


app = FastAPI(
    title="Snapture Backend",
    description="WebSocket server bridging frontend to Gemini Live API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure clips directory exists, then mount it (read-only)
os.makedirs("clips", exist_ok=True)
app.mount("/clips", StaticFiles(directory="clips"), name="clips")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gemini_configured": bool(GOOGLE_API_KEY),
        "active_sessions": len(session_manager.sessions),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for client connections."""
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    session = session_manager.create_session(session_id, websocket)
    logger.info(f"Client connected: {session_id}")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                await session.handle_message(message)
            except json.JSONDecodeError:
                logger.error(f"Session {session_id}: Invalid JSON received")
                await session.send_error("Invalid JSON format")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Session {session_id}: Error: {e}")
    finally:
        await session.cleanup()
        session_manager.remove_session(session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,
        log_level="info",
        reload_excludes=["clips", "*.mp4"],
    )

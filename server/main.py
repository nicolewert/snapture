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
from contextlib import asynccontextmanager
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from gemini_client import GeminiLiveClient
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


from time import time

class ClientSession:
    """Represents a connected client session with Gemini integration."""

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.gemini_client: GeminiLiveClient | None = None
        self.detector = MomentDetector()
        self._connected = True
        
        # State tracking for debouncing
        self.last_gesture = None
        self.last_gesture_time = 0
        self.is_smiling = False
        self.last_smile_change = 0

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

            await self.gemini_client.connect()
            logger.info(f"Session {self.session_id}: Gemini connected")
            return True

        except Exception as e:
            logger.error(f"Session {self.session_id}: Failed to connect to Gemini: {e}")
            await self.send_error(f"Failed to connect to Gemini: {str(e)}")
            return False

    async def handle_message(self, message: dict):
        """Process incoming message from client."""
        msg_type = message.get("type")

        if msg_type == "setup":
            config = message.get("config", {})
            model = config.get("model", "gemini-2.0-flash-live-001")
            if await self.setup_gemini(model):
                await self.send_connected()

        elif msg_type == "audio":
            if self.gemini_client:
                await self.gemini_client.send_audio(message.get("data", ""))

        elif msg_type == "video":
            video_data = message.get("data", "")
            
            # 1. Process with MomentDetector
            signals = self.detector.process_frame(video_data)
            now = time()
            
            # Check signals for triggers and update state
            if signals:
                # --- Gesture Logic ---
                current_gesture = signals.get("gesture")
                # Only trigger if gesture is valid, changed, and enough time passed (1s cooldown)
                if (current_gesture and 
                    current_gesture != "None" and 
                    current_gesture != self.last_gesture and 
                    (now - self.last_gesture_time > 1.0)):
                    
                    logger.info(f"Gesture detected: {current_gesture}")
                    self.last_gesture = current_gesture
                    self.last_gesture_time = now
                    
                    # Notify Gemini
                    if self.gemini_client:
                         await self.gemini_client.send_text(f"[SYSTEM EVENT: User performed gesture '{current_gesture}']")
                
                # Reset gesture state if it disappears for a while so we can re-trigger
                if not current_gesture or current_gesture == "None":
                    if now - self.last_gesture_time > 0.5:
                        self.last_gesture = None

                # --- Smile Logic ---
                smile_score = signals.get("smile_score", 0)
                
                # DEBUG: Log smile score if it's significant at all, to see what the range is
                if smile_score > 10:
                    logger.info(f"Smile Score detected: {smile_score:.2f}")

                is_currently_smiling = smile_score > 60
                
                if is_currently_smiling != self.is_smiling and (now - self.last_smile_change > 2.0):
                    self.is_smiling = is_currently_smiling
                    self.last_smile_change = now
                    
                    if is_currently_smiling:
                         logger.info("Smile started (Threshold passed)")
                         if self.gemini_client:
                             await self.gemini_client.send_text("[SYSTEM EVENT: User started smiling broadly]")
                    # else:
                    #      # Optional: Notify when smile stops? Might be too noisy.
                    #      pass

            # 2. Send to Gemini
            if self.gemini_client:
                await self.gemini_client.send_video(video_data)

        elif msg_type == "end":
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
        await self._send({"type": "text", "content": content})

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
        logger.info(f"Session {self.session_id}: Cleaned up")


# Application setup
session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
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
    )

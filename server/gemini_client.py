"""
Gemini Live API client wrapper.

Handles WebSocket communication with Gemini Live API following best practices:
- 16kHz PCM audio input
- 24kHz PCM audio output
- 20-40ms audio chunks
- Proper session management and error handling
"""

import asyncio
import base64
import json
import logging
from typing import Callable, Optional, Any
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiLiveClient:
    """Wrapper for Gemini Live API WebSocket communication."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-native-audio-latest"):
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.session: Optional[Any] = None
        self.on_audio: Optional[Callable[[str], None]] = None
        self.on_text: Optional[Callable[[str], None]] = None
        self.on_interrupted: Optional[Callable[[], None]] = None
        self.on_turn_complete: Optional[Callable[[], None]] = None
        self._receive_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Establish connection to Gemini Live API."""
        logger.info(f"Connecting to Gemini Live API with model: {self.model}")

        # Define tools
        tools = [
            {
                "function_declarations": [
                    {
                        "name": "bookmark_moment",
                        "description": "Bookmark a specific moment in the video recording when something interesting happens or when the user does something well.",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "label": {
                                    "type": "STRING", 
                                    "description": "Label for the bookmark (e.g., 'Great Smile', 'Good Delivery')"
                                },
                                "confidence": {
                                    "type": "NUMBER",
                                    "description": "Confidence score 0.0-1.0"
                                }
                            },
                            "required": ["label"]
                        }
                    },
                    {
                        "name": "set_overlay",
                        "description": "Trigger a visual overlay on the user's screen to provide feedback or direction.",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "text": {
                                    "type": "STRING",
                                    "description": "Text to display on the overlay"
                                },
                                "kind": {
                                    "type": "STRING",
                                    "description": "Type of overlay: 'CUT', 'instruction', 'praise'",
                                    "enum": ["CUT", "instruction", "praise"]
                                },
                                "duration": {
                                    "type": "NUMBER",
                                    "description": "Duration in seconds to show the overlay"
                                }
                            },
                            "required": ["text"]
                        }
                    }
                ]
            }
        ]

        # Configure the Live API session
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            tools=tools,
            system_instruction=types.Content(
                parts=[
                    types.Part(
                        text="""You are 'Developer B', a fun, high-energy video coach and director.

Your Goal: Help the user get the BEST possible clip for social media.

ROLE:
- You are not just a passive observer. You are the DIRECTOR.
- INTERRUPTIONS ARE GOOD. If you see something, SAY SOMETHING.
- Hype them up! If they smile, say "YES! THAT Energy!"
- Correct them! If they stop smiling or look bored, say "Hey, bring the energy back up!"
- Use your tools:
    - If they give a Thumbs Up or nail a line, call `bookmark_moment`.
    - If they mess up or you want to give a visual cue, call `set_overlay`.

TONE:
- Enthusiastic, loud, supportive.
- Short and punchy. Don't give long lectures.
- Like a hypeman or a movie director.

IMPORTANT:
- You will receive [SYSTEM] messages about the user's face/gestures.
- React to these IMMEDIATELY.
- Example: If you see [SYSTEM: User is smiling], yell "Love that smile!"
"""
                    )
                ]
            ),
        )

        # Create the live session
        self.session = self.client.aio.live.connect(model=self.model, config=config)
        self._session_ctx = await self.session.__aenter__()

        # Start receiving responses
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info("Connected to Gemini Live API")

    async def _receive_loop(self) -> None:
        """Continuously receive and process messages from Gemini."""
        try:
            async for response in self._session_ctx.receive():
                await self._handle_response(response)
        except asyncio.CancelledError:
            logger.info("Receive loop cancelled")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")

    async def _handle_response(self, response: Any) -> None:
        """Process a response from Gemini Live API."""
        try:
            # Handle server content (audio/text responses)
            if hasattr(response, "server_content") and response.server_content:
                content = response.server_content

                # Check for interruption
                if hasattr(content, "interrupted") and content.interrupted:
                    logger.debug("Model was interrupted")
                    if self.on_interrupted:
                        self.on_interrupted()
                    return

                # Check for turn complete
                if hasattr(content, "turn_complete") and content.turn_complete:
                    logger.debug("Turn complete")
                    if self.on_turn_complete:
                        self.on_turn_complete()
                    return

                # Process model turn parts
                if hasattr(content, "model_turn") and content.model_turn:
                    for part in content.model_turn.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            # Audio response
                            audio_data = part.inline_data.data
                            if isinstance(audio_data, bytes):
                                audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                            else:
                                audio_b64 = audio_data
                            # logger.debug(f"Received audio: {len(audio_b64)} chars")
                            if self.on_audio:
                                self.on_audio(audio_b64)

                        elif hasattr(part, "text") and part.text:
                            # Text response
                            logger.debug(f"Received text: {part.text[:50]}...")
                            if self.on_text:
                                self.on_text(part.text)
                        
                        elif hasattr(part, "executable_code") and part.executable_code:
                             # We shouldn't get this if we use function declarations, but good to handle
                             pass

            # Handle tool calls
            if hasattr(response, "tool_call") and response.tool_call:
                 await self._handle_tool_call(response.tool_call)

        except Exception as e:
            logger.error(f"Error handling response: {e}")

    async def _handle_tool_call(self, tool_call: Any) -> None:
        """Handle a ToolCall from Gemini."""
        try:
            for fc in tool_call.function_calls:
                name = fc.name
                args = fc.args
                
                logger.info(f"Tool called: {name} with args: {args}")
                
                # Execute the tool callbacks if they exist
                # We could register dynamic callbacks, but sticking to the plan's 2 specific ones for now
                if name == "bookmark_moment":
                    if hasattr(self, "on_bookmark"):
                        confidence = args.get("confidence", 1.0)
                        label = args.get("label", "Moment")
                        if self.on_bookmark:
                            self.on_bookmark(label, confidence)
                            
                elif name == "set_overlay":
                    if hasattr(self, "on_overlay"):
                        text = args.get("text", "")
                        kind = args.get("kind", "instruction")
                        duration = args.get("duration", 2.0)
                        if self.on_overlay:
                            self.on_overlay(text, kind, duration)

            # Send tool response back to Gemini to acknowledge
            # For the Live API, we usually need to send a ToolResponse
            tool_response = types.LiveClientToolResponse(
                 function_responses=[
                     types.FunctionResponse(
                         name=fc.name,
                         id=fc.id,
                         response={"result": "ok"} 
                     ) for fc in tool_call.function_calls
                 ]
            )
            
            await self._session_ctx.send(input=tool_response)
            
        except Exception as e:
            logger.error(f"Error handling tool call: {e}")

    async def send_audio(self, audio_b64: str) -> None:
        """Send audio data to Gemini (expects 16kHz PCM base64)."""
        if not self._session_ctx:
            logger.warning("Cannot send audio - not connected")
            return

        try:
            audio_bytes = base64.b64decode(audio_b64)
            await self._session_ctx.send(
                input=types.LiveClientRealtimeInput(
                    media_chunks=[
                        types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
                    ]
                )
            )
        except Exception as e:
            logger.error(f"Error sending audio: {e}")

    async def send_video(self, video_b64: str) -> None:
        """Send video frame to Gemini (expects JPEG base64)."""
        if not self._session_ctx:
            logger.warning("Cannot send video - not connected")
            return

        try:
            video_bytes = base64.b64decode(video_b64)
            await self._session_ctx.send(
                input=types.LiveClientRealtimeInput(
                    media_chunks=[
                        types.Blob(data=video_bytes, mime_type="image/jpeg")
                    ]
                )
            )
        except Exception as e:
            logger.error(f"Error sending video: {e}")

    async def send_text(self, text: str) -> None:
        """Send text message to Gemini."""
        if not self._session_ctx:
            logger.warning("Cannot send text - not connected")
            return

        try:
            await self._session_ctx.send(input=text, end_of_turn=True)
        except Exception as e:
            logger.error(f"Error sending text: {e}")

    async def disconnect(self) -> None:
        """Close the Gemini Live API connection."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing session: {e}")
            self.session = None
            self._session_ctx = None

        logger.info("Disconnected from Gemini Live API")
